#!/usr/bin/env python3
# Author: Xinyu Chen, Jonathan Külz
# Date: 29.11.24
from typing import Optional

import torch

from robot.robot import Robot
from transforms.axis_angle import batch_rotation_matrices_to_axis_angle
from utilities.structures import NetworkInput


def batch_kinematic_jacobian(robot: Robot,
                             q: torch.Tensor,
                             ) -> torch.Tensor:
    """
    Compute the geometric Jacobian wrt the base frame
    """
    nj = robot.dof
    frames = robot.joint_frames(q, eef=True)

    eef_pose = frames.eef
    J = torch.empty([*q.shape[:-1], 6, nj], device=robot.device)
    for jnt in range(nj):
        joint_axis = frames.rotation[..., jnt, :, -1]
        r_j = eef_pose.translation - frames.translation[..., jnt, :]

        J[..., :3, jnt] = torch.linalg.cross(joint_axis, r_j)
        J[..., 3:, jnt] = joint_axis
    return J


def compute_delta_se3(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """"
    Express the linear transformation from the current frame to the target frame in the world frame
    and then convert it into axis-angle representation

    Equivalent to Rotate(current.rotation) @ current.inverse() @ target @ Rotate(current.rotation.transpose(-2,-1))
    """
    return torch.concat([
        target[..., :3, 3] - current[..., :3, 3],
        batch_rotation_matrices_to_axis_angle(target[..., :3, :3] @ current[..., :3, :3].transpose(-2, -1))
    ], dim=-1)


def get_numeric_ik_mask(dof: int) -> torch.Tensor:
    """A default mask for numerical IK, considering only the number of degrees of freedom."""
    mask = torch.ones(6, dtype=torch.bool)  # Default, valid for six or more dof
    if dof <= 3:
        mask[3:] = False
    elif dof == 4:  # xy-plane with arbitrary z-rot
        mask[-1] = False
        mask[2] = False
    elif dof == 5:
        mask[-1] = False  # Ignore rotations around the z-axis
    return mask


@torch.no_grad()
def numerical_ik(data: NetworkInput,
                 robot: Robot,
                 q_init: Optional[torch.Tensor] = None,
                 tolerance: float = 1e-5,
                 step_size: float = 0.2,
                 max_iteration: int = 200,
                 reg: float = 1e-3,
                 error_weights: Optional[torch.Tensor] = None,
                 error_mask: Optional[torch.Tensor] = None,
                 ) -> torch.Tensor:
    """
    Performs numerical inverse kinematics with the Jacobian-transpose method.
    Args:
        data: The problem to be solved
        robot: A (batched) robot object
        q_init: If provided, will use these initial guesses
        tolerance: Convergence criterion
        step_size: Step size for the update
        max_iteration: Maximum number of iterations
        reg: Condition number for least-squares inversion of the jacobian matrix / damping for least-squares.
        error_weights: 6d array with weights for x, y, z, n_1, n_2, n_3 errors to determine the currently best solution.
        error_mask: If provided (needs to be binary!), will only solve IK to minimize the error for the True values.

    Notes:
        Errors are represented as [x, y, z, n_1, n_2, n_3] where n is the rotation axis.
        CPU is significantly faster than GPU for this operation due to the small matrix sizes.
    Returns:

    """
    original_device = robot.device
    if original_device.type != 'cpu':
        data = NetworkInput(unique_goals=data.unique_goals.cpu(), factor_goals=data.factor_goals, step=data.step)
        robot = Robot(robot._parameters.cpu(), robot.convention)
        if q_init is not None:
            q_init = q_init.cpu()
        if error_weights is not None:
            error_weights = error_weights.cpu()
        if error_mask is not None:
            error_mask = error_mask.cpu()

    if q_init is None:
        q_init = 2 * torch.pi * torch.rand((data.b, data.gpr, data.factor_goals, robot.dof)) - torch.pi
    else:
        if q_init.shape == (data.b, data.gpr, data.factor_goals, robot.dof):
            pass
        else:
            if q_init.shape[1:] == (data.b, data.gpr, data.factor_goals, robot.dof):
                n_heads = q_init.shape[0]
                q_merged = q_init.view(n_heads * data.b, data.gpr, data.factor_goals, robot.dof)
                env_merged = data.env.repeat(n_heads) if data.env is not None else None
                data_merged = NetworkInput(
                    unique_goals=data.unique_goals.repeat(n_heads, 1, 1),
                    env=env_merged,
                    factor_goals=data.factor_goals,
                    step=data.step,
                )
                robot_merged = Robot(robot._parameters.repeat(n_heads, 1, 1), robot.convention)
                result = numerical_ik(data_merged, robot_merged, q_merged, tolerance, step_size,
                                      max_iteration, reg, error_weights=error_weights, error_mask=error_mask)
                return result.view(n_heads, data.b, data.gpr, data.factor_goals, robot.dof).to(original_device)
            else:
                raise ValueError("Initial joint configuration has wrong shape")

    b, gpr, robots_per_problem, dof = q_init.shape
    q_init = q_init.view(-1, robot.dof)
    current_q: torch.Tensor = q_init.clone()
    best_q: torch.Tensor = q_init.clone()
    min_errors: torch.Tensor = torch.ones_like(best_q[..., 0]) * torch.inf
    goals: torch.Tensor = data.goals_transform.get_matrix().view(-1, 4, 4)
    if error_mask is None:
        error_mask = torch.ones(6, dtype=torch.bool, device=q_init.device)

    if error_weights is None:
        error_weights = torch.ones(6, device=q_init.device)
        error_weights[3:] = error_weights[3:] / torch.pi
    else:
        error_weights = error_weights.to(q_init)

    for i in range(max_iteration + 1):
        current_eef: torch.Tensor = robot.fk(current_q, full=False).get_matrix()
        err = compute_delta_se3(target=goals, current=current_eef)
        err_weighted = torch.norm(err * error_weights, dim=-1)

        improvement = err_weighted < min_errors
        min_errors = torch.where(improvement, err_weighted, min_errors)
        best_q = torch.where(improvement[..., None], current_q, best_q)

        if i == max_iteration:
            break  # Skip the last update, which is not processed anyway

        J = batch_kinematic_jacobian(robot, current_q)
        # compute damped least-squares inverse and delta_q

        dls = not error_mask.all()  # Default to damped least-squares only if masking is applied
        if not dls:
            try:
                delta_q = torch.linalg.lstsq(J, err, rcond=reg)[0].clip(-torch.pi, torch.pi)
            except torch.linalg.LinAlgError:
                dls = True
        if dls:
            # https://courses.cs.washington.edu/courses/cse599j/12sp/papers/DampedLeastSquares.pdf
            W = torch.diag_embed(error_mask.to(torch.float32) + 1e-6)
            J_sri = (J.transpose(-2, -1) @ W @ J + reg * torch.eye(J.shape[-1], device=J.device)).inverse() @ J.transpose(-2, -1) @ W
            delta_q = torch.einsum('bjx,bx->bj', J_sri, err)

        current_q = (torch.pi + current_q + step_size * delta_q) % (2 * torch.pi) - torch.pi

    return best_q.view(b, gpr, robots_per_problem, dof).to(original_device)
