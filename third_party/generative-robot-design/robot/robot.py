#!/usr/bin/env python3
# Author: Jonathan Külz
from typing import Tuple

import torch

from robot.kinematics import forward_kinematics
from robot_modules.robco_modules import base_offset
from transforms.batched import BatchTransform, Rotate
from transforms.parameterized import CONVENTION_IMPLEMENTATIONS, ParameterConvention, ParameterizedTransform


class Robot:

    def __init__(self, parameters: torch.Tensor, convention: ParameterConvention):
        self.convention = convention
        self.dtype = parameters.dtype
        self.device = parameters.device
        if not len(parameters.shape) <= 3:
            raise ValueError("The transforms parameter for a robot should have at most two batch levels: "
                             "n_robots and n_joints. Shape give: {}".format(parameters.shape))
        self._parameters = parameters

    @classmethod
    def from_transforms(cls, transforms: ParameterizedTransform):
        """Initializes the robot from a DH transformation object"""
        return cls(transforms.hardware_parameters, transforms.convention)

    @property
    def b(self) -> Tuple[int, ...]:
        """Returns the (levels of) batch size(s)"""
        return self._parameters.shape[:-1]

    @property
    def base_offset(self) -> torch.Tensor:
        """Returns the base offset of the robot"""
        if self.convention is ParameterConvention.ROBCO:
            return base_offset.to(self._parameters)
        return torch.eye(4, dtype=self.dtype, device=self.device)

    @property
    def dof(self) -> int:
        """Returns the number of degrees of freedom of the robot"""
        return self._parameters.shape[-2]

    @property
    def zero_joint_angles(self) -> torch.Tensor:
        """Returns a tensor of zeros with shape (b x nj)"""
        return torch.zeros(*self.b, dtype=self.dtype, device=self.device)

    def fk(self, q: torch.Tensor, full: bool = False):
        """Compute forward kinematics.

        The robot is stateless — ``q`` is not stored; each call is independent.

        Args:
            q: Joint angles in radians.
                * Shape ``(b, n_joints)`` — one configuration per robot.
                * Shape ``(b, gpr, n_joints)`` — multiple configurations per robot; hardware parameters are broadcast
                across the gpr dimension.
            full: If ``False`` (default), return only the end-effector transform, shape ``(*b, 4, 4)``.
                If ``True``, return all joint frames, shape ``(*b, n_joints, 4, 4)``.

        Returns:
            :class:`~transforms.batched.BatchTransform` including the base offset.
        """
        fk = forward_kinematics(self.transforms(q), full=full, from_robot=True)
        shape = fk.get_matrix().shape
        return BatchTransform(matrix=self.base_offset.expand(shape)).compose(fk)

    def joint_frames(self, q: torch.Tensor, eef: bool = False) -> BatchTransform:
        """
        Returns (b x nj) 4x4 homogeneous transformation matrices, each one representing the transformation from world
        frame to a joint frame, aligned with that joints current orientation (frame after joint rotation).

        For MDH parameters, this is easy: the joint-frame FK gives us what we want, because the frame references are
            just after the joint rotation.
        For DH parameters, it's harder: The DH frames are s.t. the joint rotation is applied as the first operation. To
            get the joint frames for an n-dof robot, we do the following: First, append a "world" frame to the front of
            the frame-FK. Then, apply the individual joint rotations to all frames, and discard the last frame, which is
            the TCP. The remaining frames are the joint frames.

        :param q: The joint angles for which the joint frames shall be returned
        :param eef: If true, will also return the end-effector frame. This is coincident with the last frame for the MDH
            convention, but differs for the DH convention.
        """
        if self.convention is ParameterConvention.MDH:
            frames = self.fk(q, full=True)
            if eef:
                frames = frames.stack(BatchTransform(frames.get_matrix()[..., -1, None, :, :]), dim=-3)
            return frames

        dims = [1] * q.dim()
        mat = torch.concat([
            self.base_offset.view(*dims, 4, 4).repeat(*q.shape[:-1], 1, 1, 1),
            self.fk(q, full=True).get_matrix()
        ], dim=len(dims) - 1)
        frame_transforms = self.transforms(q)
        if eef:
            mat = mat.contiguous()
            q = torch.concat([q, torch.zeros([*q.shape[:-1], 1], device=q.device)], dim=-1)
        else:
            mat = mat[..., :-1, :, :].contiguous()

        joint_rotations = Rotate.axis_angle(q, axis='Z')
        if self.convention is ParameterConvention.DH:
            return BatchTransform(mat).compose(joint_rotations)
        elif self.convention is ParameterConvention.ROBCO:
            if eef:
                proximals = BatchTransform(torch.concat([
                    frame_transforms.proximal_transform(),
                    torch.eye(4, dtype=frame_transforms.dtype, device=frame_transforms.device).view(*dims, 4, 4).repeat(
                        *q.shape[:-1], 1, 1, 1)], dim=len(dims) - 1))
            else:
                proximals = BatchTransform(frame_transforms.proximal_transform())
            return BatchTransform(mat).compose(proximals).compose(joint_rotations)
        else:
            raise ValueError("Unknown convention")

    def transforms(self, q: torch.Tensor) -> ParameterizedTransform:
        """Returns the parameterized transform with the desired joint configuration q."""
        parameter_class = CONVENTION_IMPLEMENTATIONS[self.convention]
        parameters = self._parameters
        if len(q.shape) == 3:  # assumes the second dimension is a batch of configurations for every robot
            nr, gpr, nj = q.shape
            parameters = parameters.unsqueeze(1).repeat(1, gpr, 1, 1)
            q = q.view(nr, gpr, nj, 1)
        else:
            assert len(q.shape) == 2, "Not implemented"
            if len(parameters.shape) == 2:  # only one robot but various joint configurations
                gpr, nj = q.shape
                parameters = parameters.unsqueeze(0).repeat(gpr, 1, 1)
                q = q.view(gpr, nj, 1)
            elif len(parameters.shape) == 3:  # multiple robots, only one joint configurations for each
                nr, nj = q.shape
                q = q.view(nr, nj, 1)
        return parameter_class(torch.concat([parameters, q], dim=-1))
