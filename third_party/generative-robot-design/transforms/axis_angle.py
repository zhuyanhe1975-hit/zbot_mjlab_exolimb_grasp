#!/usr/bin/env python3
# Author: Xinyu Chen, Jonathan Külz
# Date: 29.11.24
import torch


def batch_rotation_matrices_to_axis_angle(R: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Converts a 3x3 rotation matrices to a axis-angle vectors.

    Parameters:
    - R (BatchTransformation): A batched tensor of shape (..., 3, 3) where each 3x3 submatrix is a rotation matrix.
    - epsilon (float): Smaller R entries are considered zero.

    Returns:
    BatchTransformation: A tensor of shape (..., 3) where each column is an axis-angle vector.
    """
    assert R.shape[-2:] == (3, 3), f"Rotation matrix input must have the shape (..., 3, 3), has {R.shape}."
    R[torch.abs(R) < epsilon] = 0

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = torch.acos((trace - 1) / 2)

    # if trace->3, acos->0. There are numerical issues causing nan value
    if torch.any((torch.abs(trace - 3) < epsilon)):
        angle[torch.abs(trace - 3) < epsilon] = 0
    # if trace->-1, acos->-pi.
    if torch.any((torch.abs(trace - (-1)) < epsilon)):
        angle[torch.abs(trace - (-1)) < epsilon] = torch.pi

    # Avoid division by zero by setting a small epsilon for angles near zero
    sin_angle = torch.sin(angle).clamp(min=epsilon)

    # Compute the rotation axis for each matrix
    # By definition, the rotation axis is the eigenvector corresponding to the eigenvalue 1
    axis_1 = torch.stack([
        (R[..., 2, 1] - R[..., 1, 2]) / (2 * sin_angle),
        (R[..., 0, 2] - R[..., 2, 0]) / (2 * sin_angle),
        (R[..., 1, 0] - R[..., 0, 1]) / (2 * sin_angle)
    ], dim=-1)

    # The above computation does not work when R is symmetric, so we compute axis_2 in this case.
    I = torch.eye(3, device=R.device)
    for _ in range(len(R.shape[:-2])):
        I = I.unsqueeze(0)
    D, P = torch.linalg.eig(R - I)
    del I
    vector_idx = torch.argmax((torch.abs(D.real) < epsilon).to(torch.int8), dim=-1, keepdim=True)
    axis_2 = torch.gather(P.real, -1, index=vector_idx.repeat_interleave(repeats=3, dim=-1)[..., None]).squeeze()
    axis = torch.where(
        torch.logical_or(
            torch.all((R.transpose(-2, -1) - R) < epsilon, dim=(-2, -1)),
            torch.isclose(angle, torch.tensor([torch.pi], device=R.device))
        )[..., None], axis_2, axis_1)

    # Handle cases where the angle is close to zero by setting axis to zero
    axis = torch.where(angle[..., None] < epsilon, torch.zeros_like(axis), axis)
    if torch.any(axis.isnan()):
        raise ValueError('nan axis')

    # Return the axis-angle vector (axis * angle)
    return axis * angle[..., None]
