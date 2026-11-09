#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 19.06.24
from typing import Union

import torch

from transforms.batched import BatchTransform


def forward_kinematics(frame_offsets: Union[torch.Tensor, BatchTransform],
                       full: bool = False,
                       from_robot: bool = False
                       ) -> BatchTransform:
    """
    Computes the forward kinematics for a given sequence of frame offsets.

    Args:
        frame_offsets: A tensor or Transform3d object containing the frame offsets. The returned forward kinematics
            will be computed for the coordinate systems defined by these relative offsets, i.e., if the frame offsets
            are based on MDH parameters, the FK coordinate frames will be placed directly AFTER the "physical" joint,
            if the frame offsets are based on DH parameters, the FK coordinate frames will be placed directly BEFORE
            the "physical" joint.
        full: If True, the full forward kinematics will be computed, i.e., the output has the same shape as the input.
            If False, only the forward kinematics for the last frame will be returned and the output has one dimension
            less.
        from_robot: This method once was supported as standalone, but isn't anymore. If this argument is not set, it
            will raise an error.

    Returns: A Transform3d object containing the forward kinematics for the given frame offsets.
    """
    if not from_robot:
        raise ValueError("The forward_kinematics method is not supported as standalone function anymore.")

    if isinstance(frame_offsets, torch.Tensor):
        frame_offsets = BatchTransform(frame_offsets)

    if frame_offsets.b is None:
        raise ValueError("Calling forward kinematics for a single frame is not supported.")

    nj = frame_offsets.b[-1]

    fk = BatchTransform(matrix=frame_offsets.get_matrix().clone())
    for jnt in range(1, nj):
        fk[..., jnt, :, :] = fk[..., jnt - 1, :, :] @ frame_offsets[..., jnt, :, :]

    if not full:
        return fk.eef
    return fk
