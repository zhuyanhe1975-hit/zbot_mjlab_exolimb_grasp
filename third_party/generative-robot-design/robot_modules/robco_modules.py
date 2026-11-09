#!/usr/bin/env python3
# Author: Jonathan Külz
from typing import Iterable, Tuple

import torch

from robot_modules.module import Custom, Module
from transforms.batched import Rotate

base_offset = torch.tensor([  # Taking the default proximal transform into account already
    [0.0000, 1.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 1.0000, -0.0640],
    [1.0000, 0.0000, 0.0000, 0.1887],
    [0.0000, 0.0000, 0.0000, 1.0000]
], dtype=torch.float32)  # Module BaseDrive 0100
rot_joint_standard = torch.tensor([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
], dtype=torch.float32)
rot_joint_l = torch.tensor([
    [0, -1, 0],
    [0, 0, 1],
    [-1, 0, 0]
], dtype=torch.float32)  # R_x(-90) @ R_z(90)


class RobcoModule(Module):

    rotates_y: bool


class I(RobcoModule):
    """This module corresponds to a simple D86 joint."""

    proximal_transform: torch.Tensor = torch.tensor([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0.064],
        [0, 0, 0, 1]])
    distal_transform: torch.Tensor = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0.0555],
        [0, 0, 0, 1]
    ])
    names: Tuple[str] = ('D86',)
    rotates_y: bool = False


class II(I):
    """This module corresponds to a joint + short I-shaped extension"""
    distal_transform: torch.Tensor = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0.2055],
        [0, 0, 0, 1]
    ])
    names: Tuple[str] = ('D86', 'L150-I-80-80')
    rotates_y: bool = False


class III(I):
    """This module corresponds to a joint + long I-shaped extension"""
    distal_transform: torch.Tensor = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0.4055],
        [0, 0, 0, 1]
    ])
    names: Tuple[str] = ('D86', 'I86-350')
    rotates_y: bool = False


class IV(I):
    """This module corresponds to a joint + 165mm L-shaped extension"""
    distal_transform: torch.Tensor = torch.tensor([
        [0, -1,  0,  0],
        [0,  0,  1,  0.165],
        [-1,  0,  0,  0.1085],
        [0,  0,  0,  1]])
    names: Tuple[str] = ('D86', 'L86-165')
    rotates_y: bool = True


class V(I):
    """This module corresponds to a joint + 440mm L-shaped extension"""
    distal_transform: torch.Tensor = torch.tensor([
        [ 0, -1,  0,  0],
        [ 0,  0,  1,  0.44],
        [-1,  0,  0,  0.1085],
        [ 0,  0,  0,  1]])
    names: Tuple[str] = ('D86', 'L86-440')
    rotates_y: bool = True


def module_interpolate(modules: Iterable[RobcoModule], weights: torch.Tensor) -> Module:
    """
    Interpolates continuously between n different modules.

    This method would not work in general but is tailored to the limited available transformations in the available
    hardware. If you want to use this with arbitrary modules, you need a proper rotation interpolation, e.g., using
    quaternions.
    """
    modules = tuple(modules)
    b, n = weights.shape
    assert all(m.proximal_transform is modules[0].proximal_transform for m in modules[1:])  # Not implemented otherwise
    proximal = modules[0].proximal_transform.view(1, 4, 4).expand(b, 4, 4).to(weights.device)
    distal = torch.eye(4).view(1, 4, 4).expand(b, 4, 4).clone().to(weights.device)

    modules_distal = torch.stack([m.distal_transform.to(distal.device) for m in modules], dim=0).view(1, n, 4, 4).expand(b, n, 4, 4)
    rotate_extra = torch.tensor([m.rotates_y for m in modules], device=modules_distal.device, dtype=torch.bool)

    distal[..., :3, 3] = torch.einsum('bwn,bw->bn', modules_distal[..., :3, 3], weights)
    partial_post_rotation = Rotate.axis_angle(weights[:, rotate_extra].sum(dim=-1) * torch.pi / 2, 'y')
    distal[:, :3, :3] = rot_joint_standard.to(distal.device) @ partial_post_rotation.get_matrix()[:, :3, :3]

    return Custom(proximal, distal)


ALL_MODULES = (I, II, III, IV, V)
