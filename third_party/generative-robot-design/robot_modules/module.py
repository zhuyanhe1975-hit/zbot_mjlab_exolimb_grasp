#!/usr/bin/env python3
# Author: Jonathan Külz
import torch

from transforms.batched import Rotate


class Module:
    """
    Holds information about kinematic data for a robot module.

    Assumptions:
    - Modules are "unidirectional", i.e., there is a unique way to connect two of them
    - All coordinate systems within the module can be specified with just 3 operations. Specifically:
        - proximal_transform describes proximal frame to joint-in
        - joint-in to joint-out is a rotation of theta about their shared z-Axis
        - joint-out to distal is described by distal_transform
        - If two modules are in connection, their distal/proximal frames are equal
    """

    proximal_transform: torch.Tensor
    distal_transform: torch.Tensor

    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        """Complete proximal to distal frame transform"""
        pt = self.proximal_transform.view(*theta.shape, 4, 4).to(theta.device, theta.dtype)
        dt = self.distal_transform.view(*theta.shape, 4, 4).to(theta.device, theta.dtype)
        return pt @ Rotate.axis_angle(theta, 'z').get_matrix() @ dt


class Custom(Module):
    """A module that defines proximal and distal transform on an instance level"""

    def __init__(self, proximal: torch.Tensor, distal: torch.Tensor):
        self.proximal_transform = proximal
        self.distal_transform = distal
