#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 19.06.24
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import torch

from robot_modules.robco_modules import ALL_MODULES, module_interpolate

from .batched import BatchTransform
from .dh_conventions import dh_to_homogeneous, homogeneous_to_dh, homogeneous_to_mdh, mdh_to_homogeneous


def abstract_method_not_implemented(*args, **kwargs):
    raise NotImplementedError("This method is abstract and not implemented.")


class ParameterConvention(Enum):
    """A parameter convention for a kinematic chain."""

    MDH = 1  # Modified Denavit-Hartenberg convention after Craig
    DH = 2  # Denavit-Hartenberg convention
    ROBCO = 3  # Robco Modules


class ParameterizedTransform(BatchTransform, ABC):
    """
    A batched transform which is made from a set of (differentiable) parameters.
    """

    convention: ParameterConvention  # Implement this in subclasses
    parameter_names: Tuple[str]  # Implement this in subclasses

    def __init__(self, parameters: torch.Tensor, requires_grad: bool = True):
        """Initialize a ParameterizedTransform."""
        self.requires_grad: bool = requires_grad
        self._parameters: torch.Tensor = parameters
        super().__init__(matrix=self._matrix_from_parameters())

    @property
    @abstractmethod
    def hardware_parameters(self):
        """Returns all parameters but the joint angle."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def theta(self):
        """Returns the joint angle."""
        raise NotImplementedError()

    @abstractmethod
    def update_joint_parameters(self, th: torch.Tensor, joint_types: np.array):
        """Updates the parameters of the parameters according to joint configuration th."""

    @abstractmethod
    def _matrix_from_parameters(self) -> torch.Tensor:
        """Returns the matrix representation of the transform"""

    def parameter_slice(self, index: Union[int, List[int], slice, torch.BoolTensor, torch.LongTensor]) -> ParameterizedTransform:
        """Returns an instance of this class with parameters sliced according to the index."""
        if isinstance(index, int):
            index = [index]
        if isinstance(index, Iterable) and not isinstance(index, torch.Tensor):
            index = [i if not isinstance(i, int) else slice(i, i + 1) for i in index]
        parameters = self.parameters[index]
        return self.__class__(parameters, requires_grad=self.requires_grad)

    def clone(self) -> ParameterizedTransform:
        """
        Deep copy of ParameterizedTransform object. All internal tensors are cloned individually.

        Returns:
            new ParameterizedTransform object.
        """
        parameters = self._parameters.detach().clone()
        return self.__class__(parameters, requires_grad=self.requires_grad)

    def get_matrix(self) -> torch.Tensor:
        """Overrides the get_matrix method to ensure it is recomputed from the paameters on every call"""
        return self._matrix_from_parameters()

    def stack(self, *others, dim=0):
        """
        Stacks multiple ParameterizedTransform objects together.

        Args:
            others: ParameterizedTransform objects to stack.
            dim: Dimension along which to stack.

        Returns:
            new ParameterizedTransform object.
        """
        transforms = [self] + list(others)
        if not all(t.b == self.b for t in transforms):
            raise ValueError("All transforms must have the same batch size.")
        parameters = torch.cat([t._parameters for t in transforms], dim=dim).to(self.device, dtype=self.dtype)
        return self.__class__(parameters, requires_grad=self.requires_grad)

    @property
    def parameters(self) -> torch.Tensor:
        """Returns the joint parameters"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: torch.Tensor):
        """Sets the joint parameters"""
        # These checks allow to set non-leaf parameters and is necessary for example for __getitem__
        if parameters.requires_grad is True and self.requires_grad is False:
            parameters = parameters.detach()
        elif self.requires_grad is True and parameters.requires_grad is False:
            parameters.requires_grad = True
        self._parameters = parameters

    @classmethod
    def get_num_parameters(cls) -> int:
        """Returns the number of parameters"""
        return len(cls.parameter_names)

    def __repr__(self) -> str:
        """Returns a string representation of the transform."""
        info = ', '.join([f'{name}={self.parameters[..., i]}' for i, name in enumerate(self.parameter_names)])
        return f"{self.__class__}({info})".replace('\n       ', '')


class DHLikeBase(ParameterizedTransform, ABC):
    """A super class for DH-like transformations."""

    parameter_names: Tuple[str, str, str, str] = ('alpha', 'a', 'd', 'theta')
    matrix_to_param: Callable[[torch.Tensor], torch.Tensor] = partial(abstract_method_not_implemented)
    param_to_matrix: Callable[[torch.Tensor], torch.Tensor] = partial(abstract_method_not_implemented)

    @classmethod
    def from_homogeneous(cls,
                         homogeneous: torch.Tensor,
                         requires_grad: bool = True,
                         ):
        """Creates a MDHTransform from a homogeneous transformation matrix."""
        parameters = cls.matrix_to_param(homogeneous)
        return cls(parameters, requires_grad)

    @property
    def alpha(self):
        """Returns the link twist."""
        return self.parameters[..., 0]

    @property
    def a(self):
        """Returns the link length."""
        return self.parameters[..., 1]

    @property
    def d(self):
        """Returns the joint offset."""
        return self.parameters[..., 2]

    @property
    def theta(self):
        """Returns the joint angle."""
        return self.parameters[..., 3]

    @property
    def hardware_parameters(self):
        """Returns the non-joint-angle parameters: ``[alpha, a, d]``.

        The full parameter tensor has four columns ``[alpha, a, d, theta]``.
        This property strips ``theta`` and returns only the three hardware
        (morphology) parameters that define the robot's geometry.

        For a 4-D parameter tensor of shape ``(b, gpr, n_joints, 4)`` the goals-per-robot
        dimension is collapsed (hardware params are identical across goals for the same
        robot), returning shape ``(b, n_joints, 3)``.  For 3-D or lower inputs, leading
        batch dimensions are preserved and the output shape is ``(..., n_joints, 3)``.
        """
        if len(self.parameters.shape) == 4:
            # The hardware parameters are all the same along the "goal per robot" dimension
            return self.parameters[:, 0, :, :3]
        return self.parameters[..., :3]

    def _matrix_from_parameters(self) -> torch.Tensor:
        """Returns the matrix representation of the transform. Redos the computation on every call"""
        return self.param_to_matrix(self.parameters)

    def update_joint_parameters(self, th: torch.Tensor, joint_types: np.array):
        """Updates the parameters of the parameters according to joint configuration th."""
        is_revolute = joint_types == 'revolute'
        is_prismatic = joint_types == 'prismatic'
        assert np.all(np.logical_xor(is_revolute, is_prismatic))
        self._parameters[is_prismatic.nonzero()][:, 2] = th[is_prismatic.nonzero()]
        self._parameters[is_revolute.nonzero()][:, 3] = th[is_revolute.nonzero()]


class DHTransform(DHLikeBase):
    """A transformation derived from Denavit-Hartenberg parameters."""

    convention: ParameterConvention = ParameterConvention.DH
    matrix_to_param: Callable[[torch.Tensor], torch.Tensor] = partial(homogeneous_to_dh)
    param_to_matrix: Callable[[torch.Tensor], torch.Tensor] = partial(dh_to_homogeneous)


class MDHTransform(DHLikeBase):
    """A transformation derived from modified Denavit-Hartenberg parameters."""

    convention: ParameterConvention = ParameterConvention.MDH
    matrix_to_param: Callable[[torch.Tensor], torch.Tensor] = partial(homogeneous_to_mdh)
    param_to_matrix: Callable[[torch.Tensor], torch.Tensor] = partial(mdh_to_homogeneous)


class ModuleTransform(ParameterizedTransform):

    convention = ParameterConvention.ROBCO
    parameter_names = ('weight_I', 'weight_II', 'weight_III', 'weight_IV', 'weight_V', 'theta')

    def __init__(self, parameters: torch.Tensor, requires_grad: bool = True):
        """Initialize a ParameterizedTransform."""
        super().__init__(parameters, requires_grad)

    @property
    def hardware_parameters(self):
        """Returns the non-joint angle parameters."""
        if len(self.parameters.shape) == 4:
            # The hardware parameters are all the same along the "goal per robot" dimension
            return self.parameters[:, 0, :, :-1]
        return self.parameters[..., :-1]

    @property
    def theta(self):
        return self.parameters[..., -1]

    def proximal_transform(self) -> torch.Tensor:
        """Returns the proximal transformation matrix (input to just before joint)."""
        n = self.parameters.shape[-1] - 1
        modular_repr = module_interpolate(ALL_MODULES, self._parameters[..., :-1].view(-1, n))
        return modular_repr.proximal_transform.view([*self.parameters.shape[:-1], 4, 4]).to(self.device, self.dtype)

    def update_joint_parameters(self, th: torch.Tensor, joint_types: np.array):
        self.parameters[..., -1] = th

    def _matrix_from_parameters(self) -> torch.Tensor:
        """Creates a batch transform from the present parameters."""
        all_hardware_parameters = self.parameters[..., :-1]
        if len(all_hardware_parameters.shape) == 4:
            b, nr, nj, n = all_hardware_parameters.shape
        else:
            b, nj, n = all_hardware_parameters.shape
            nr = None
        modular_repr = module_interpolate(ALL_MODULES, all_hardware_parameters.view(-1, n))
        matrix = modular_repr.transform(self.theta)
        if nr is not None:
            matrix = matrix.view(b, nr, nj, 4, 4)
        return matrix


CONVENTION_IMPLEMENTATIONS = {
    ParameterConvention.MDH: MDHTransform,
    ParameterConvention.DH: DHTransform,
    ParameterConvention.ROBCO: ModuleTransform
}
