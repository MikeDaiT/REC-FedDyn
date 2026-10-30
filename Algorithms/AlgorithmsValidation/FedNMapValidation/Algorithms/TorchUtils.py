"""Small PyTorch helpers shared by the torch-native algorithm methods."""

from collections.abc import Callable
from typing import TypeAlias

import torch

TorchObjective: TypeAlias = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
TorchProx: TypeAlias = Callable[[torch.Tensor, float, float, bool], torch.Tensor]

ADAM_LR = 1e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


class FullGradientAdam:
    """ADAM state for one local full-gradient subproblem solve."""

    def __init__(self, point: torch.Tensor, learning_rate: float = ADAM_LR,
                 betas: tuple[float, float] = ADAM_BETAS,
                 epsilon: float = ADAM_EPS) -> None:
        self.point = point.detach().clone()
        self.m = torch.zeros_like(point)
        self.v = torch.zeros_like(point)
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = epsilon
        self.step_number = 0

    @torch.no_grad()
    def step(self, gradient: torch.Tensor) -> torch.Tensor:
        self.step_number += 1
        self.m.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(gradient, gradient,
                                        value=1.0 - self.beta2)
        m_hat = self.m / (1.0 - self.beta1**self.step_number)
        v_hat = self.v / (1.0 - self.beta2**self.step_number)
        self.point.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)
        return self.point


def norm(point: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(point)


def synchronize(point: torch.Tensor) -> None:
    if point.is_cuda:
        torch.cuda.synchronize(point.device)


def statistics(point: torch.Tensor, functions: list[TorchObjective],
               prox: TorchProx, constant: float,
               stationarity_parameter: float) -> tuple[torch.Tensor, torch.Tensor]:
    value = point.new_zeros(())
    gradient = torch.zeros_like(point)
    for function in functions:
        local_value, local_gradient = function(point)
        value += local_value.detach() / len(functions)
        gradient += local_gradient.detach() / len(functions)
    value += prox(point, 1.0, constant, True)
    proximal_point = prox(
        point - stationarity_parameter * gradient,
        stationarity_parameter,
        constant,
        False,
    )
    mapping_norm = norm(point - proximal_point) / stationarity_parameter
    return value, mapping_norm
