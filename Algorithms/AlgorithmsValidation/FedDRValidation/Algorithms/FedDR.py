"""Experiment-local FedDR with the authors' released aggregation rule.

This file is isolated from ``Algorithms/FedDR.py``.  The server proximal step
and the project's full-gradient local solver are retained.  All current/stale
client reflection caches are aggregated using client sample-count weights.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


TorchObjective = Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


@dataclass(slots=True)
class AlgorithmResults:
    oracle_calling: torch.Tensor
    process_time: torch.Tensor
    iterate_point: torch.Tensor
    function_value: torch.Tensor
    subgradient: torch.Tensor
    index: torch.Tensor


def _synchronize(point: torch.Tensor) -> None:
    if point.is_cuda:
        torch.cuda.synchronize(point.device)


def _statistics(
    point: torch.Tensor,
    functions: list[TorchObjective],
    prox,
    constant: float,
    stationarity_parameter: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    mapping_norm = (
        torch.linalg.vector_norm(point - proximal_point)
        / stationarity_parameter
    )
    return value, mapping_norm


class FedDR:
    """Torch FedDR used only by the FEMNIST author-aggregation experiment."""

    def __init__(
        self,
        InitialPoint: torch.Tensor,
        IterationBudget: int,
        ProxFunction,
        ClientsNumber: int,
        SamplingNumber: int,
        SmoothL: float,
        ConvexMu: float,
        Constant: float,
        RandomSeed: int | None = None,
    ) -> None:
        if not isinstance(InitialPoint, torch.Tensor):
            raise TypeError("This experiment-local FedDR requires a torch tensor.")
        self.x_0 = InitialPoint.detach().clone()
        self.R = int(IterationBudget)
        self.prox = ProxFunction
        self.N = int(ClientsNumber)
        self.s = int(SamplingNumber)
        self.L = float(SmoothL)
        self.mu = float(ConvexMu)
        self.constant = float(Constant)
        self.seed = RandomSeed
        self.final_state: dict[str, object] = {}

    def local_approx_torch(
        self,
        proximal_center: torch.Tensor,
        initial_point: torch.Tensor,
        objective: TorchObjective,
        proximal_parameter: float,
        local_updates: int,
        local_learning_rate: float = 1e-3,
    ) -> torch.Tensor:
        point = initial_point.detach().clone()
        if self.L == 0.0 and self.mu == 0.0:
            for _ in range(local_updates):
                _, data_gradient = objective(point)
                gradient = data_gradient.detach() + (
                    point - proximal_center
                ) / proximal_parameter
                point -= local_learning_rate * gradient
            return point

        local_smoothness = self.L + 1.0 / proximal_parameter
        if self.mu > 0.0:
            local_strong_convexity = self.mu + 1.0 / proximal_parameter
        elif self.mu == 0.0:
            local_strong_convexity = 1.0 / proximal_parameter
        elif self.mu == -self.L and proximal_parameter < 1.0 / self.L:
            local_strong_convexity = 1.0 / proximal_parameter - self.L
        else:
            raise ValueError("Invalid FedDR local subproblem parameters.")
        momentum = (
            (local_smoothness**0.5 - local_strong_convexity**0.5)
            / (local_smoothness**0.5 + local_strong_convexity**0.5)
        )
        previous_gradient_point = point.clone()
        for _ in range(local_updates):
            _, data_gradient = objective(point)
            gradient = data_gradient.detach() + (
                point - proximal_center
            ) / proximal_parameter
            gradient_point = point - gradient / local_smoothness
            point = gradient_point + momentum * (
                gradient_point - previous_gradient_point
            )
            previous_gradient_point = gradient_point
        return point

    def go_torch(
        self,
        ProximalParameter: float,
        RelaxationParameter: float,
        LocalUpdates: int,
        Function: list[TorchObjective],
        ClientWeights: torch.Tensor | list[float] | None = None,
        StatisticsFunction: list[TorchObjective] | None = None,
        StationarityParameter: float | None = None,
        LocalLearningRate: float = 1e-3,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        gamma = float(ProximalParameter)
        relaxation = float(RelaxationParameter)
        local_updates = int(LocalUpdates)
        if gamma <= 0.0 or not 0.0 < relaxation <= 2.0:
            raise ValueError("Invalid FedDR proximal or relaxation parameter.")
        if local_updates <= 0 or LocalLearningRate <= 0.0:
            raise ValueError("LocalUpdates and LocalLearningRate must be positive.")
        if not 1 <= self.s <= self.N or len(Function) != self.N:
            raise ValueError("Invalid client sampling or objective count.")

        if ClientWeights is None:
            client_weights = self.x_0.new_ones(self.N)
        else:
            client_weights = torch.as_tensor(
                ClientWeights,
                dtype=self.x_0.dtype,
                device=self.x_0.device,
            )
        if client_weights.shape != (self.N,):
            raise ValueError("ClientWeights must contain one weight per client.")
        if not bool(torch.isfinite(client_weights).all()) or bool(
            (client_weights < 0.0).any()
        ):
            raise ValueError("ClientWeights must be finite and nonnegative.")
        weight_sum = client_weights.sum()
        if float(weight_sum) <= 0.0:
            raise ValueError("ClientWeights must have a positive sum.")
        client_weights = client_weights / weight_sum

        statistics_functions = (
            Function if StatisticsFunction is None else StatisticsFunction
        )
        stationarity_parameter = (
            gamma
            if StationarityParameter is None
            else float(StationarityParameter)
        )
        generator = random.Random(self.seed)

        if InitialState is None:
            initial = self.x_0.detach().clone()
            local_y = initial.repeat((self.N,) + (1,) * initial.ndim)
            local_x = torch.empty_like(local_y)
            local_hat = torch.empty_like(local_y)
            for client in range(self.N):
                local_x[client] = self.local_approx_torch(
                    local_y[client],
                    initial,
                    Function[client],
                    gamma,
                    local_updates,
                    LocalLearningRate,
                )
                local_hat[client] = 2.0 * local_x[client] - local_y[client]
            initialization_calls = self.N * local_updates
            server_point = initial.clone()
            server_average = torch.matmul(client_weights, local_hat)
            algorithm_time = 0.0
        else:
            local_y = InitialState["local_y"].detach().clone()
            local_x = InitialState["local_x"].detach().clone()
            local_hat = InitialState["local_hat"].detach().clone()
            server_point = InitialState["server_point"].detach().clone()
            server_average = InitialState["server_average"].detach().clone()
            algorithm_time = float(InitialState["algorithm_time"])
            initialization_calls = 0
            generator.setstate(InitialState["rng_state"])

        oracle_calling = torch.zeros(
            self.R, dtype=torch.int64, device=server_point.device
        )
        process_time = server_point.new_zeros(self.R)
        iterates = server_point.new_empty((self.R,) + tuple(server_point.shape))
        function_value = server_point.new_zeros(self.R)
        subgradient = server_point.new_zeros(self.R)
        index = torch.zeros(
            self.R, dtype=torch.bool, device=server_point.device
        )

        for round_index in range(self.R):
            _synchronize(server_point)
            round_start = time.perf_counter()
            for client in generator.sample(range(self.N), self.s):
                local_y[client] += relaxation * (
                    server_point - local_x[client]
                )
                local_x[client] = self.local_approx_torch(
                    local_y[client],
                    local_x[client],
                    Function[client],
                    gamma,
                    local_updates,
                    LocalLearningRate,
                )
                local_hat[client] = 2.0 * local_x[client] - local_y[client]

            # Match the released trainer: recompute a sample-count-weighted
            # average over every active or stale reflected-model cache.
            server_average = torch.matmul(client_weights, local_hat)

            server_point = self.prox(
                server_average, gamma, self.constant, False
            ).detach()
            _synchronize(server_point)
            algorithm_time += time.perf_counter() - round_start
            process_time[round_index] = algorithm_time
            oracle_calling[round_index] = self.s * local_updates
            if round_index == 0:
                oracle_calling[round_index] += initialization_calls

            iterates[round_index] = server_point
            if RecordStatistics:
                function_value[round_index], subgradient[round_index] = (
                    _statistics(
                        server_point,
                        statistics_functions,
                        self.prox,
                        self.constant,
                        stationarity_parameter,
                    )
                )

        self.final_state = {
            "local_y": local_y.clone(),
            "local_x": local_x.clone(),
            "local_hat": local_hat.clone(),
            "server_point": server_point.clone(),
            "server_average": server_average.clone(),
            "client_weights": client_weights.clone(),
            "algorithm_time": algorithm_time,
            "rng_state": generator.getstate(),
        }
        return AlgorithmResults(
            oracle_calling,
            process_time,
            iterates,
            function_value,
            subgradient,
            index,
        )
