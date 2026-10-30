import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from Algorithms.Base_Optimizer import (
    Optimizer, FloatArray, Float, ProxFunction, parameter_norm,
)
from Algorithms.TorchUtils import (
    TorchObjective, statistics as torch_statistics, synchronize,
)

ObjectiveFunction: TypeAlias = Callable[[FloatArray], tuple[Float, FloatArray]]


@dataclass(slots=True)
class AlgorithmResults:
    oracle_calling: np.ndarray
    process_time: FloatArray
    iterate_point: FloatArray
    function_value: FloatArray
    subgradient: FloatArray
    index: np.ndarray


class Zhang(Optimizer):
    """Cached-direction partial adaptation of Zhang-Hu-Johansson."""

    def __init__(self, InitialPoint: FloatArray, IterationBudget: int, ProxFunction: ProxFunction,
                 ClientsNumber: int, SamplingNumber: int, SmoothL: Float, ConvexMu: Float,
                 Constant: Float,
                 RandomSeed: int | None = None) -> None:
        super().__init__(InitialPoint, IterationBudget, ProxFunction, ClientsNumber,
                         SamplingNumber, SmoothL, ConvexMu, Constant, RandomSeed)

    def _statistics(self, point: FloatArray, function: list[ObjectiveFunction],
                    stationarity_parameter: Float) -> tuple[Float, Float]:
        average_value = Float(0.0)
        average_gradient = np.zeros(self.parameter_shape, dtype=self.dtype)
        for local_function in function:
            local_value, local_gradient = local_function(point)
            average_value += local_value / self.N
            average_gradient += np.asarray(local_gradient, dtype=self.dtype) / self.N

        objective_value = average_value + self.prox(
            point, Float(1.0), self.constant, True
        )
        virtual_point = np.asarray(
            self.prox(point - stationarity_parameter * average_gradient,
                      stationarity_parameter, self.constant, False), dtype=self.dtype
        )
        stationarity_norm = parameter_norm(point - virtual_point) / stationarity_parameter
        return Float(objective_value), Float(stationarity_norm)

    def go(self, LocalLearningRate: Float, ServerLearningRate: Float, LocalUpdates: int,
           Function: list[ObjectiveFunction], StatisticsFunction: list[ObjectiveFunction] | None = None,
           StationarityParameter: Float | None = None,
           InitialState: dict[str, object] | None = None,
           RecordStatistics: bool = True) -> AlgorithmResults:
        eta = Float(LocalLearningRate)
        eta_g = Float(ServerLearningRate)
        tau = int(LocalUpdates)
        if eta <= 0 or eta_g <= 0 or tau <= 0:
            raise ValueError("Invalid Zhang algorithm parameters.")
        if not 1 <= self.s <= self.N:
            raise ValueError("SamplingNumber must lie between 1 and ClientsNumber.")

        proximal_parameter = eta * eta_g * tau
        statistics_function = Function if StatisticsFunction is None else StatisticsFunction
        stationarity_parameter = (
            proximal_parameter if StationarityParameter is None else Float(StationarityParameter)
        )
        rng = random.Random(self.seed)
        if InitialState is None:
            global_preprox = self.x_0.copy()
            # Match the authors' initialization: the first local round starts
            # directly from the supplied initial point.  The server proximal
            # step is first applied after that round's aggregation.
            global_postprox = global_preprox.copy()
            stored_direction = np.zeros(
                (self.N,) + self.parameter_shape, dtype=self.dtype
            )
            average_direction = np.zeros(self.parameter_shape, dtype=self.dtype)
            algorithm_time = 0.0
        else:
            global_preprox = np.asarray(
                InitialState["global_preprox"], dtype=self.dtype
            ).copy()
            global_postprox = np.asarray(
                InitialState["global_postprox"], dtype=self.dtype
            ).copy()
            stored_direction = np.asarray(
                InitialState["stored_direction"], dtype=self.dtype
            ).copy()
            average_direction = np.asarray(
                InitialState["average_direction"], dtype=self.dtype
            ).copy()
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        process_time = np.zeros(self.R, dtype=self.dtype)
        # Shape: (communication rounds, *parameter_shape).
        iterates = np.empty((self.R,) + self.parameter_shape, dtype=self.dtype)
        function_value = np.zeros(self.R, dtype=self.dtype)
        subgradient = np.zeros(self.R, dtype=self.dtype)
        index = np.zeros(self.R, dtype=np.bool_)
        oracle_calling = np.zeros(self.R, dtype=np.int64)
        for ite in range(self.R):
            round_start = time.perf_counter()
            round_oracle_calling = 0
            selected_clients = rng.sample(range(self.N), self.s)
            local_preprox_final = np.empty(
                (self.s,) + self.parameter_shape, dtype=self.dtype
            )
            local_gradient_average = np.empty(
                (self.s,) + self.parameter_shape, dtype=self.dtype
            )

            for position, client in enumerate(selected_clients):
                local_preprox = global_postprox.copy()
                local_postprox = global_postprox.copy()
                gradient_sum = np.zeros(self.parameter_shape, dtype=self.dtype)
                client_correction = average_direction - stored_direction[client]

                for local_step in range(tau):
                    _, local_gradient = Function[client](local_postprox)
                    local_gradient = np.asarray(local_gradient, dtype=self.dtype)
                    gradient_sum += local_gradient
                    local_preprox -= eta * (local_gradient + client_correction)
                    local_postprox = np.asarray(
                        self.prox(
                            local_preprox,
                            Float((local_step + 1) * eta),
                            self.constant,
                            False,
                        ),
                        dtype=self.dtype,
                    )
                    round_oracle_calling += 1

                local_preprox_final[position] = local_preprox
                local_gradient_average[position] = gradient_sum / tau

            average_local_preprox = np.mean(local_preprox_final, axis=0)
            next_global_preprox = global_postprox + eta_g * (average_local_preprox - global_postprox)

            # Only sampled clients refresh their local gradient direction.
            stored_direction[selected_clients] = local_gradient_average

            # The correction used next round is formed from the average of all
            # cached directions, including stale directions of inactive clients.
            average_direction = np.mean(stored_direction, axis=0)
            global_preprox = next_global_preprox
            global_postprox = np.asarray(
                self.prox(
                    global_preprox,
                    proximal_parameter,
                    self.constant,
                    False,
                ),
                dtype=self.dtype,
            )

            algorithm_time += time.perf_counter() - round_start
            process_time[ite] = algorithm_time
            iterates[ite] = global_postprox
            oracle_calling[ite] = round_oracle_calling

            if RecordStatistics:
                function_value[ite], subgradient[ite] = self._statistics(
                    global_postprox, statistics_function, stationarity_parameter
                )

        self.final_state = {
            "global_preprox": global_preprox.copy(),
            "global_postprox": global_postprox.copy(),
            "stored_direction": stored_direction.copy(),
            "average_direction": average_direction.copy(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }

        return AlgorithmResults(oracle_calling, process_time, iterates, function_value, subgradient, index)

    def go_torch(
        self, LocalLearningRate: float, ServerLearningRate: float,
        LocalUpdates: int, Function: list[TorchObjective],
        StatisticsFunction: list[TorchObjective] | None = None,
        StationarityParameter: float | None = None,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        if not isinstance(self.x_0, torch.Tensor):
            raise TypeError("go_torch requires a torch InitialPoint.")
        local_rate = float(LocalLearningRate)
        server_rate = float(ServerLearningRate)
        local_updates = int(LocalUpdates)
        proximal_parameter = local_rate * server_rate * local_updates
        statistics_function = Function if StatisticsFunction is None else StatisticsFunction
        stationarity_parameter = (
            proximal_parameter if StationarityParameter is None
            else float(StationarityParameter)
        )
        rng = random.Random(self.seed)

        if InitialState is None:
            global_preprox = self.x_0.detach().clone()
            # Match the authors' initialization: the first local round starts
            # directly from the supplied initial point.  The server proximal
            # step is first applied after that round's aggregation.
            global_postprox = global_preprox.clone()
            stored_direction = global_preprox.new_zeros(
                (self.N,) + tuple(global_preprox.shape)
            )
            average_direction = torch.zeros_like(global_preprox)
            algorithm_time = 0.0
        else:
            global_preprox = InitialState["global_preprox"].detach().clone()
            global_postprox = InitialState["global_postprox"].detach().clone()
            stored_direction = InitialState["stored_direction"].detach().clone()
            average_direction = InitialState["average_direction"].detach().clone()
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        device = global_postprox.device
        oracle_calling = torch.zeros(self.R, dtype=torch.int64, device=device)
        process_time = global_postprox.new_zeros(self.R)
        iterates = global_postprox.new_empty(
            (self.R,) + tuple(global_postprox.shape)
        )
        function_value = global_postprox.new_zeros(self.R)
        subgradient = global_postprox.new_zeros(self.R)
        index = torch.zeros(self.R, dtype=torch.bool, device=device)

        for round_index in range(self.R):
            synchronize(global_postprox)
            round_start = time.perf_counter()
            selected = rng.sample(range(self.N), self.s)
            local_final = global_postprox.new_empty(
                (self.s,) + tuple(global_postprox.shape)
            )
            local_gradient_average = torch.empty_like(local_final)

            for position, client in enumerate(selected):
                local_preprox = global_postprox.clone()
                local_postprox = global_postprox.clone()
                gradient_sum = torch.zeros_like(global_postprox)
                correction = average_direction - stored_direction[client]
                for local_step in range(local_updates):
                    _, local_gradient = Function[client](local_postprox)
                    local_gradient = local_gradient.detach()
                    gradient_sum += local_gradient
                    local_preprox -= local_rate * (local_gradient + correction)
                    local_postprox = self.prox(
                        local_preprox, (local_step + 1) * local_rate,
                        float(self.constant), False,
                    ).detach()
                local_final[position] = local_preprox
                local_gradient_average[position] = gradient_sum / local_updates

            global_preprox = global_postprox + server_rate * (
                local_final.mean(dim=0) - global_postprox
            )
            stored_direction[selected] = local_gradient_average
            average_direction = stored_direction.mean(dim=0)
            global_postprox = self.prox(
                global_preprox, proximal_parameter,
                float(self.constant), False,
            ).detach()
            synchronize(global_postprox)
            algorithm_time += time.perf_counter() - round_start
            process_time[round_index] = algorithm_time
            oracle_calling[round_index] = self.s * local_updates
            iterates[round_index] = global_postprox
            if RecordStatistics:
                function_value[round_index], subgradient[round_index] = torch_statistics(
                    global_postprox, statistics_function, self.prox,
                    float(self.constant), stationarity_parameter,
                )

        self.final_state = {
            "global_preprox": global_preprox.clone(),
            "global_postprox": global_postprox.clone(),
            "stored_direction": stored_direction.clone(),
            "average_direction": average_direction.clone(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling, process_time, iterates, function_value,
            subgradient, index,
        )
