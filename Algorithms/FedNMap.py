import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from Algorithms.Base_Optimizer import (
    FloatArray, Float, Optimizer, ProxFunction, parameter_norm,
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
    preprox_point: FloatArray
    function_value: FloatArray
    subgradient: FloatArray


class FedNMap(Optimizer):
    """Partial FedNMap with cached directions and sampled-iterate aggregation."""

    def __init__(self, InitialPoint: FloatArray, IterationBudget: int, ProxFunction: ProxFunction,
                 ClientsNumber: int, SamplingNumber: int, SmoothL: Float, ConvexMu: Float,
                 Constant: Float,
                 RandomSeed: int | None = None) -> None:
        super().__init__(InitialPoint, IterationBudget, ProxFunction, ClientsNumber,
                         SamplingNumber, SmoothL, ConvexMu, Constant, RandomSeed)

    def go(self, 
           ProximalParameter: Float, 
           LocalLearningRate: Float, 
           ServerLearningRate: Float,
           LocalUpdates: int, 
           Function: list[ObjectiveFunction],
           InitialState: dict[str, object] | None = None,
           RecordStatistics: bool = True) -> AlgorithmResults:
        gamma = Float(ProximalParameter)
        eta_a = Float(LocalLearningRate)
        eta_s = Float(ServerLearningRate)
        local_updates = int(LocalUpdates)
        
        statistics_function = Function
        rng = random.Random(self.seed)
        if InitialState is None:
            z = self.x_0.copy()
            global_point = np.asarray(
                self.prox(z, gamma, self.constant, False), dtype=self.dtype
            )
            correction = np.zeros(
                (self.N,) + self.parameter_shape, dtype=self.dtype
            )
            stored_direction = np.zeros_like(correction)
            average_direction = np.zeros(self.parameter_shape, dtype=self.dtype)
            algorithm_time = 0.0
        else:
            z = np.asarray(InitialState["z"], dtype=self.dtype).copy()
            global_point = np.asarray(
                InitialState["global_point"], dtype=self.dtype
            ).copy()
            correction = np.asarray(
                InitialState["correction"], dtype=self.dtype
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
        # Shapes: (communication rounds, *parameter_shape).
        iterates = np.empty((self.R,) + self.parameter_shape, dtype=self.dtype)
        preprox_iterates = np.empty((self.R,) + self.parameter_shape, dtype=self.dtype)
        function_value = np.zeros(self.R, dtype=self.dtype)
        subgradient = np.zeros(self.R, dtype=self.dtype)
        oracle_calling = np.zeros(self.R, dtype=np.int64)
        for ite in range(self.R):
            round_start = time.perf_counter()
            round_oracle_calling = 0
            selected_clients = rng.sample(range(self.N), self.s)
            correction[selected_clients] += (
                average_direction - stored_direction[selected_clients]
            )
            # Server-side normalization keeps the cached control variates in
            # the zero-mean subspace; nonsampled clients do no local work.
            correction -= np.mean(correction, axis=0, keepdims=True)
            normal_shift = (z - global_point) / gamma
            current_direction = np.empty(
                (self.s,) + self.parameter_shape, dtype=self.dtype
            )
            local_final_iterate = np.empty(
                (self.s,) + self.parameter_shape, dtype=self.dtype
            )

            for position, client in enumerate(selected_clients):
                local_z = z.copy()
                for local_step in range(local_updates):
                    local_x = global_point if local_step == 0 else np.asarray(
                        self.prox(local_z, gamma, self.constant, False),
                        dtype=self.dtype,
                    )
                    _, local_gradient = Function[client](local_x)
                    local_z -= eta_a * (np.asarray(local_gradient, dtype=self.dtype) + normal_shift + correction[client])
                    round_oracle_calling += 1
                local_final_iterate[position] = local_z
                current_direction[position] = (z - local_z) / (eta_a * local_updates)

            # Replace the sampled clients' directions, retain the nonsampled
            # clients' cached directions, and average over all clients.
            stored_direction[selected_clients] = current_direction
            average_direction = np.mean(stored_direction, axis=0)

            # Only sampled clients produce local iterates in this round, so
            # the server model update averages their newly returned iterates.
            selected_iterate_average = np.mean(local_final_iterate, axis=0)
            z += eta_s * (selected_iterate_average - z)
            global_point = np.asarray(
                self.prox(z, gamma, self.constant, False),
                dtype=self.dtype,
            )

            algorithm_time += time.perf_counter() - round_start
            process_time[ite] = algorithm_time
            oracle_calling[ite] = round_oracle_calling
            iterates[ite] = global_point
            preprox_iterates[ite] = z

            if RecordStatistics:
                average_value = Float(0.0)
                average_gradient = np.zeros(
                    self.parameter_shape, dtype=self.dtype
                )
                for function in statistics_function:
                    local_value, local_gradient = function(global_point)
                    average_value += local_value / self.N
                    average_gradient += np.asarray(
                        local_gradient, dtype=self.dtype
                    ) / self.N
                function_value[ite] = average_value + self.prox(
                    global_point, Float(1.0), self.constant, True
                )
                proximal_point = np.asarray(
                    self.prox(
                        global_point - gamma * average_gradient,
                        gamma,
                        self.constant,
                        False,
                    ),
                    dtype=self.dtype,
                )
                natural_map = (global_point - proximal_point) / gamma
                subgradient[ite] = parameter_norm(natural_map) ** 2

        self.final_state = {
            "z": z.copy(),
            "global_point": global_point.copy(),
            "correction": correction.copy(),
            "stored_direction": stored_direction.copy(),
            "average_direction": average_direction.copy(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }

        return AlgorithmResults(oracle_calling, process_time, iterates, preprox_iterates,
                                function_value, subgradient)

    def go_torch(
        self, ProximalParameter: float, LocalLearningRate: float,
        ServerLearningRate: float, LocalUpdates: int,
        Function: list[TorchObjective],
        StationarityParameter: float | None = None,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        if not isinstance(self.x_0, torch.Tensor):
            raise TypeError("go_torch requires a torch InitialPoint.")
        gamma = float(ProximalParameter)
        local_rate = float(LocalLearningRate)
        server_rate = float(ServerLearningRate)
        local_updates = int(LocalUpdates)
        stationarity_parameter = (
            gamma if StationarityParameter is None
            else float(StationarityParameter)
        )
        rng = random.Random(self.seed)

        if InitialState is None:
            z = self.x_0.detach().clone()
            global_point = self.prox(
                z, gamma, float(self.constant), False
            ).detach()
            correction = z.new_zeros((self.N,) + tuple(z.shape))
            stored_direction = torch.zeros_like(correction)
            average_direction = torch.zeros_like(z)
            algorithm_time = 0.0
        else:
            z = InitialState["z"].detach().clone()
            global_point = InitialState["global_point"].detach().clone()
            correction = InitialState["correction"].detach().clone()
            stored_direction = InitialState["stored_direction"].detach().clone()
            average_direction = InitialState["average_direction"].detach().clone()
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        device = z.device
        oracle_calling = torch.zeros(self.R, dtype=torch.int64, device=device)
        process_time = z.new_zeros(self.R)
        iterates = z.new_empty((self.R,) + tuple(z.shape))
        preprox_iterates = torch.empty_like(iterates)
        function_value = z.new_zeros(self.R)
        subgradient = z.new_zeros(self.R)

        for round_index in range(self.R):
            synchronize(z)
            round_start = time.perf_counter()
            selected = rng.sample(range(self.N), self.s)
            correction[selected] += average_direction - stored_direction[selected]
            correction -= correction.mean(dim=0, keepdim=True)
            normal_shift = (z - global_point) / gamma
            local_final = z.new_empty((self.s,) + tuple(z.shape))
            current_direction = torch.empty_like(local_final)

            for position, client in enumerate(selected):
                local_z = z.clone()
                for local_step in range(local_updates):
                    local_x = global_point if local_step == 0 else self.prox(
                        local_z, gamma, float(self.constant), False
                    ).detach()
                    _, local_gradient = Function[client](local_x)
                    local_z -= local_rate * (
                        local_gradient.detach() + normal_shift + correction[client]
                    )
                local_final[position] = local_z
                current_direction[position] = (
                    z - local_z
                ) / (local_rate * local_updates)

            stored_direction[selected] = current_direction
            average_direction = stored_direction.mean(dim=0)
            z += server_rate * (local_final.mean(dim=0) - z)
            global_point = self.prox(
                z, gamma, float(self.constant), False
            ).detach()
            synchronize(global_point)
            algorithm_time += time.perf_counter() - round_start
            process_time[round_index] = algorithm_time
            oracle_calling[round_index] = self.s * local_updates
            iterates[round_index] = global_point
            preprox_iterates[round_index] = z
            if RecordStatistics:
                value, stationarity = torch_statistics(
                    global_point, Function, self.prox,
                    float(self.constant), stationarity_parameter,
                )
                function_value[round_index] = value
                subgradient[round_index] = stationarity

        self.final_state = {
            "z": z.clone(),
            "global_point": global_point.clone(),
            "correction": correction.clone(),
            "stored_direction": stored_direction.clone(),
            "average_direction": average_direction.clone(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling, process_time, iterates, preprox_iterates,
            function_value, subgradient,
        )
