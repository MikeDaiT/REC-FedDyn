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
from Algorithms.FMRI import (
    FMRIAccuracyFunction,
    FMRIAlgorithmResults,
    FMRIModel,
    FMRIObjectiveFunction,
    FMRIValueFunction,
    model_mean,
    proximal_model,
    record_statistics,
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


class FedDR(Optimizer):
    """Randomized Douglas-Rachford splitting with partial participation."""

    def __init__(self, InitialPoint: FloatArray, IterationBudget: int, ProxFunction: ProxFunction,
                 ClientsNumber: int, SamplingNumber: int, SmoothL: Float, ConvexMu: Float,
                 Constant: Float,
                 RandomSeed: int | None = None) -> None:
        super().__init__(InitialPoint, IterationBudget, ProxFunction, ClientsNumber,
                         SamplingNumber, SmoothL, ConvexMu, Constant, RandomSeed)

    def local_approx(self, proximal_center: FloatArray, initial_point: FloatArray,
                     objective_function: ObjectiveFunction, proximal_parameter: Float,
                     local_updates: int) -> FloatArray:
        """Approximately evaluate the local proximal point."""

        x = np.asarray(initial_point, dtype=self.dtype).copy()
        y = x.copy()
        y_prev = x.copy()
        gamma = proximal_parameter

        if self.L > 0 and self.mu > 0:
            L_Q = self.L + 1.0 / gamma
            mu_Q = self.mu + 1.0 / gamma
        elif self.L > 0 and self.mu == 0:
            L_Q = self.L + 1.0 / gamma
            mu_Q = 1.0 / gamma
        elif self.L > 0 and self.mu == -self.L:
            if gamma >= 1.0 / self.L:
                raise ValueError('Nonconvex case: inappropriate proximal parameter.')
            L_Q = self.L + 1.0 / gamma
            mu_Q = 1.0 / gamma - self.L
        else:
            if not (self.L == 0 and self.mu == 0):
                raise ValueError(f"Unsupported values: L={self.L}, mu={self.mu}.")

        if self.L > 0:
            for _ in range(local_updates):
                beta = (np.sqrt(L_Q) - np.sqrt(mu_Q)) / (np.sqrt(L_Q) + np.sqrt(mu_Q))
                grad = np.asarray(objective_function(x)[1], dtype=self.dtype) \
                       + (x - proximal_center) / gamma
                y = x - grad / L_Q
                x = y + beta * (y - y_prev)
                y_prev = y
            return x
        elif self.L == 0 and self.mu == 0:
            pass
            # ADAM: need to be complete later, should adaptive to pytorch

    def local_approx_torch(
        self, proximal_center: torch.Tensor, initial_point: torch.Tensor,
        objective: TorchObjective, proximal_parameter: float,
        local_updates: int, local_learning_rate: float = 1e-3,
    ) -> torch.Tensor:
        x = initial_point.detach().clone()
        if self.L == 0 and self.mu == 0:
            for _ in range(local_updates):
                _, data_gradient = objective(x)
                gradient = data_gradient.detach() + (
                    x - proximal_center
                ) / proximal_parameter
                x -= local_learning_rate * gradient
            return x

        local_smoothness = float(self.L + 1.0 / proximal_parameter)
        if self.mu > 0:
            local_strong_convexity = float(
                self.mu + 1.0 / proximal_parameter
            )
        elif self.mu == 0:
            local_strong_convexity = 1.0 / proximal_parameter
        elif self.mu == -self.L and proximal_parameter < 1.0 / self.L:
            local_strong_convexity = 1.0 / proximal_parameter - float(self.L)
        else:
            raise ValueError("Invalid FedDR local subproblem parameters.")
        momentum = (
            (local_smoothness**0.5 - local_strong_convexity**0.5)
            / (local_smoothness**0.5 + local_strong_convexity**0.5)
        )
        previous_gradient_point = x.clone()
        for _ in range(local_updates):
            _, data_gradient = objective(x)
            gradient = data_gradient.detach() + (
                x - proximal_center
            ) / proximal_parameter
            gradient_point = x - gradient / local_smoothness
            x = gradient_point + momentum * (
                gradient_point - previous_gradient_point
            )
            previous_gradient_point = gradient_point
        return x

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
        proximal_point = np.asarray(
            self.prox(point - stationarity_parameter * average_gradient,
                      stationarity_parameter, self.constant, False), dtype=self.dtype
        )
        stationarity_norm = parameter_norm(point - proximal_point) / stationarity_parameter
        return Float(objective_value), Float(stationarity_norm)

    def go(self, ProximalParameter: Float, RelaxationParameter: Float, LocalUpdates: int,
           Function: list[ObjectiveFunction],
           StatisticsFunction: list[ObjectiveFunction] | None = None,
           StationarityParameter: Float | None = None,
           InitialState: dict[str, object] | None = None,
           RecordStatistics: bool = True) -> AlgorithmResults:
        gamma = Float(ProximalParameter)
        alpha = Float(RelaxationParameter)
        local_updates = int(LocalUpdates)
        if gamma <= 0 or not 0 < alpha <= 2 or local_updates <= 0:
            raise ValueError("Invalid FedDR parameters.")
        if not 1 <= self.s <= self.N:
            raise ValueError("SamplingNumber must lie between 1 and ClientsNumber.")

        statistics_function = Function if StatisticsFunction is None else StatisticsFunction
        stationarity_parameter = gamma if StationarityParameter is None else Float(StationarityParameter)
        rng = random.Random(self.seed)

        if InitialState is None:
            local_y = np.repeat(self.x_0[None, ...], self.N, axis=0)
            local_x = np.empty((self.N,) + self.parameter_shape, dtype=self.dtype)
            local_hat = np.empty((self.N,) + self.parameter_shape, dtype=self.dtype)
            initialization_oracle_calling = 0
            for client in range(self.N):
                local_x[client] = self.local_approx(
                    local_y[client], self.x_0, Function[client], gamma, local_updates
                )
                local_hat[client] = 2.0 * local_x[client] - local_y[client]
                initialization_oracle_calling += local_updates
            server_point = self.x_0.copy()
            # Preserve tilde{x}^0 = n^{-1} sum_i hat{x}_i^0, the invariant
            # required by the incremental server aggregation.
            server_average = np.mean(local_hat, axis=0)
            algorithm_time = 0.0
        else:
            local_y = np.asarray(InitialState["local_y"], dtype=self.dtype).copy()
            local_x = np.asarray(InitialState["local_x"], dtype=self.dtype).copy()
            local_hat = np.asarray(InitialState["local_hat"], dtype=self.dtype).copy()
            server_point = np.asarray(
                InitialState["server_point"], dtype=self.dtype
            ).copy()
            server_average = np.asarray(
                InitialState["server_average"], dtype=self.dtype
            ).copy()
            algorithm_time = float(InitialState["algorithm_time"])
            initialization_oracle_calling = 0
            rng.setstate(InitialState["rng_state"])

        process_time = np.zeros(self.R, dtype=self.dtype)
        iterates = np.empty((self.R,) + self.parameter_shape, dtype=self.dtype)
        function_value = np.zeros(self.R, dtype=self.dtype)
        subgradient = np.zeros(self.R, dtype=self.dtype)
        index = np.zeros(self.R, dtype=np.bool_)
        oracle_calling = np.zeros(self.R, dtype=np.int64)
        for ite in range(self.R):
            round_start = time.perf_counter()
            selected_clients = rng.sample(range(self.N), self.s)
            round_oracle_calling = 0

            for client in selected_clients:
                previous_hat = local_hat[client].copy()
                local_y[client] += alpha * (server_point - local_x[client])
                local_x[client] = self.local_approx(
                    local_y[client], local_x[client], Function[client], gamma,
                    local_updates
                )
                local_hat[client] = 2.0 * local_x[client] - local_y[client]
                server_average += (local_hat[client] - previous_hat) / self.N
                round_oracle_calling += local_updates

            server_point = np.asarray(
                self.prox(server_average, gamma, self.constant, False),
                dtype=self.dtype,
            )
            algorithm_time += time.perf_counter() - round_start
            process_time[ite] = algorithm_time
            iterates[ite] = server_point
            oracle_calling[ite] = round_oracle_calling
            if ite == 0:
                # FedDR performs one all-client local-prox initialization before
                # round 0; include that gradient work in the first round entry.
                oracle_calling[ite] += initialization_oracle_calling
            if RecordStatistics:
                function_value[ite], subgradient[ite] = self._statistics(
                    server_point, statistics_function, stationarity_parameter
                )

        self.final_state = {
            "local_y": local_y.copy(),
            "local_x": local_x.copy(),
            "local_hat": local_hat.copy(),
            "server_point": server_point.copy(),
            "server_average": server_average.copy(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }

        return AlgorithmResults(oracle_calling, process_time, iterates, function_value, subgradient, index)

    def go_torch(
        self, ProximalParameter: float, RelaxationParameter: float,
        LocalUpdates: int, Function: list[TorchObjective],
        StatisticsFunction: list[TorchObjective] | None = None,
        StationarityParameter: float | None = None,
        LocalLearningRate: float = 1e-3,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        if not isinstance(self.x_0, torch.Tensor):
            raise TypeError("go_torch requires a torch InitialPoint.")
        gamma = float(ProximalParameter)
        relaxation = float(RelaxationParameter)
        local_updates = int(LocalUpdates)
        if gamma <= 0 or not 0 < relaxation <= 2 or local_updates <= 0:
            raise ValueError("Invalid FedDR parameters.")
        statistics_function = Function if StatisticsFunction is None else StatisticsFunction
        stationarity_parameter = gamma if StationarityParameter is None else float(StationarityParameter)
        rng = random.Random(self.seed)

        if InitialState is None:
            initial = self.x_0.detach().clone()
            local_y = initial.repeat((self.N,) + (1,) * initial.ndim)
            local_x = torch.empty_like(local_y)
            local_hat = torch.empty_like(local_y)
            for client in range(self.N):
                local_x[client] = self.local_approx_torch(
                    local_y[client], initial, Function[client], gamma,
                    local_updates, LocalLearningRate,
                )
                local_hat[client] = 2.0 * local_x[client] - local_y[client]
            initialization_calls = self.N * local_updates
            server_point = initial.clone()
            server_average = torch.mean(local_hat, dim=0)
            algorithm_time = 0.0
        else:
            local_y = InitialState["local_y"].detach().clone()
            local_x = InitialState["local_x"].detach().clone()
            local_hat = InitialState["local_hat"].detach().clone()
            server_point = InitialState["server_point"].detach().clone()
            server_average = InitialState["server_average"].detach().clone()
            algorithm_time = float(InitialState["algorithm_time"])
            initialization_calls = 0
            rng.setstate(InitialState["rng_state"])

        oracle_calling = torch.zeros(self.R, dtype=torch.int64, device=server_point.device)
        process_time = server_point.new_zeros(self.R)
        iterates = server_point.new_empty((self.R,) + tuple(server_point.shape))
        function_value = server_point.new_zeros(self.R)
        subgradient = server_point.new_zeros(self.R)
        index = torch.zeros(self.R, dtype=torch.bool, device=server_point.device)

        for round_index in range(self.R):
            synchronize(server_point)
            round_start = time.perf_counter()
            for client in rng.sample(range(self.N), self.s):
                previous_hat = local_hat[client].clone()
                local_y[client] += relaxation * (server_point - local_x[client])
                local_x[client] = self.local_approx_torch(
                    local_y[client], local_x[client], Function[client], gamma,
                    local_updates, LocalLearningRate,
                )
                local_hat[client] = 2.0 * local_x[client] - local_y[client]
                server_average += (local_hat[client] - previous_hat) / self.N
            server_point = self.prox(
                server_average, gamma, float(self.constant), False
            ).detach()
            synchronize(server_point)
            algorithm_time += time.perf_counter() - round_start
            process_time[round_index] = algorithm_time
            oracle_calling[round_index] = self.s * local_updates
            if round_index == 0:
                oracle_calling[round_index] += initialization_calls
            iterates[round_index] = server_point
            if RecordStatistics:
                function_value[round_index], subgradient[round_index] = torch_statistics(
                    server_point, statistics_function, self.prox,
                    float(self.constant), stationarity_parameter,
                )

        self.final_state = {
            "local_y": local_y.clone(),
            "local_x": local_x.clone(),
            "local_hat": local_hat.clone(),
            "server_point": server_point.clone(),
            "server_average": server_average.clone(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling, process_time, iterates, function_value,
            subgradient, index,
        )

    def _local_approx_fmri(
        self,
        proximal_center: FMRIModel,
        initial_model: FMRIModel,
        objective_function: FMRIObjectiveFunction,
        proximal_parameter: Float,
        local_updates: int,
    ) -> FMRIModel:
        """Apply the original FedDR local accelerated solver to MRI state."""

        x = initial_model.copy()
        gradient_point = x.copy()
        previous_gradient_point = x.copy()
        gamma = Float(proximal_parameter)
        if self.L > 0 and self.mu > 0:
            local_smoothness = self.L + 1.0 / gamma
            local_strong_convexity = self.mu + 1.0 / gamma
        elif self.L > 0 and self.mu == 0:
            local_smoothness = self.L + 1.0 / gamma
            local_strong_convexity = 1.0 / gamma
        elif self.L > 0 and self.mu == -self.L:
            if gamma >= 1.0 / self.L:
                raise ValueError(
                    "Nonconvex case: inappropriate proximal parameter."
                )
            local_smoothness = self.L + 1.0 / gamma
            local_strong_convexity = 1.0 / gamma - self.L
        else:
            raise ValueError(f"Unsupported values: L={self.L}, mu={self.mu}.")

        beta = (
            np.sqrt(local_smoothness) - np.sqrt(local_strong_convexity)
        ) / (
            np.sqrt(local_smoothness) + np.sqrt(local_strong_convexity)
        )
        for _ in range(local_updates):
            _, smooth_gradient = objective_function(x)
            gradient = smooth_gradient + (x - proximal_center) / gamma
            gradient_point = x - gradient / local_smoothness
            x = gradient_point + beta * (
                gradient_point - previous_gradient_point
            )
            previous_gradient_point = gradient_point
        return x

    def go_fmri(
        self,
        ProximalParameter: Float,
        RelaxationParameter: Float,
        LocalUpdates: int,
        Function: list[FMRIObjectiveFunction],
        StatisticsFunction: FMRIValueFunction,
        AccuracyFunction: FMRIAccuracyFunction,
        InitialIntercept: Float = Float(0.0),
        InitialState: dict[str, object] | None = None,
    ) -> FMRIAlgorithmResults:
        """Run FedDR on a 3-D coefficient volume plus scalar intercept."""

        gamma = Float(ProximalParameter)
        relaxation = Float(RelaxationParameter)
        local_updates = int(LocalUpdates)
        if gamma <= 0 or not 0 < relaxation <= 2 or local_updates <= 0:
            raise ValueError("Invalid fMRI FedDR parameters.")
        rng = random.Random(self.seed)
        if InitialState is None:
            initial_model = FMRIModel(self.x_0.copy(), Float(InitialIntercept))
            local_y = [initial_model.copy() for _ in range(self.N)]
            local_x: list[FMRIModel] = []
            local_hat: list[FMRIModel] = []
            initialization_oracles = 0
            for client in range(self.N):
                local_model = self._local_approx_fmri(
                    local_y[client],
                    initial_model,
                    Function[client],
                    gamma,
                    local_updates,
                )
                local_x.append(local_model)
                local_hat.append(2.0 * local_model - local_y[client])
                initialization_oracles += local_updates
            server_point = initial_model.copy()
            # Preserve tilde{x}^0 = n^{-1} sum_i hat{x}_i^0, the invariant
            # required by the incremental server aggregation.
            server_average = model_mean(local_hat, self.parameter_shape)
            algorithm_time = 0.0
            completed_rounds = 0
        else:
            local_y = [point.copy() for point in InitialState["local_y"]]
            local_x = [point.copy() for point in InitialState["local_x"]]
            local_hat = [point.copy() for point in InitialState["local_hat"]]
            server_point = InitialState["server_point"].copy()
            server_average = InitialState["server_average"].copy()
            algorithm_time = float(InitialState["algorithm_time"])
            completed_rounds = int(InitialState["completed_rounds"])
            initialization_oracles = 0
            rng.setstate(InitialState["rng_state"])

        oracle_calling = np.zeros(self.R, dtype=np.int64)
        process_time = np.zeros(self.R, dtype=np.float64)
        function_value = np.zeros(self.R, dtype=np.float64)
        test_accuracy = np.zeros(self.R, dtype=np.float64)
        index = np.zeros(self.R, dtype=np.bool_)
        refinement_count = np.zeros(self.R, dtype=np.int32)

        for iteration in range(self.R):
            round_start = time.perf_counter()
            selected_clients = rng.sample(range(self.N), self.s)
            for client in selected_clients:
                previous_hat = local_hat[client].copy()
                local_y[client] = local_y[client] + relaxation * (
                    server_point - local_x[client]
                )
                local_x[client] = self._local_approx_fmri(
                    local_y[client],
                    local_x[client],
                    Function[client],
                    gamma,
                    local_updates,
                )
                local_hat[client] = 2.0 * local_x[client] - local_y[client]
                server_average = server_average + (
                    local_hat[client] - previous_hat
                ) / self.N
                oracle_calling[iteration] += local_updates

            server_point = proximal_model(
                server_average,
                self.prox,
                gamma,
                self.constant,
            )
            algorithm_time += time.perf_counter() - round_start
            process_time[iteration] = algorithm_time
            function_value[iteration], test_accuracy[iteration] = record_statistics(
                server_point,
                StatisticsFunction,
                AccuracyFunction,
                self.prox,
                self.constant,
            )

        # Initialization solves every client's first local proximal problem.
        # Keep that work separate: oracle_calling contains only the clients
        # participating in each recorded communication round.
        self.initialization_oracle_calling = initialization_oracles
        self.final_state = {
            "local_y": [point.copy() for point in local_y],
            "local_x": [point.copy() for point in local_x],
            "local_hat": [point.copy() for point in local_hat],
            "server_point": server_point.copy(),
            "server_average": server_average.copy(),
            "algorithm_time": algorithm_time,
            "completed_rounds": completed_rounds + self.R,
            "rng_state": rng.getstate(),
        }
        return FMRIAlgorithmResults(
            oracle_calling=oracle_calling,
            process_time=process_time,
            function_value=function_value,
            test_accuracy=test_accuracy,
            final_point=server_point.copy(),
            index=index,
            refinement_count=refinement_count,
            communication_rounds=np.arange(
                completed_rounds + 1,
                completed_rounds + self.R + 1,
                dtype=np.int64,
            ),
        )
