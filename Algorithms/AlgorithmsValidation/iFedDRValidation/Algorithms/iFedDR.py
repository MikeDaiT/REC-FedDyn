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
    model_dot,
    model_mean,
    proximal_model,
    record_statistics,
    squared_norm,
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
    refinement_count: np.ndarray
    communication_rounds: np.ndarray
    local_iterations: np.ndarray


class iFedDR(Optimizer):
    """Cached-direction partial adaptation of error-corrected inexact FedDR."""

    def __init__(self, InitialPoint: FloatArray, IterationBudget: int, ProxFunction: ProxFunction,
                 ClientsNumber: int, SamplingNumber: int, SmoothL: Float, ConvexMu: Float,
                 Constant: Float,
                 RandomSeed: int | None = None) -> None:
        super().__init__(InitialPoint, IterationBudget, ProxFunction, ClientsNumber,
                         SamplingNumber, SmoothL, ConvexMu, Constant, RandomSeed)

    def local_approx(self, initial_point: FloatArray, proximal_center: FloatArray,
                     objective_function: ObjectiveFunction, gamma: Float,
                     local_updates: int) -> FloatArray:
        """Refine a local proximal point."""

        x = np.asarray(initial_point, dtype=self.dtype).copy()
        y = x.copy()
        y_prev = x.copy()

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
        self, initial_point: torch.Tensor, proximal_center: torch.Tensor,
        objective: TorchObjective, gamma: float, local_updates: int,
        local_learning_rate: float = 1e-3,
    ) -> torch.Tensor:
        x = initial_point.detach().clone()
        if self.L == 0 and self.mu == 0:
            for _ in range(local_updates):
                _, data_gradient = objective(x)
                gradient = data_gradient.detach() + (x - proximal_center) / gamma
                x -= local_learning_rate * gradient
            return x

        local_smoothness = float(self.L + 1.0 / gamma)
        if self.mu > 0:
            local_strong_convexity = float(self.mu + 1.0 / gamma)
        elif self.mu == 0:
            local_strong_convexity = 1.0 / gamma
        elif self.mu == -self.L and gamma < 1.0 / self.L:
            local_strong_convexity = 1.0 / gamma - float(self.L)
        else:
            raise ValueError("Invalid iFedDR local subproblem parameters.")
        momentum = (
            (local_smoothness**0.5 - local_strong_convexity**0.5)
            / (local_smoothness**0.5 + local_strong_convexity**0.5)
        )
        previous_gradient_point = x.clone()
        for _ in range(local_updates):
            _, data_gradient = objective(x)
            gradient = data_gradient.detach() + (x - proximal_center) / gamma
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

    def go(self, ProximalParameter: Float, RelaxationParameter: Float,
           SigmaSquared: Float,
           InitialLocalUpdates: int, Function: list[ObjectiveFunction],
           RefinementFactor: Float = Float(2.0),
           MaxRefinements: int = 100, IncreaseInitialUpdates: bool = False,
           StatisticsFunction: list[ObjectiveFunction] | None = None,
           StationarityParameter: Float | None = None,
           InitialState: dict[str, object] | None = None,
           RecordStatistics: bool = True) -> AlgorithmResults:
        gamma = Float(ProximalParameter)
        relaxation = Float(RelaxationParameter)
        sigma_squared = Float(SigmaSquared)
        initial_local_updates = int(InitialLocalUpdates)
        max_refinements = int(MaxRefinements)
        # Retained until the experiment configurations are migrated.  The
        # released implementation uses a fixed amount of work per refinement.
        del RefinementFactor
        if (
            gamma <= 0
            or not 0 < relaxation < 2
            or not 0 < sigma_squared < 1
            or initial_local_updates <= 0
            or max_refinements <= 0
        ):
            raise ValueError("Invalid iFedDR parameters.")
        if not 1 <= self.s <= self.N:
            raise ValueError("SamplingNumber must lie between 1 and ClientsNumber.")

        statistics_function = Function if StatisticsFunction is None else StatisticsFunction
        stationarity_parameter = gamma if StationarityParameter is None else Float(StationarityParameter)
        rng = random.Random(self.seed)

        if InitialState is None:
            local_s = np.repeat(self.x_0[None, ...], self.N, axis=0)
            previous_local_point = np.repeat(self.x_0[None, ...], self.N, axis=0)
            reflected_direction = np.repeat(self.x_0[None, ...], self.N, axis=0)
            previous_server_point = self.x_0.copy()
            previous_alpha = Float(0.0)
            cumulative_communications = 0
            cumulative_local_iterations = 0
            total_refinements = 0
            algorithm_time = 0.0
        else:
            local_s = np.asarray(InitialState["local_s"], dtype=self.dtype).copy()
            previous_local_point = np.asarray(
                InitialState["previous_local_point"], dtype=self.dtype
            ).copy()
            reflected_direction = np.asarray(
                InitialState["reflected_direction"], dtype=self.dtype
            ).copy()
            previous_server_point = np.asarray(
                InitialState["previous_server_point"], dtype=self.dtype
            ).copy()
            previous_alpha = Float(InitialState["previous_alpha"])
            cumulative_communications = int(InitialState["cumulative_communications"])
            cumulative_local_iterations = int(
                InitialState["cumulative_local_iterations"]
            )
            total_refinements = int(InitialState["total_refinements"])
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        process_time = np.zeros(self.R, dtype=self.dtype)
        iterates = np.empty((self.R,) + self.parameter_shape, dtype=self.dtype)
        function_value = np.zeros(self.R, dtype=self.dtype)
        subgradient = np.zeros(self.R, dtype=self.dtype)
        index = np.zeros(self.R, dtype=np.bool_)
        refinement_count = np.zeros(self.R, dtype=np.int32)
        communication_rounds = np.zeros(self.R, dtype=np.int32)
        local_iterations = np.zeros(self.R, dtype=np.int32)
        oracle_calling = np.zeros(self.R, dtype=np.int64)
        for ite in range(self.R):
            round_start = time.perf_counter()
            round_oracle_calling = 0
            selected_clients = rng.sample(range(self.N), self.s)
            local_s[selected_clients] -= relaxation * previous_alpha * (
                previous_local_point[selected_clients] - previous_server_point
            )
            local_point = local_s[selected_clients].copy()
            multiplier = max(1, total_refinements + 1) if IncreaseInitialUpdates else 1
            round_local_updates = initial_local_updates * multiplier
            accepted = False
            refinement_calls = 0
            current_round_iterations = 0
            base_quantity = Float(0.0)
            eps_absolute = Float(1e-13)
            eps_comparison = Float(1e-8)

            while refinement_calls < max_refinements:
                for position, client in enumerate(selected_clients):
                    local_point[position] = self.local_approx(
                        local_point[position], local_s[client], Function[client], gamma,
                        round_local_updates
                    )
                    round_oracle_calling += round_local_updates
                    current_round_iterations += round_local_updates

                local_gradient = np.empty(
                    (self.s,) + self.parameter_shape, dtype=self.dtype
                )
                for position, client in enumerate(selected_clients):
                    _, gradient = Function[client](local_point[position])
                    local_gradient[position] = np.asarray(gradient, dtype=self.dtype)
                    round_oracle_calling += 1

                reflected_direction[selected_clients] = local_point - gamma * local_gradient
                reflected_average = np.mean(reflected_direction, axis=0)
                server_point = np.asarray(
                    self.prox(reflected_average, gamma, self.constant, False),
                    dtype=self.dtype,
                )
                local_difference = local_point - server_point
                second_difference = (
                    gamma * local_gradient - local_s[selected_clients] + server_point
                )
                proximal_residual = (
                    local_s[selected_clients] - gamma * local_gradient - local_point
                )

                xi = Float(np.sum(local_difference ** 2))
                zeta = Float(np.sum(second_difference ** 2) / gamma ** 2)
                mu_scalar = Float(np.sum(
                    local_difference
                    * (local_s[selected_clients] - gamma * local_gradient - server_point)
                ))
                error_value = Float(np.sum(proximal_residual ** 2))
                first_quantity = Float(np.sqrt(error_value))
                second_quantity = Float(
                    np.sqrt(sigma_squared * max(xi, zeta))
                )
                if base_quantity == 0:
                    base_quantity = Float(min(first_quantity, second_quantity))
                relative_condition = (
                    first_quantity <= second_quantity
                    or max(first_quantity, second_quantity)
                    <= eps_absolute + eps_comparison * base_quantity
                )
                refinement_calls += 1

                if xi == 0.0:
                    adaptive_alpha = Float(0.0)
                    accepted = relative_condition
                else:
                    adaptive_alpha = Float(mu_scalar / xi)
                    accepted = relative_condition
                if accepted:
                    break

                total_refinements += 1

            if not accepted:
                raise RuntimeError(
                    "iFedDR relative-error condition was not satisfied after "
                    f"{max_refinements} refinement calls."
                )

            previous_local_point[selected_clients] = local_point
            previous_server_point = server_point
            previous_alpha = adaptive_alpha
            cumulative_communications += 1
            cumulative_local_iterations += current_round_iterations
            algorithm_time += time.perf_counter() - round_start

            process_time[ite] = algorithm_time
            iterates[ite] = server_point
            refinement_count[ite] = refinement_calls
            communication_rounds[ite] = cumulative_communications
            local_iterations[ite] = cumulative_local_iterations
            oracle_calling[ite] = round_oracle_calling
            if RecordStatistics:
                function_value[ite], subgradient[ite] = self._statistics(
                    server_point, statistics_function, stationarity_parameter
                )

        self.final_state = {
            "local_s": local_s.copy(),
            "previous_local_point": previous_local_point.copy(),
            "reflected_direction": reflected_direction.copy(),
            "previous_server_point": previous_server_point.copy(),
            "previous_alpha": previous_alpha,
            "cumulative_communications": cumulative_communications,
            "cumulative_local_iterations": cumulative_local_iterations,
            "total_refinements": total_refinements,
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }

        return AlgorithmResults(
            oracle_calling, process_time, iterates, function_value, subgradient,
            index, refinement_count, communication_rounds, local_iterations
        )

    def go_torch(
        self, ProximalParameter: float, RelaxationParameter: float,
        SigmaSquared: float, InitialLocalUpdates: int,
        Function: list[TorchObjective], RefinementFactor: float = 2.0,
        MaxRefinements: int = 100, IncreaseInitialUpdates: bool = False,
        MaxLocalIterations: int | None = None,
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
        sigma_squared = float(SigmaSquared)
        initial_updates = int(InitialLocalUpdates)
        del RefinementFactor
        max_refinements = int(MaxRefinements)
        max_local_iterations = (
            None if MaxLocalIterations is None else int(MaxLocalIterations)
        )
        if (
            gamma <= 0
            or not 0 < relaxation < 2
            or not 0 < sigma_squared < 1
            or initial_updates <= 0
            or max_refinements <= 0
        ):
            raise ValueError("Invalid iFedDR parameters.")
        if max_local_iterations is not None and max_local_iterations <= 0:
            raise ValueError("MaxLocalIterations must be positive.")
        statistics_function = Function if StatisticsFunction is None else StatisticsFunction
        stationarity_parameter = gamma if StationarityParameter is None else float(StationarityParameter)
        rng = random.Random(self.seed)

        if InitialState is None:
            initial = self.x_0.detach().clone()
            local_s = initial.repeat((self.N,) + (1,) * initial.ndim)
            previous_local_point = local_s.clone()
            reflected_direction = local_s.clone()
            previous_server_point = initial.clone()
            previous_alpha = initial.new_zeros(())
            cumulative_communications = 0
            cumulative_local_iterations = 0
            total_refinements = 0
            algorithm_time = 0.0
        else:
            local_s = InitialState["local_s"].detach().clone()
            previous_local_point = InitialState["previous_local_point"].detach().clone()
            reflected_direction = InitialState["reflected_direction"].detach().clone()
            previous_server_point = InitialState["previous_server_point"].detach().clone()
            previous_alpha = InitialState["previous_alpha"].detach().clone()
            cumulative_communications = int(InitialState["cumulative_communications"])
            cumulative_local_iterations = int(InitialState["cumulative_local_iterations"])
            total_refinements = int(InitialState["total_refinements"])
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        device = previous_server_point.device
        oracle_calling = torch.zeros(self.R, dtype=torch.int64, device=device)
        process_time = previous_server_point.new_zeros(self.R)
        iterates = previous_server_point.new_empty(
            (self.R,) + tuple(previous_server_point.shape)
        )
        function_value = previous_server_point.new_zeros(self.R)
        subgradient = previous_server_point.new_zeros(self.R)
        index = torch.zeros(self.R, dtype=torch.bool, device=device)
        refinement_count = torch.zeros(self.R, dtype=torch.int32, device=device)
        communication_rounds = torch.zeros(self.R, dtype=torch.int64, device=device)
        local_iterations = torch.zeros(self.R, dtype=torch.int64, device=device)

        for round_index in range(self.R):
            synchronize(previous_server_point)
            round_start = time.perf_counter()
            selected = rng.sample(range(self.N), self.s)
            local_s[selected] -= relaxation * previous_alpha * (
                previous_local_point[selected] - previous_server_point
            )
            local_point = local_s[selected].clone()
            multiplier = max(1, total_refinements + 1) if IncreaseInitialUpdates else 1
            round_local_updates = initial_updates * multiplier
            accepted = False
            refinement_calls = 0
            current_round_iterations = 0
            client_iterations = 0
            round_calls = 0
            base_quantity = None

            while refinement_calls < max_refinements:
                steps = round_local_updates
                if (
                    max_local_iterations is not None
                    and client_iterations + steps > max_local_iterations
                ):
                    raise RuntimeError(
                        "iFedDR relative-error condition was not satisfied "
                        "before MaxLocalIterations was reached."
                    )
                for position, client in enumerate(selected):
                    local_point[position] = self.local_approx_torch(
                        local_point[position], local_s[client], Function[client],
                        gamma, steps, LocalLearningRate,
                    )
                    round_calls += steps
                    current_round_iterations += steps
                client_iterations += steps

                local_gradient = torch.empty_like(local_point)
                for position, client in enumerate(selected):
                    _, gradient = Function[client](local_point[position])
                    local_gradient[position] = gradient.detach()
                    round_calls += 1

                reflected_direction[selected] = local_point - gamma * local_gradient
                reflected_average = reflected_direction.mean(dim=0)
                server_point = self.prox(
                    reflected_average, gamma, float(self.constant), False
                ).detach()
                local_difference = local_point - server_point
                second_difference = gamma * local_gradient - local_s[selected] + server_point
                proximal_residual = local_s[selected] - gamma * local_gradient - local_point
                xi_tensor = torch.sum(local_difference.square())
                zeta_tensor = torch.sum(second_difference.square()) / gamma**2
                mu_tensor = torch.sum(
                    local_difference
                    * (local_s[selected] - gamma * local_gradient - server_point)
                )
                error_tensor = torch.sum(proximal_residual.square())
                first_quantity = error_tensor.sqrt()
                second_quantity = (
                    sigma_squared
                    * torch.maximum(xi_tensor, zeta_tensor)
                ).sqrt()
                if base_quantity is None:
                    base_quantity = torch.minimum(
                        first_quantity, second_quantity
                    )
                accepted_tensor = torch.logical_or(
                    first_quantity <= second_quantity,
                    torch.maximum(first_quantity, second_quantity)
                    <= 1e-13 + 1e-8 * base_quantity,
                )
                refinement_calls += 1
                safe_xi = torch.clamp_min(
                    xi_tensor, torch.finfo(xi_tensor.dtype).tiny
                )
                adaptive_alpha = torch.where(
                    xi_tensor == 0.0,
                    torch.zeros_like(xi_tensor),
                    mu_tensor / safe_xi,
                )
                accepted = bool(accepted_tensor)
                if accepted:
                    break
                total_refinements += 1

            if not accepted:
                raise RuntimeError(
                    "iFedDR relative-error condition was not satisfied after "
                    f"{max_refinements} refinement calls."
                )

            previous_local_point[selected] = local_point
            previous_server_point = server_point
            previous_alpha = adaptive_alpha
            cumulative_communications += 1
            cumulative_local_iterations += current_round_iterations
            synchronize(server_point)
            algorithm_time += time.perf_counter() - round_start
            process_time[round_index] = algorithm_time
            iterates[round_index] = server_point
            oracle_calling[round_index] = round_calls
            refinement_count[round_index] = refinement_calls
            communication_rounds[round_index] = cumulative_communications
            local_iterations[round_index] = cumulative_local_iterations
            if RecordStatistics:
                function_value[round_index], subgradient[round_index] = torch_statistics(
                    server_point, statistics_function, self.prox,
                    float(self.constant), stationarity_parameter,
                )

        self.final_state = {
            "local_s": local_s.clone(),
            "previous_local_point": previous_local_point.clone(),
            "reflected_direction": reflected_direction.clone(),
            "previous_server_point": previous_server_point.clone(),
            "previous_alpha": previous_alpha.clone(),
            "cumulative_communications": cumulative_communications,
            "cumulative_local_iterations": cumulative_local_iterations,
            "total_refinements": total_refinements,
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling, process_time, iterates, function_value,
            subgradient, index, refinement_count, communication_rounds,
            local_iterations,
        )

    def _local_approx_fmri(
        self,
        initial_model: FMRIModel,
        proximal_center: FMRIModel,
        objective_function: FMRIObjectiveFunction,
        proximal_parameter: Float,
        local_updates: int,
    ) -> FMRIModel:
        """Refine one native-volume local proximal point."""

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
        SigmaSquared: Float,
        InitialLocalUpdates: int,
        Function: list[FMRIObjectiveFunction],
        StatisticsFunction: FMRIValueFunction,
        AccuracyFunction: FMRIAccuracyFunction,
        RefinementFactor: Float = Float(2.0),
        MaxRefinements: int = 100,
        IncreaseInitialUpdates: bool = False,
        InitialIntercept: Float = Float(0.0),
        InitialState: dict[str, object] | None = None,
    ) -> FMRIAlgorithmResults:
        """Run inexact FedDR on a native MRI volume with continuation."""

        gamma = Float(ProximalParameter)
        relaxation = Float(RelaxationParameter)
        sigma_squared = Float(SigmaSquared)
        initial_local_updates = int(InitialLocalUpdates)
        del RefinementFactor
        max_refinements = int(MaxRefinements)
        if (
            gamma <= 0
            or not 0 < relaxation < 2
            or not 0 < sigma_squared < 1
            or initial_local_updates <= 0
            or max_refinements <= 0
        ):
            raise ValueError("Invalid fMRI iFedDR parameters.")
        rng = random.Random(self.seed)
        if InitialState is None:
            initial_model = FMRIModel(self.x_0.copy(), Float(InitialIntercept))
            local_s = [initial_model.copy() for _ in range(self.N)]
            previous_local_point = [initial_model.copy() for _ in range(self.N)]
            reflected_direction = [initial_model.copy() for _ in range(self.N)]
            previous_server_point = initial_model.copy()
            previous_alpha = Float(0.0)
            cumulative_communications = 0
            total_refinements = 0
            algorithm_time = 0.0
        else:
            local_s = [point.copy() for point in InitialState["local_s"]]
            previous_local_point = [
                point.copy() for point in InitialState["previous_local_point"]
            ]
            reflected_direction = [
                point.copy() for point in InitialState["reflected_direction"]
            ]
            previous_server_point = InitialState["previous_server_point"].copy()
            previous_alpha = Float(InitialState["previous_alpha"])
            cumulative_communications = int(
                InitialState["cumulative_communications"]
            )
            total_refinements = int(InitialState["total_refinements"])
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        oracle_calling = np.zeros(self.R, dtype=np.int64)
        process_time = np.zeros(self.R, dtype=np.float64)
        function_value = np.zeros(self.R, dtype=np.float64)
        test_accuracy = np.zeros(self.R, dtype=np.float64)
        index = np.zeros(self.R, dtype=np.bool_)
        refinement_count = np.zeros(self.R, dtype=np.int32)
        communication_rounds = np.zeros(self.R, dtype=np.int64)

        for iteration in range(self.R):
            round_start = time.perf_counter()
            selected_clients = rng.sample(range(self.N), self.s)
            for client in selected_clients:
                local_s[client] = local_s[client] - (
                    relaxation * previous_alpha
                ) * (
                    previous_local_point[client] - previous_server_point
                )

            local_points = [
                local_s[client].copy()
                for client in selected_clients
            ]
            multiplier = (
                max(1, total_refinements + 1)
                if IncreaseInitialUpdates
                else 1
            )
            round_local_updates = initial_local_updates * multiplier
            refinement_calls = 0
            base_quantity = Float(0.0)
            accepted = False

            while refinement_calls < max_refinements:
                for position, client in enumerate(selected_clients):
                    local_points[position] = self._local_approx_fmri(
                        local_points[position],
                        local_s[client],
                        Function[client],
                        gamma,
                        round_local_updates,
                    )
                    oracle_calling[iteration] += round_local_updates

                local_gradients: list[FMRIModel] = []
                for position, client in enumerate(selected_clients):
                    _, gradient = Function[client](local_points[position])
                    local_gradients.append(gradient)
                    reflected_direction[client] = (
                        local_points[position] - gamma * gradient
                    )
                    oracle_calling[iteration] += 1

                reflected_average = model_mean(
                    reflected_direction, self.parameter_shape
                )
                server_point = proximal_model(
                    reflected_average,
                    self.prox,
                    gamma,
                    self.constant,
                )

                xi = Float(0.0)
                zeta_numerator = Float(0.0)
                mu_scalar = Float(0.0)
                error_value = Float(0.0)
                for position, client in enumerate(selected_clients):
                    local_difference = local_points[position] - server_point
                    second_difference = (
                        gamma * local_gradients[position]
                        - local_s[client]
                        + server_point
                    )
                    proximal_residual = (
                        local_s[client]
                        - gamma * local_gradients[position]
                        - local_points[position]
                    )
                    alpha_direction = (
                        local_s[client]
                        - gamma * local_gradients[position]
                        - server_point
                    )
                    xi += squared_norm(local_difference)
                    zeta_numerator += squared_norm(second_difference)
                    mu_scalar += model_dot(local_difference, alpha_direction)
                    error_value += squared_norm(proximal_residual)

                zeta = zeta_numerator / gamma**2
                first_quantity = Float(np.sqrt(error_value))
                second_quantity = Float(
                    np.sqrt(sigma_squared * max(xi, zeta))
                )
                if base_quantity == 0.0:
                    base_quantity = Float(min(first_quantity, second_quantity))
                accepted = (
                    first_quantity <= second_quantity
                    or max(first_quantity, second_quantity)
                    <= 1e-13 + 1e-8 * base_quantity
                )
                refinement_calls += 1
                adaptive_alpha = Float(0.0 if xi == 0.0 else mu_scalar / xi)
                if accepted:
                    break

                total_refinements += 1

            if not accepted:
                raise RuntimeError(
                    "iFedDR relative-error condition was not satisfied after "
                    f"{max_refinements} refinement calls."
                )

            for position, client in enumerate(selected_clients):
                previous_local_point[client] = local_points[position]
            previous_server_point = server_point
            previous_alpha = adaptive_alpha
            cumulative_communications += 1
            refinement_count[iteration] = refinement_calls
            communication_rounds[iteration] = cumulative_communications
            algorithm_time += time.perf_counter() - round_start
            process_time[iteration] = algorithm_time
            function_value[iteration], test_accuracy[iteration] = record_statistics(
                server_point,
                StatisticsFunction,
                AccuracyFunction,
                self.prox,
                self.constant,
            )

        self.final_state = {
            "local_s": [point.copy() for point in local_s],
            "previous_local_point": [
                point.copy() for point in previous_local_point
            ],
            "reflected_direction": [
                point.copy() for point in reflected_direction
            ],
            "previous_server_point": previous_server_point.copy(),
            "previous_alpha": previous_alpha,
            "cumulative_communications": cumulative_communications,
            "total_refinements": total_refinements,
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return FMRIAlgorithmResults(
            oracle_calling=oracle_calling,
            process_time=process_time,
            function_value=function_value,
            test_accuracy=test_accuracy,
            final_point=previous_server_point.copy(),
            index=index,
            refinement_count=refinement_count,
            communication_rounds=communication_rounds,
        )
