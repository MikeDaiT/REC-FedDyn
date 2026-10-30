import random
import time

import numpy as np
import torch

from Algorithms.Base_Optimizer import Float, FloatArray, Optimizer, ProxFunction
from Algorithms.REC_FedDyn import AlgorithmResults, ObjectiveFunction
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


class FedDyn(Optimizer):
    """FedDyn with exact or accelerated-gradient local subproblem solvers."""

    def __init__(
        self,
        InitialPoint: FloatArray,
        IterationBudget: int,
        ProxFunction: ProxFunction,
        ClientsNumber: int,
        SamplingNumber: int,
        SmoothL: Float,
        ConvexMu: Float,
        Constant: Float,
        RandomSeed: int | None = None,
    ) -> None:
        super().__init__(
            InitialPoint,
            IterationBudget,
            ProxFunction,
            ClientsNumber,
            SamplingNumber,
            SmoothL,
            ConvexMu,
            Constant,
            RandomSeed,
        )

    def go(
        self,
        PenaltyParameter: Float,
        Function: list[ObjectiveFunction],
        LocalHessian: FloatArray,
        LocalLinearTerm: FloatArray,
    ) -> AlgorithmResults:
        """
        Run FedDyn using exact local quadratic subproblem solutions.

        LocalHessian[i] is H_i in grad f_i(x) = H_i x - c_i.
        LocalLinearTerm[i] is c_i.
        """

        gamma = PenaltyParameter
        y = self.x_0.copy()
        dimension = self.x_0.size

        lamb = np.zeros((self.N, dimension), dtype=Float)
        h = np.zeros(dimension, dtype=Float)
        x = np.zeros((self.s, dimension), dtype=Float)

        rng = random.Random(self.seed)

        # Exact quadratic local solves do not query gradient oracles.
        oracleCalling = np.zeros(self.R, dtype=np.int64)
        processTime = np.zeros(self.R, dtype=Float)
        iterates = np.empty((self.R, dimension), dtype=Float)
        functionValue = np.zeros(self.R, dtype=Float)
        subGradient = np.zeros((self.R, dimension), dtype=Float)
        indeX = np.zeros(self.R, dtype=bool)

        identity = np.eye(dimension, dtype=Float)
        systemMatrices = LocalHessian + gamma * identity[None, :, :]
        choleskyFactors = np.linalg.cholesky(systemMatrices)

        algorithm_time = 0.0

        for ite in range(self.R):
            round_start = time.perf_counter()
            selectedClients = rng.sample(range(self.N), self.s)

            for position, client in enumerate(selectedClients):
                rightHandSide = LocalLinearTerm[client] + lamb[client] + gamma * y

                temporaryPoint = np.linalg.solve(choleskyFactors[client], rightHandSide)
                x[position] = np.linalg.solve(choleskyFactors[client].T, temporaryPoint)

                lamb[client] -= gamma * (x[position] - y)

            x_sum = np.sum(x, axis=0)
            h -= gamma * (x_sum - self.s * y) / self.N

            iterates[ite] = x_sum / self.s
            y = np.asarray(self.prox(iterates[ite] - h / gamma, 1.0 / gamma, self.constant, False), dtype=Float)

            algorithm_time += time.perf_counter() - round_start
            processTime[ite] = algorithm_time

            # The following evaluations are only used for monitoring.
            for function in Function:
                localValue, localGradient = function(y)
                functionValue[ite] += localValue
                subGradient[ite] += localGradient

            functionValue[ite] = functionValue[ite] / self.N + self.prox(y, 1.0, self.constant, True)
            subGradient[ite] = subGradient[ite] / self.N + gamma * (iterates[ite] - y) - h

        return AlgorithmResults(
            oracle_calling=oracleCalling,
            process_time=processTime,
            iterate_point=iterates,
            function_value=functionValue,
            subgradient=subGradient,
            index=indeX,
        )

    def go_exact_diagonal(
        self,
        PenaltyParameter: Float,
        Function: list[ObjectiveFunction],
        LocalHessianDiagonal: FloatArray,
        LocalLinearTerm: FloatArray,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        """Run exact FedDyn when every local quadratic has a diagonal Hessian.

        The local model is ``f_i(x) = 0.5 <x, H_i x> - <c_i, x>``.
        ``LocalHessianDiagonal[i]`` contains the diagonal of ``H_i`` and
        ``LocalLinearTerm[i]`` contains ``c_i``.  This representation retains
        the native image shape and avoids constructing dense image Hessians.
        """

        gamma = Float(PenaltyParameter)
        if gamma <= 0:
            raise ValueError("PenaltyParameter must be positive.")
        if not 1 <= self.s <= self.N:
            raise ValueError("SamplingNumber must lie between 1 and ClientsNumber.")

        expected_shape = (self.N,) + self.parameter_shape
        hessian_diagonal = np.asarray(LocalHessianDiagonal, dtype=self.dtype)
        linear_term = np.asarray(LocalLinearTerm, dtype=self.dtype)
        if hessian_diagonal.shape != expected_shape:
            raise ValueError(
                f"LocalHessianDiagonal must have shape {expected_shape}."
            )
        if linear_term.shape != expected_shape:
            raise ValueError(f"LocalLinearTerm must have shape {expected_shape}.")
        if np.any(hessian_diagonal + gamma <= 0):
            raise ValueError("Every diagonal local system must be positive definite.")

        rng = random.Random(self.seed)
        if InitialState is None:
            y = self.x_0.copy()
            lamb = np.zeros(expected_shape, dtype=self.dtype)
            h = np.zeros(self.parameter_shape, dtype=self.dtype)
            x = np.zeros((self.s,) + self.parameter_shape, dtype=self.dtype)
            algorithm_time = 0.0
        else:
            y = np.asarray(InitialState["y"], dtype=self.dtype).copy()
            lamb = np.asarray(InitialState["lamb"], dtype=self.dtype).copy()
            h = np.asarray(InitialState["h"], dtype=self.dtype).copy()
            x = np.asarray(InitialState["x"], dtype=self.dtype).copy()
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        oracle_calling = np.zeros(self.R, dtype=np.int64)
        process_time = np.zeros(self.R, dtype=self.dtype)
        iterates = np.empty((self.R,) + self.parameter_shape, dtype=self.dtype)
        function_value = np.zeros(self.R, dtype=self.dtype)
        subgradient_shape = (
            (self.R,) + self.parameter_shape
            if RecordStatistics
            else (self.R,)
        )
        subgradient = np.zeros(subgradient_shape, dtype=self.dtype)
        index = np.zeros(self.R, dtype=np.bool_)
        for ite in range(self.R):
            round_start = time.perf_counter()
            selected_clients = rng.sample(range(self.N), self.s)

            for position, client in enumerate(selected_clients):
                x[position] = self.exact_diagonal_local_solution(
                    y,
                    lamb[client],
                    gamma,
                    hessian_diagonal[client],
                    linear_term[client],
                )
                lamb[client] -= gamma * (x[position] - y)

            x_sum = np.sum(x, axis=0)
            h -= gamma * (x_sum - self.s * y) / self.N
            iterates[ite] = x_sum / self.s
            y = np.asarray(
                self.prox(
                    iterates[ite] - h / gamma,
                    Float(1.0 / gamma),
                    self.constant,
                    False,
                ),
                dtype=self.dtype,
            )

            algorithm_time += time.perf_counter() - round_start
            process_time[ite] = algorithm_time

            if RecordStatistics:
                for function in Function:
                    local_value, local_gradient = function(y)
                    function_value[ite] += local_value
                    subgradient[ite] += local_gradient
                function_value[ite] = (
                    function_value[ite] / self.N
                    + self.prox(y, Float(1.0), self.constant, True)
                )
                subgradient[ite] = (
                    subgradient[ite] / self.N
                    + gamma * (iterates[ite] - y) - h
                )

        self.final_state = {
            "y": y.copy(),
            "lamb": lamb.copy(),
            "h": h.copy(),
            "x": x.copy(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }

        return AlgorithmResults(
            oracle_calling=oracle_calling,
            process_time=process_time,
            iterate_point=iterates,
            function_value=function_value,
            subgradient=subgradient,
            index=index,
        )

    def exact_diagonal_local_solution(
        self,
        ServerPoint: FloatArray,
        LocalDual: FloatArray,
        PenaltyParameter: Float,
        HessianDiagonal: FloatArray,
        LinearTerm: FloatArray,
    ) -> FloatArray:
        """Solve one diagonal quadratic FedDyn local subproblem exactly."""

        gamma = Float(PenaltyParameter)
        denominator = np.asarray(HessianDiagonal, dtype=self.dtype) + gamma
        if gamma <= 0 or np.any(denominator <= 0):
            raise ValueError("The diagonal local system must be positive definite.")
        return (
            np.asarray(LinearTerm, dtype=self.dtype)
            + np.asarray(LocalDual, dtype=self.dtype)
            + gamma * np.asarray(ServerPoint, dtype=self.dtype)
        ) / denominator

    def go_approximate(
        self,
        PenaltyParameter: Float,
        LocalUpdates: int,
        Function: list[ObjectiveFunction],
        LocalLearningRate: Float | None = None,
        StationarityParameter: Float | None = None,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        """Run fixed-work FedDyn with Nesterov local subproblem solves."""

        alpha = Float(PenaltyParameter)
        local_updates = int(LocalUpdates)
        if alpha <= 0 or local_updates <= 0:
            raise ValueError("Invalid inexact FedDyn parameters.")
        if self.L == 0 and self.mu == 0:
            raise ValueError("Use go_torch for the neural-network ADAM case.")
        if self.L <= 0:
            raise ValueError(f"Unsupported values: L={self.L}, mu={self.mu}.")

        local_smoothness = Float(self.L + alpha)
        if self.mu > 0:
            local_strong_convexity = Float(self.mu + alpha)
        elif self.mu == 0:
            local_strong_convexity = alpha
        elif self.mu == -self.L:
            if alpha <= self.L:
                raise ValueError("Nonconvex case: inappropriate penalty parameter.")
            local_strong_convexity = Float(alpha - self.L)
        else:
            raise ValueError(f"Unsupported values: L={self.L}, mu={self.mu}.")

        learning_rate = (
            Float(1.0 / local_smoothness)
            if LocalLearningRate is None
            else Float(LocalLearningRate)
        )
        momentum = Float(
            (np.sqrt(local_smoothness) - np.sqrt(local_strong_convexity))
            / (np.sqrt(local_smoothness) + np.sqrt(local_strong_convexity))
        )
        stationarity_parameter = (
            Float(1.0 / self.L)
            if StationarityParameter is None
            else Float(StationarityParameter)
        )

        rng = random.Random(self.seed)
        if InitialState is None:
            y = self.x_0.copy()
            lamb = np.zeros((self.N,) + self.parameter_shape, dtype=Float)
            h = np.zeros(self.parameter_shape, dtype=Float)
            x = np.zeros((self.s,) + self.parameter_shape, dtype=Float)
            algorithm_time = 0.0
        else:
            y = np.asarray(InitialState["y"], dtype=Float).copy()
            lamb = np.asarray(InitialState["lamb"], dtype=Float).copy()
            h = np.asarray(InitialState["h"], dtype=Float).copy()
            x = np.asarray(InitialState["x"], dtype=Float).copy()
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        oracleCalling = np.zeros(self.R, dtype=np.int64)
        processTime = np.zeros(self.R, dtype=Float)
        # Shape: (communication rounds, *parameter_shape).
        iterates = np.empty((self.R,) + self.parameter_shape, dtype=Float)
        functionValue = np.zeros(self.R, dtype=Float)
        subGradient = np.zeros((self.R,) + self.parameter_shape, dtype=Float)
        indeX = np.zeros(self.R, dtype=bool)

        for ite in range(self.R):
            round_start = time.perf_counter()
            round_oracle_calling = 0
            selected_clients = rng.sample(range(self.N), self.s)

            for position, client in enumerate(selected_clients):
                local_point = y.copy()
                previous_gradient_point = y.copy()
                old_lamb = lamb[client].copy()

                for _ in range(local_updates):
                    _, local_gradient = Function[client](local_point)
                    gradient_point = local_point - learning_rate * (
                        local_gradient
                        - old_lamb
                        + alpha * (local_point - y)
                    )
                    local_point = gradient_point + momentum * (
                        gradient_point - previous_gradient_point
                    )
                    previous_gradient_point = gradient_point
                    round_oracle_calling += 1

                x[position] = local_point
                lamb[client] -= alpha * (local_point - y)

            x_sum = np.sum(x, axis=0)
            h -= alpha * (x_sum - self.s * y) / self.N
            iterates[ite] = x_sum / self.s
            y = np.asarray(
                self.prox(
                    iterates[ite] - h / alpha,
                    Float(1.0 / alpha),
                    self.constant,
                    False,
                ),
                dtype=Float,
            )

            algorithm_time += time.perf_counter() - round_start
            oracleCalling[ite] = round_oracle_calling
            processTime[ite] = algorithm_time

            if RecordStatistics:
                for function in Function:
                    local_value, local_gradient = function(y)
                    functionValue[ite] += local_value / self.N
                    subGradient[ite] += local_gradient / self.N
                functionValue[ite] += self.prox(
                    y, Float(1.0), self.constant, True
                )
                proximal_point = np.asarray(
                    self.prox(
                        y - stationarity_parameter * subGradient[ite],
                        stationarity_parameter,
                        self.constant,
                        False,
                    ),
                    dtype=Float,
                )
                subGradient[ite] = (y - proximal_point) / stationarity_parameter

        self.resolved_local_learning_rate = learning_rate
        self.resolved_local_momentum = momentum
        self.final_state = {
            "y": y.copy(),
            "lamb": lamb.copy(),
            "h": h.copy(),
            "x": x.copy(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling=oracleCalling,
            process_time=processTime,
            iterate_point=iterates,
            function_value=functionValue,
            subgradient=subGradient,
            index=indeX,
        )

    def _local_approx_torch(
        self,
        server_point: torch.Tensor,
        local_dual: torch.Tensor,
        objective: TorchObjective,
        penalty: float,
        local_updates: int,
        learning_rate: float | None,
    ) -> torch.Tensor:
        x = server_point.detach().clone()
        if self.L == 0 and self.mu == 0:
            step_size = 1e-3 if learning_rate is None else learning_rate
            for _ in range(local_updates):
                _, data_gradient = objective(x)
                gradient = (
                    data_gradient.detach() - local_dual
                    + penalty * (x - server_point)
                )
                x -= step_size * gradient
            return x

        local_smoothness = float(self.L + penalty)
        if self.mu > 0:
            local_strong_convexity = float(self.mu + penalty)
        elif self.mu == 0:
            local_strong_convexity = penalty
        elif self.mu == -self.L and penalty > self.L:
            local_strong_convexity = float(penalty - self.L)
        else:
            raise ValueError("Invalid FedDyn local subproblem parameters.")
        step_size = 1.0 / local_smoothness if learning_rate is None else learning_rate
        momentum = (
            (local_smoothness**0.5 - local_strong_convexity**0.5)
            / (local_smoothness**0.5 + local_strong_convexity**0.5)
        )
        previous_gradient_point = x.clone()
        for _ in range(local_updates):
            _, data_gradient = objective(x)
            gradient = (
                data_gradient.detach() - local_dual
                + penalty * (x - server_point)
            )
            gradient_point = x - step_size * gradient
            x = gradient_point + momentum * (
                gradient_point - previous_gradient_point
            )
            previous_gradient_point = gradient_point
        return x

    def go_torch(
        self,
        PenaltyParameter: float,
        LocalUpdates: int,
        Function: list[TorchObjective],
        LocalLearningRate: float | None = None,
        StationarityParameter: float | None = None,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        """Fixed-local-work FedDyn on torch tensors."""

        if not isinstance(self.x_0, torch.Tensor):
            raise TypeError("go_torch requires a torch InitialPoint.")
        penalty = float(PenaltyParameter)
        local_updates = int(LocalUpdates)
        if penalty <= 0 or local_updates <= 0:
            raise ValueError("Invalid FedDyn parameters.")
        stationarity_parameter = (
            (1.0 / penalty if self.L == 0 else 1.0 / float(self.L))
            if StationarityParameter is None
            else float(StationarityParameter)
        )
        rng = random.Random(self.seed)
        if InitialState is None:
            y = self.x_0.detach().clone()
            lamb = torch.zeros(
                (self.N,) + tuple(y.shape), dtype=y.dtype, device=y.device
            )
            h = torch.zeros_like(y)
            local_points = torch.zeros(
                (self.s,) + tuple(y.shape), dtype=y.dtype, device=y.device
            )
            algorithm_time = 0.0
        else:
            y = InitialState["y"].detach().clone()
            lamb = InitialState["lamb"].detach().clone()
            h = InitialState["h"].detach().clone()
            local_points = InitialState["local_points"].detach().clone()
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])

        oracle_calling = torch.zeros(self.R, dtype=torch.int64, device=y.device)
        process_time = y.new_zeros(self.R)
        iterates = y.new_empty((self.R,) + tuple(y.shape))
        function_value = y.new_zeros(self.R)
        neural_problem = self.L == 0 and self.mu == 0
        subgradient = (
            y.new_zeros(self.R) if neural_problem
            else y.new_zeros((self.R,) + tuple(y.shape))
        )
        index = torch.zeros(self.R, dtype=torch.bool, device=y.device)

        for round_index in range(self.R):
            synchronize(y)
            round_start = time.perf_counter()
            for position, client in enumerate(rng.sample(range(self.N), self.s)):
                old_dual = lamb[client].clone()
                local_points[position] = self._local_approx_torch(
                    y, old_dual, Function[client], penalty, local_updates,
                    LocalLearningRate,
                )
                lamb[client] -= penalty * (local_points[position] - y)
            local_average = torch.mean(local_points, dim=0)
            h -= penalty * self.s / self.N * (local_average - y)
            y = self.prox(
                local_average - h / penalty,
                1.0 / penalty,
                float(self.constant),
                False,
            ).detach()
            iterates[round_index] = y if neural_problem else local_average
            synchronize(y)
            algorithm_time += time.perf_counter() - round_start
            process_time[round_index] = algorithm_time
            oracle_calling[round_index] = self.s * local_updates

            if RecordStatistics:
                if neural_problem:
                    function_value[round_index], subgradient[round_index] = (
                        torch_statistics(
                            y, Function, self.prox, float(self.constant),
                            stationarity_parameter,
                        )
                    )
                    continue
                average_gradient = torch.zeros_like(y)
                for function in Function:
                    value, gradient = function(y)
                    function_value[round_index] += value.detach() / self.N
                    average_gradient += gradient.detach() / self.N
                function_value[round_index] += self.prox(
                    y, 1.0, float(self.constant), True
                )
                proximal_point = self.prox(
                    y - stationarity_parameter * average_gradient,
                    stationarity_parameter,
                    float(self.constant),
                    False,
                )
                subgradient[round_index] = (
                    y - proximal_point
                ) / stationarity_parameter

        self.final_state = {
            "y": y.clone(),
            "lamb": lamb.clone(),
            "h": h.clone(),
            "local_points": local_points.clone(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling, process_time, iterates, function_value,
            subgradient, index,
        )

    def go_fmri(
        self,
        PenaltyParameter: Float,
        LocalUpdates: int,
        Function: list[FMRIObjectiveFunction],
        StatisticsFunction: FMRIValueFunction,
        AccuracyFunction: FMRIAccuracyFunction,
        InitialIntercept: Float = Float(0.0),
        InitialState: dict[str, object] | None = None,
    ) -> FMRIAlgorithmResults:
        """Run inexact FedDyn with Nesterov local solves and continuation."""

        penalty = Float(PenaltyParameter)
        local_updates = int(LocalUpdates)
        local_smoothness = Float(self.L + penalty)
        local_strong_convexity = Float(self.mu + penalty)
        if (
            penalty <= 0
            or local_updates <= 0
            or local_smoothness <= 0
            or local_strong_convexity <= 0
        ):
            raise ValueError("Invalid fMRI FedDyn parameters.")
        momentum = Float(
            (np.sqrt(local_smoothness) - np.sqrt(local_strong_convexity))
            / (np.sqrt(local_smoothness) + np.sqrt(local_strong_convexity))
        )
        rng = random.Random(self.seed)
        if InitialState is None:
            y = FMRIModel(self.x_0.copy(), Float(InitialIntercept))
            local_dual = [
                FMRIModel.zeros(self.parameter_shape) for _ in range(self.N)
            ]
            surrogate_dual = FMRIModel.zeros(self.parameter_shape)
            local_points = [
                FMRIModel.zeros(self.parameter_shape) for _ in range(self.s)
            ]
            algorithm_time = 0.0
            completed_rounds = 0
        else:
            y = InitialState["y"].copy()
            local_dual = [point.copy() for point in InitialState["local_dual"]]
            surrogate_dual = InitialState["surrogate_dual"].copy()
            local_points = [point.copy() for point in InitialState["local_points"]]
            algorithm_time = float(InitialState["algorithm_time"])
            completed_rounds = int(InitialState["completed_rounds"])
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
            for position, client in enumerate(selected_clients):
                local_point = y.copy()
                previous_gradient_point = y.copy()
                previous_dual = local_dual[client].copy()
                for _ in range(local_updates):
                    _, local_gradient = Function[client](local_point)
                    gradient_point = local_point - (
                        local_gradient
                        - previous_dual
                        + penalty * (local_point - y)
                    ) / local_smoothness
                    local_point = gradient_point + momentum * (
                        gradient_point - previous_gradient_point
                    )
                    previous_gradient_point = gradient_point
                    oracle_calling[iteration] += 1

                local_points[position] = local_point
                local_dual[client] = local_dual[client] - penalty * (
                    local_point - y
                )

            local_average = model_mean(local_points, self.parameter_shape)
            surrogate_dual = surrogate_dual - (
                penalty * self.s / self.N
            ) * (local_average - y)
            y = proximal_model(
                local_average - surrogate_dual / penalty,
                self.prox,
                Float(1.0 / penalty),
                self.constant,
            )
            algorithm_time += time.perf_counter() - round_start
            process_time[iteration] = algorithm_time
            function_value[iteration], test_accuracy[iteration] = record_statistics(
                y,
                StatisticsFunction,
                AccuracyFunction,
                self.prox,
                self.constant,
            )

        self.resolved_fmri_local_learning_rate = Float(1.0 / local_smoothness)
        self.resolved_fmri_local_momentum = momentum
        self.final_state = {
            "y": y.copy(),
            "local_dual": [point.copy() for point in local_dual],
            "surrogate_dual": surrogate_dual.copy(),
            "local_points": [point.copy() for point in local_points],
            "algorithm_time": algorithm_time,
            "completed_rounds": completed_rounds + self.R,
            "rng_state": rng.getstate(),
        }
        return FMRIAlgorithmResults(
            oracle_calling=oracle_calling,
            process_time=process_time,
            function_value=function_value,
            test_accuracy=test_accuracy,
            final_point=y.copy(),
            index=index,
            refinement_count=refinement_count,
            communication_rounds=np.arange(
                completed_rounds + 1,
                completed_rounds + self.R + 1,
                dtype=np.int64,
            ),
        )
