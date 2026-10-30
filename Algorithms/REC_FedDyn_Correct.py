"""Corrected variant of the REC_FedDyn implementation."""

import random
import numpy as np
import numpy.typing as npt
import torch
from collections.abc import Callable
from typing import TypeAlias, Final
from Algorithms.Base_Optimizer import (
    Optimizer, FloatArray, Float, ProxFunction, parameter_norm,
)
from Algorithms.TorchUtils import (
    TorchObjective, norm as torch_norm, statistics as torch_statistics,
    synchronize,
)
from Algorithms.FMRI import (
    FMRIAccuracyFunction,
    FMRIAlgorithmResults,
    FMRIModel,
    FMRIObjectiveFunction,
    FMRIValueFunction,
    model_mean,
    model_norm,
    proximal_model,
    record_statistics,
)
from dataclasses import dataclass
import time
    
ObjectiveFunction: TypeAlias = Callable[[FloatArray], tuple[Float, FloatArray]]
@dataclass(slots=True)
class AlgorithmResults:
    oracle_calling: npt.NDArray[np.int64]
    process_time: npt.NDArray[Float]
    iterate_point: FloatArray
    function_value: FloatArray
    subgradient: FloatArray
    index: npt.NDArray[np.bool_]
    selected_clients: torch.Tensor | None = None
    local_oracle_calling: torch.Tensor | None = None
    local_criterion_satisfied: torch.Tensor | None = None



class REC_FedDyn(Optimizer):
    '''Distributed ADMM Variants(Renamed later)'''
    
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
        RandomSeed: int | None = None
        ) -> None:
        '''Algortihm property. '''
        super().__init__(InitialPoint, IterationBudget, ProxFunction, ClientsNumber, SamplingNumber, SmoothL, ConvexMu, Constant, RandomSeed)
    
        
    def local_approx(
        self, 
        y_r: FloatArray, 
        ObjectiveFunction: ObjectiveFunction, 
        LocalDual: FloatArray, 
        PenaltyParameter: Float, 
        sigma: Float,
        MaxIteration: int = 500,
        ) -> tuple[FloatArray, int, int]:
        '''
        Local clients' approximating computing.
            y_r: local approximation initial point;
            ObjectiveFunction: local objective function(current point), return the function value and the gradient;
            LocalDual: local client's dual information;
            sigma: relative error criterion parameter;
        Return:
            approximating point, local gradient calling number, and stopping
            criterion satisfying index.
        '''
        MAX_ITERATION: Final[int] = MaxIteration # max permitted number of iterations
        x = np.asarray(y_r, dtype=self.dtype).copy()
        y = x.copy()
        y_prev = x.copy()
        num = 0
        LocalD = LocalDual
        gamma = PenaltyParameter
        if self.L > 0 and self.mu > 0:
            L_Q = self.L + gamma
            mu_Q = self.mu + gamma
        elif self.L > 0 and self.mu == 0:
            L_Q = self.L + gamma
            mu_Q = gamma
        elif self.L > 0 and self.mu == -self.L:
            if gamma <= self.L:
                raise ValueError('Nonconvex case: inappropriate gamma.')
            L_Q = gamma + self.L
            mu_Q = gamma - self.L
        else:
            raise ValueError(f"Unsupported values: L={self.L}, mu={self.mu}.")
            
        # set numerical compared base
        baseQuantity = 0
        eps_absolute = 1e-13
        eps_comparison = 1e-8
        
        if self.L > 0:
            # Here for the Nesterov accelerated method, original analysis focus on the iterate y below, but x has the similar convergence guarantee and for the convenience of justifying the stopping criterion, we use iterate x in the experiments
            for t in range(MAX_ITERATION):
                beta = (np.sqrt(L_Q) - np.sqrt(mu_Q)) / (np.sqrt(L_Q) + np.sqrt(mu_Q))
                grad = ObjectiveFunction(x)[1] - LocalD + gamma * (x - y_r)
                num += 1
                
                A = parameter_norm(grad)
                B = sigma * gamma * parameter_norm(x - y_r)
                if baseQuantity == 0:
                    baseQuantity = np.min([A, B]) # base quantity initialized
                    
                relative_error_satisfied = (
                    A <= B
                    or np.max([A, B])
                    <= eps_absolute + eps_comparison * baseQuantity
                )
                if relative_error_satisfied:
                    return x - grad / gamma, num, 0
                else:
                    y = x - grad / L_Q
                    x = y + beta * (y - y_prev)
                    y_prev = y
            if t == MAX_ITERATION - 1:
                grad = ObjectiveFunction(x)[1] - LocalD + gamma * (x - y_r)
                num += 1
                return x - grad / gamma, num, 1
        elif self.L == 0 and self.mu == 0:
            pass
            # ADAM: need to be complete later, should adaptive to pytorch

    def local_approx_torch(
        self,
        server_point: torch.Tensor,
        objective: TorchObjective,
        local_dual: torch.Tensor,
        penalty: float,
        sigma: float,
        max_iterations: int = 500,
        local_learning_rate: float = 1e-3,
    ) -> tuple[torch.Tensor, int, int]:
        """Torch-native local solve with the current relative-error tests."""
        x = server_point.detach().clone()
        previous_gradient_point = x.clone()
        base_quantity = None

        if self.L > 0:
            local_smoothness = float(self.L + penalty)
            if self.mu > 0:
                local_strong_convexity = float(self.mu + penalty)
            elif self.mu == 0:
                local_strong_convexity = float(penalty)
            elif self.mu == -self.L and penalty > self.L:
                local_strong_convexity = float(penalty - self.L)
            else:
                raise ValueError("Invalid REC_FedDyn local subproblem parameters.")
            momentum = (
                (local_smoothness**0.5 - local_strong_convexity**0.5)
                / (local_smoothness**0.5 + local_strong_convexity**0.5)
            )
            use_gradient_descent = False
        elif self.L == 0 and self.mu == 0:
            local_smoothness = 0.0
            momentum = 0.0
            use_gradient_descent = True
        else:
            raise ValueError(f"Unsupported values: L={self.L}, mu={self.mu}.")

        for iteration in range(max_iterations + 1):
            _, data_gradient = objective(x)
            gradient = (
                data_gradient.detach() - local_dual
                + penalty * (x - server_point)
            )
            first = torch_norm(gradient)
            second = sigma * penalty * torch_norm(x - server_point)
            if base_quantity is None:
                base_quantity = torch.minimum(first, second)
            relative_error_satisfied = torch.logical_or(
                first <= second,
                torch.maximum(first, second) <= 1e-13 + 1e-8 * base_quantity,
            )
            if bool(relative_error_satisfied):
                return x - gradient / penalty, iteration + 1, 0

            if iteration == max_iterations:
                return x - gradient / penalty, iteration + 1, 1

            if use_gradient_descent:
                x -= local_learning_rate * gradient
            else:
                gradient_point = x - gradient / local_smoothness
                x = gradient_point + momentum * (
                    gradient_point - previous_gradient_point
                )
                previous_gradient_point = gradient_point

    def go_torch(
        self,
        PenaltyParameter: float,
        Sigma: float,
        Function: list[TorchObjective],
        MaxLocalIterations: int = 500,
        LocalLearningRate: float = 1e-3,
        StationarityParameter: float | None = None,
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
    ) -> AlgorithmResults:
        """Run REC_FedDyn without moving parameters or gradients off-device."""

        if not isinstance(self.x_0, torch.Tensor):
            raise TypeError("go_torch requires a torch InitialPoint.")
        penalty = float(PenaltyParameter)
        neural_problem = self.L == 0 and self.mu == 0
        stationarity_parameter = (
            1.0 / penalty if StationarityParameter is None
            else float(StationarityParameter)
        )
        rng = random.Random(self.seed)
        if InitialState is None:
            y = self.x_0.detach().clone()
            local_dual = torch.zeros(
                (self.N,) + tuple(y.shape), dtype=y.dtype, device=y.device
            )
            h = torch.zeros_like(y)
            local_points = torch.zeros(
                (self.s,) + tuple(y.shape), dtype=y.dtype, device=y.device
            )
            algorithm_time = 0.0
        else:
            y = InitialState["y"].detach().clone()
            local_dual = InitialState["local_dual"].detach().clone()
            h = InitialState["h"].detach().clone()
            local_points = InitialState["local_points"].detach().clone()
            algorithm_time = float(InitialState["algorithm_time"])
            rng.setstate(InitialState["rng_state"])
        oracle_calling = torch.zeros(self.R, dtype=torch.int64, device=y.device)
        process_time = y.new_zeros(self.R)
        iterates = y.new_empty((self.R,) + tuple(y.shape))
        function_value = y.new_zeros(self.R)
        subgradient = y.new_zeros(self.R)
        index = torch.zeros(self.R, dtype=torch.bool, device=y.device)
        selected_clients = torch.empty(
            (self.R, self.s), dtype=torch.int64, device=y.device
        )
        local_oracle_calling = torch.zeros_like(selected_clients)
        local_criterion_satisfied = torch.zeros(
            (self.R, self.s), dtype=torch.bool, device=y.device
        )

        for round_index in range(self.R):
            synchronize(y)
            round_start = time.perf_counter()
            selected = rng.sample(range(self.N), self.s)
            selected_clients[round_index] = torch.tensor(
                selected, dtype=torch.int64, device=y.device
            )
            for position, client in enumerate(selected):
                point, calls, failed = self.local_approx_torch(
                    y,
                    Function[client],
                    local_dual[client],
                    penalty,
                    float(Sigma),
                    MaxLocalIterations,
                    LocalLearningRate,
                )
                local_points[position] = point
                oracle_calling[round_index] += calls
                local_oracle_calling[round_index, position] = calls
                local_criterion_satisfied[round_index, position] = not bool(failed)
                local_dual[client] -= penalty * (point - y)
                index[round_index] |= bool(failed)

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

            if RecordStatistics:
                if neural_problem:
                    function_value[round_index], subgradient[round_index] = (
                        torch_statistics(
                            y, Function, self.prox, float(self.constant),
                            stationarity_parameter,
                        )
                    )
                else:
                    average_gradient = torch.zeros_like(y)
                    for function in Function:
                        value, gradient = function(y)
                        function_value[round_index] += value.detach() / self.N
                        average_gradient += gradient.detach() / self.N
                    function_value[round_index] += self.prox(
                        y, 1.0, float(self.constant), True
                    )
                    subgradient[round_index] = torch_norm(
                        average_gradient
                        + penalty * (local_average - y) - h
                    )

        self.final_state = {
            "y": y.clone(),
            "local_dual": local_dual.clone(),
            "h": h.clone(),
            "local_points": local_points.clone(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling, process_time, iterates, function_value,
            subgradient, index, selected_clients, local_oracle_calling,
            local_criterion_satisfied,
        )
                    

        
    def go(
        self, 
        PenaltyParameter: Float, 
        Sigma: Float, 
        Function: list[ObjectiveFunction],
        InitialState: dict[str, object] | None = None,
        RecordStatistics: bool = True,
        ) -> AlgorithmResults:
        '''
        Algorithm process.
            Function: a list of clients' objective function, 
                each function(point), return function value and gradient information;
            Penalty parameter; 
            Sigma: relative error stopping criterion parameter
        '''
        gamma = PenaltyParameter
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
        iterates = np.empty((self.R,) + self.parameter_shape, dtype=Float)
        functionValue = np.zeros(self.R, dtype=Float)
        subGradient = np.zeros(self.R, dtype=Float)
        indeX = np.zeros(self.R, dtype=bool)

        for ite in range(self.R):
            round_start = time.perf_counter()
            roundOracleCalling = 0

            i = 0
            for client in rng.sample(range(self.N), self.s):
                x_mid, num_mid, idx = self.local_approx(
                    y,
                    Function[client],
                    lamb[client],
                    gamma,
                    Sigma,
                )
                x[i] = np.asarray(x_mid, dtype=Float).copy()
                roundOracleCalling += num_mid
                lamb[client] -= gamma * (x[i] - y)
                if indeX[ite] == 0 and idx == 1:
                    indeX[ite] = 1
                i += 1
            x_sum = np.sum(x, 0)
            h -= gamma * (x_sum - self.s * y) / self.N
            iterates[ite, ...] = x_sum / self.s
            y = np.asarray(
                self.prox(
                    iterates[ite] - h / gamma,
                    1 / gamma,
                    self.constant,
                    False,
                ),
                dtype=Float,
            )
            algorithm_time += time.perf_counter() - round_start
            oracleCalling[ite] = roundOracleCalling
            processTime[ite] = algorithm_time

            if RecordStatistics:
                subgradient = np.zeros(self.parameter_shape, dtype=Float)
                for function in Function:
                    localValue, localGrad = function(y)
                    functionValue[ite] += localValue
                    subgradient += localGrad
                functionValue[ite] = functionValue[ite] / self.N + self.prox(y, 1.0, self.constant, True)
                subGradient[ite] = parameter_norm(subgradient / self.N + gamma * (iterates[ite] - y) - h)

        self.final_state = {
            "y": y.copy(),
            "lamb": lamb.copy(),
            "h": h.copy(),
            "x": x.copy(),
            "algorithm_time": algorithm_time,
            "rng_state": rng.getstate(),
        }
        return AlgorithmResults(
            oracle_calling=oracleCalling, process_time=processTime, iterate_point=iterates,
            function_value=functionValue, subgradient=subGradient, index=indeX)

    def _local_approx_fmri(
        self,
        initial_model: FMRIModel,
        objective_function: FMRIObjectiveFunction,
        local_dual: FMRIModel,
        penalty_parameter: Float,
        sigma: Float,
        maximum_iterations: int,
    ) -> tuple[FMRIModel, int, bool]:
        """Apply the original REC_FedDyn local solver to a native MRI volume."""

        x = initial_model.copy()
        gradient_point = x.copy()
        previous_gradient_point = x.copy()
        gamma = Float(penalty_parameter)
        if self.L > 0 and self.mu > 0:
            local_smoothness = self.L + gamma
            local_strong_convexity = self.mu + gamma
        elif self.L > 0 and self.mu == 0:
            local_smoothness = self.L + gamma
            local_strong_convexity = gamma
        elif self.L > 0 and self.mu == -self.L:
            if gamma <= self.L:
                raise ValueError("Nonconvex case: inappropriate gamma.")
            local_smoothness = gamma + self.L
            local_strong_convexity = gamma - self.L
        else:
            raise ValueError(f"Unsupported values: L={self.L}, mu={self.mu}.")

        beta = (
            np.sqrt(local_smoothness) - np.sqrt(local_strong_convexity)
        ) / (
            np.sqrt(local_smoothness) + np.sqrt(local_strong_convexity)
        )
        base_quantity = Float(0.0)
        for local_iteration in range(maximum_iterations):
            _, smooth_gradient = objective_function(x)
            gradient = (
                smooth_gradient - local_dual + gamma * (x - initial_model)
            )
            first_quantity = model_norm(gradient)
            second_quantity = sigma * gamma * model_norm(x - initial_model)
            if base_quantity == 0.0:
                base_quantity = Float(min(first_quantity, second_quantity))
            if (
                first_quantity <= second_quantity
                or max(first_quantity, second_quantity)
                <= 1e-13 + 1e-8 * base_quantity
            ):
                return x - gradient / gamma, local_iteration + 1, False

            gradient_point = x - gradient / local_smoothness
            x = gradient_point + beta * (
                gradient_point - previous_gradient_point
            )
            previous_gradient_point = gradient_point

        _, smooth_gradient = objective_function(x)
        gradient = smooth_gradient - local_dual + gamma * (x - initial_model)
        return x - gradient / gamma, maximum_iterations + 1, True

    def go_fmri(
        self,
        PenaltyParameter: Float,
        Sigma: Float,
        Function: list[FMRIObjectiveFunction],
        StatisticsFunction: FMRIValueFunction,
        AccuracyFunction: FMRIAccuracyFunction,
        InitialIntercept: Float = Float(0.0),
        MaxLocalIterations: int = 500,
        InitialState: dict[str, object] | None = None,
    ) -> FMRIAlgorithmResults:
        """Run REC_FedDyn on a 3-D coefficient volume plus scalar intercept."""

        gamma = Float(PenaltyParameter)
        sigma = Float(Sigma)
        if gamma <= 0 or not sigma > 0 or MaxLocalIterations <= 0:
            raise ValueError("Invalid fMRI REC_FedDyn parameters.")
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
                local_point, calls, failed = self._local_approx_fmri(
                    y,
                    Function[client],
                    local_dual[client],
                    gamma,
                    sigma,
                    MaxLocalIterations,
                )
                local_points[position] = local_point
                oracle_calling[iteration] += calls
                local_dual[client] = local_dual[client] - gamma * (
                    local_point - y
                )
                index[iteration] |= failed

            local_average = model_mean(local_points, self.parameter_shape)
            surrogate_dual = surrogate_dual - (
                gamma * self.s / self.N
            ) * (local_average - y)
            y = proximal_model(
                local_average - surrogate_dual / gamma,
                self.prox,
                Float(1.0 / gamma),
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
