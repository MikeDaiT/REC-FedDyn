from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

try:
    import torch
except ImportError:
    torch = None

FloatArray: TypeAlias = npt.NDArray[np.float64]
Float: TypeAlias = np.float64
ProxFunction: TypeAlias = Callable[[FloatArray, Float, Float, bool], Float | FloatArray]


def parameter_norm(point: FloatArray) -> Float:
    """Return the Euclidean norm for vectors and Frobenius norm otherwise."""

    point_array = np.asarray(point, dtype=np.float64)
    if point_array.ndim <= 1:
        return Float(np.linalg.norm(point_array))
    return Float(np.sqrt(np.sum(point_array * point_array)))



class Optimizer():
    '''Base class of optimization algorithms'''
    
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
        '''
        Base class property.
            Initial point (stored in its native parameter shape); Iteration budget;
            Regularization function(Point, Penalty parameter), 
                detail in the ProximalOperator module;
            Clients number N; Sampling number s;
            Gradient lipschitz constant: 
                L > 0 for L-lipschitz continuous gradient;
            Strongly convex constant:
                mu > 0 for strongly convex problem,
                mu = 0 for averaged strongly convex problem and general convex problem,
                mu = -L for nonconvex problem.
            Justify L before mu, L = 0 and mu = 0 for nonconvex neural network problem;
            Random seed: for the partial sampling.
        '''
        self.dtype = np.float64 # Numpy data type
        if torch is not None and isinstance(InitialPoint, torch.Tensor):
            self.x_0 = InitialPoint.detach().clone()
        else:
            self.x_0 = np.asarray(InitialPoint, dtype=self.dtype).copy()
        self.parameter_shape = self.x_0.shape # Problem's parameter shape
        self.R = IterationBudget
        self.prox = ProxFunction
        self.N = ClientsNumber
        self.s = SamplingNumber
        self.L = SmoothL
        self.mu = ConvexMu
        self.constant = Float(Constant)
        self.seed = RandomSeed
