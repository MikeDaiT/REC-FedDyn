import numpy as np
import numpy.typing as npt
from typing import TypeAlias

try:
    import torch
except ImportError:
    torch = None

FloatArray: TypeAlias = npt.NDArray[np.float64]
Float: TypeAlias = np.float64

# -------------------------------------------------------------------------------
#       Proximal operator property: 
#           \arg\min \gamma g(x) + \frac 12\|x-x_{center}\|^2
# -------------------------------------------------------------------------------

    
def l_1(
    point: FloatArray,
    parameter: Float,
    constant: Float,
    onlyValue: bool
    ) -> Float | FloatArray :
    r'''
        l1 norm proximal operator: 
            g(x) = \lambda \|x\|_1;
            point: current proximal center x_{center};
            parameter: algorithmic proximal parameter \eta;
            constant: problem regularization constant \gamma;
            onlyValue: 1 for return the function value at the current point,
                       0 for return the proximal point.
            Return: kernal value at the current point or proximal point.
    '''
    if torch is not None and isinstance(point, torch.Tensor):
        if onlyValue == 1:
            return float(constant) * torch.linalg.vector_norm(point, ord=1)
        if onlyValue == 0:
            threshold = float(constant) * float(parameter)
            return torch.sign(point) * torch.clamp(
                torch.abs(point) - threshold, min=0.0
            )
        raise ValueError('onlyValue must be True (1) or False (0).')
    if onlyValue == 1:
        return constant * np.linalg.norm(point, 1)
    elif onlyValue == 0:
        return np.maximum(np.abs(point) - constant * parameter, Float(0.0)) * np.sign(point)


def _gradient_nd(array: FloatArray) -> FloatArray:
    """Forward differences along every axis, with zero boundary differences."""

    gradient = np.zeros((array.ndim,) + array.shape, dtype=np.float64)
    for axis in range(array.ndim):
        lower = [slice(None)] * array.ndim
        upper = [slice(None)] * array.ndim
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        gradient[(axis, *lower)] = array[tuple(upper)] - array[tuple(lower)]
    return gradient


def _gradient_adjoint_nd(vectorField: FloatArray) -> FloatArray:
    """Adjoint of :func:`_gradient_nd`."""

    array_shape = vectorField.shape[1:]
    if vectorField.shape[0] != len(array_shape):
        raise ValueError('The vector field must have one component per array axis.')

    adjoint = np.zeros(array_shape, dtype=np.float64)
    for axis in range(len(array_shape)):
        lower = [slice(None)] * len(array_shape)
        upper = [slice(None)] * len(array_shape)
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        component = vectorField[(axis, *lower)]
        adjoint[tuple(lower)] -= component
        adjoint[tuple(upper)] += component
    return adjoint


def _tv_nd(array: FloatArray) -> Float:
    """Isotropic total variation of an N-dimensional array."""

    gradient = _gradient_nd(array)
    return Float(np.sum(np.sqrt(np.sum(gradient ** 2, axis=0))))


def _project_dual_nd(vectorField: FloatArray, radius: Float) -> FloatArray:
    """Project every N-dimensional dual vector onto an l2 ball."""

    norms = np.sqrt(np.sum(vectorField ** 2, axis=0))
    scales = np.maximum(Float(1.0), norms / radius)
    return vectorField / scales


# Backward-compatible names for code that imported the former 3-D helpers.
_gradient_3d = _gradient_nd
_gradient_adjoint_3d = _gradient_adjoint_nd
_tv_3d = _tv_nd
_project_dual_3d = _project_dual_nd


def TV(
    point: FloatArray,
    parameter: Float,
    constant: Float,
    onlyValue: bool,
    shape: tuple[int, ...] | None = None,
    accuracy: Float = Float(1e-5),
    maximumIteration: int = 200,
    ) -> Float | FloatArray:
    r'''
        N-dimensional isotropic total variation proximal operator:
            g(x) = \lambda TV(x);
            point: current proximal center x_{center};
            parameter: algorithmic proximal parameter \eta;
            constant: problem regularization constant \lambda;
            onlyValue: 1 for return the function value at the current point,
                       0 for return the approximating proximal point;
            shape: optional parameter shape. If omitted, the shape of ``point``
                   is used. Supplying it preserves support for flattened input;
            accuracy: relative primal-dual gap for the proximal approximation;
            maximumIteration: maximum number of FGP iterations.
            Return: regularization value at the current point or proximal point.

        The proximal point is approximated by Fast Gradient Projection (FGP)
        on the dual problem. The returned array has the same shape as point.
    '''
    pointArray = np.asarray(point, dtype=np.float64)

    resolvedShape = pointArray.shape if shape is None else tuple(shape)
    if len(resolvedShape) == 0 or any(size <= 0 for size in resolvedShape):
        raise ValueError('shape must contain at least one positive dimension.')
    if pointArray.size != int(np.prod(resolvedShape)):
        raise ValueError('The point size does not agree with shape.')
    if constant < 0 or parameter < 0:
        raise ValueError('constant and parameter must be nonnegative.')

    volume = pointArray.reshape(resolvedShape)

    if onlyValue == 1:
        return Float(constant * _tv_nd(volume))
    if onlyValue != 0:
        raise ValueError('onlyValue must be True (1) or False (0).')
    if accuracy <= 0:
        raise ValueError('accuracy must be positive.')
    if maximumIteration <= 0:
        raise ValueError('maximumIteration must be positive.')

    weight = Float(constant * parameter)
    if weight == 0:
        return pointArray.copy()

    # ||gradient||^2 <= 4d for d-dimensional forward differences.
    dualLipschitz = Float(4.0 * len(resolvedShape))
    dualPoint = np.zeros((len(resolvedShape),) + resolvedShape, dtype=np.float64)
    acceleratedPoint = dualPoint.copy()
    momentum = Float(1.0)

    bestPoint = volume.copy()
    bestGap = np.inf

    for _ in range(maximumIteration):
        candidate = volume - _gradient_adjoint_nd(acceleratedPoint)
        nextDualPoint = _project_dual_nd(
            acceleratedPoint + _gradient_nd(candidate) / dualLipschitz,
            weight,
        )

        candidate = volume - _gradient_adjoint_nd(nextDualPoint)
        candidateGradient = _gradient_nd(candidate)
        tvValue = Float(np.sum(np.sqrt(np.sum(candidateGradient ** 2, axis=0))))
        gap = Float(
            max(
                0.0,
                weight * tvValue - np.sum(candidateGradient * nextDualPoint),
            )
        )
        primalValue = Float(
            0.5 * np.sum((candidate - volume) ** 2) + weight * tvValue
        )

        if gap < bestGap:
            bestGap = gap
            bestPoint = candidate.copy()
        if gap <= accuracy * max(Float(1.0), primalValue):
            break

        nextMomentum = Float((1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0)
        acceleratedPoint = nextDualPoint + (
            (momentum - 1.0) / nextMomentum
        ) * (nextDualPoint - dualPoint)
        dualPoint = nextDualPoint
        momentum = nextMomentum

    return bestPoint.reshape(pointArray.shape)
        
