import numpy as np
import numpy.typing as npt
from typing import TypeAlias

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:
    torch = None
    torch_functional = None

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


class OverlappingGroupLasso:
    r'''Spatial overlapping-group-lasso value and dual-FISTA proximal map.

    The regularized part of a flat neural-network parameter vector is viewed
    as ``(hidden_units, image_height, image_width)``.  Every spatial patch
    contains the weights from all hidden units to the pixels in that patch.
    The implemented regularizer is

    .. math::

        g(w) = \frac{\lambda}{|\mathcal G|}
               \sum_{G\in\mathcal G}\|(D w)_G\|_2,

    where ``D`` optionally divides every pixel column by the square root of
    its group-membership count.  This removes the otherwise artificial extra
    penalty on interior pixels.  Biases and all parameters outside the
    configured flat slice are left unchanged by the proximal map.

    The proximal point is computed from the smooth constrained dual using
    projected FISTA.  Patch extraction and its adjoint are implemented with
    ``unfold``/``fold`` so the group-selection matrix is never materialized.
    With membership normalization, the analysis operator satisfies
    ``A.T @ A = I`` and the dual-gradient Lipschitz constant is one.

    Instances are callable and follow the four-argument interface used by the
    optimizers in this project.  NumPy inputs are supported through the same
    CPU Torch implementation; FEMNIST training uses native Torch tensors.
    '''

    def __init__(
        self,
        weight_start: int,
        hidden_units: int,
        image_shape: tuple[int, int] = (28, 28),
        patch_size: int = 4,
        stride: int = 2,
        accuracy: float = 1e-6,
        maximum_iterations: int = 200,
        membership_normalization: bool = True,
        warm_start: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("OverlappingGroupLasso requires PyTorch.")
        height, width = map(int, image_shape)
        if weight_start < 0 or hidden_units <= 0:
            raise ValueError("weight_start must be nonnegative and hidden_units positive.")
        if height <= 0 or width <= 0:
            raise ValueError("image_shape must contain two positive dimensions.")
        if patch_size <= 0 or patch_size > min(height, width):
            raise ValueError("patch_size must fit inside the image.")
        if stride <= 0 or stride > patch_size:
            raise ValueError("stride must lie between one and patch_size.")
        if accuracy <= 0.0 or maximum_iterations <= 0:
            raise ValueError("accuracy and maximum_iterations must be positive.")

        row_starts = tuple(range(0, height - patch_size + 1, stride))
        column_starts = tuple(range(0, width - patch_size + 1, stride))
        coverage = np.zeros((height, width), dtype=np.float64)
        for row in row_starts:
            for column in column_starts:
                coverage[
                    row:row + patch_size,
                    column:column + patch_size,
                ] += 1.0
        if np.any(coverage == 0.0):
            raise ValueError(
                "The patch grid does not cover every pixel; choose a compatible "
                "image shape, patch size, and stride."
            )

        self.weight_start = int(weight_start)
        self.hidden_units = int(hidden_units)
        self.image_shape = (height, width)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.accuracy = float(accuracy)
        self.maximum_iterations = int(maximum_iterations)
        self.membership_normalization = bool(membership_normalization)
        self.warm_start = bool(warm_start)
        self.group_count = len(row_starts) * len(column_starts)
        self.group_size = self.hidden_units * self.patch_size**2
        self.weight_size = self.hidden_units * height * width
        self._coverage = torch.as_tensor(coverage)
        self._dual_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}

        self.proximal_calls = 0
        self.total_dual_iterations = 0
        self.last_iterations = 0
        self.last_gap = float("nan")

    def _device_key(self, point: torch.Tensor) -> tuple[str, int | None, torch.dtype]:
        return point.device.type, point.device.index, point.dtype

    def _inverse_sqrt_coverage(self, point: torch.Tensor) -> torch.Tensor:
        coverage = self._coverage.to(device=point.device, dtype=point.dtype)
        if self.membership_normalization:
            return coverage.rsqrt()
        return torch.ones_like(coverage)

    def _analysis(self, volume: torch.Tensor) -> torch.Tensor:
        scale = self._inverse_sqrt_coverage(volume)
        patches = torch_functional.unfold(
            (volume * scale).unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.stride,
        )
        return patches.squeeze(0).transpose(0, 1)

    def _adjoint(self, groups: torch.Tensor) -> torch.Tensor:
        height, width = self.image_shape
        folded = torch_functional.fold(
            groups.transpose(0, 1).unsqueeze(0),
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.stride,
        ).squeeze(0)
        return folded * self._inverse_sqrt_coverage(folded)

    @staticmethod
    def _project_groups(groups: torch.Tensor, radius: float) -> torch.Tensor:
        norms = torch.linalg.vector_norm(groups, dim=1, keepdim=True)
        tiny = torch.finfo(groups.dtype).tiny
        scales = torch.clamp(norms / max(radius, tiny), min=1.0)
        return groups / scales

    def _regularized_volume(self, point: torch.Tensor) -> torch.Tensor:
        flat = point.reshape(-1)
        stop = self.weight_start + self.weight_size
        if stop > flat.numel():
            raise ValueError(
                "The configured first-layer weight slice exceeds the point size."
            )
        return flat[self.weight_start:stop].reshape(
            self.hidden_units, *self.image_shape
        )

    def value(self, point: torch.Tensor, constant: float) -> torch.Tensor:
        if constant < 0.0:
            raise ValueError("constant must be nonnegative.")
        volume = self._regularized_volume(point)
        group_norms = torch.linalg.vector_norm(self._analysis(volume), dim=1)
        return float(constant) * group_norms.mean()

    def inactive_group_fraction(
        self,
        point: torch.Tensor | FloatArray,
        tolerance: float = 1e-6,
    ) -> float:
        if tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative.")
        tensor = (
            point
            if isinstance(point, torch.Tensor)
            else torch.as_tensor(np.asarray(point), dtype=torch.float64)
        )
        norms = torch.linalg.vector_norm(
            self._analysis(self._regularized_volume(tensor)), dim=1
        )
        return float((norms <= tolerance).to(torch.float64).mean())

    def reset_statistics(self, clear_warm_start: bool = True) -> None:
        self.proximal_calls = 0
        self.total_dual_iterations = 0
        self.last_iterations = 0
        self.last_gap = float("nan")
        if clear_warm_start:
            self._dual_cache.clear()

    def _torch_call(
        self,
        point: torch.Tensor,
        parameter: float,
        constant: float,
        only_value: bool,
    ) -> torch.Tensor:
        if parameter < 0.0 or constant < 0.0:
            raise ValueError("parameter and constant must be nonnegative.")
        if only_value:
            return self.value(point, constant)

        flat = point.reshape(-1)
        volume = self._regularized_volume(point)
        radius = float(parameter) * float(constant) / self.group_count
        if radius == 0.0:
            return point.detach().clone()

        key = self._device_key(point)
        if self.warm_start and key in self._dual_cache:
            dual = self._project_groups(
                self._dual_cache[key].to(device=point.device, dtype=point.dtype),
                radius,
            )
        else:
            dual = volume.new_zeros((self.group_count, self.group_size))
        accelerated = dual.clone()
        momentum = 1.0

        # Without membership normalization, ||A||^2 is the maximum pixel
        # membership count.  With normalization it is exactly one.
        if self.membership_normalization:
            dual_lipschitz = 1.0
        else:
            dual_lipschitz = float(self._coverage.max())

        best_point = volume.detach().clone()
        best_dual = dual.detach().clone()
        best_gap = float("inf")
        iterations = 0

        with torch.no_grad():
            for iteration in range(1, self.maximum_iterations + 1):
                candidate = volume - self._adjoint(accelerated)
                next_dual = self._project_groups(
                    accelerated + self._analysis(candidate) / dual_lipschitz,
                    radius,
                )

                candidate = volume - self._adjoint(next_dual)
                analyzed_candidate = self._analysis(candidate)
                group_norm_sum = torch.linalg.vector_norm(
                    analyzed_candidate, dim=1
                ).sum()
                gap_tensor = torch.clamp_min(
                    radius * group_norm_sum
                    - torch.sum(analyzed_candidate * next_dual),
                    0.0,
                )
                primal_tensor = (
                    0.5 * torch.sum((candidate - volume).square())
                    + radius * group_norm_sum
                )
                gap = float(gap_tensor)
                primal = float(primal_tensor)
                iterations = iteration
                if not np.isfinite(gap) or not np.isfinite(primal):
                    raise FloatingPointError(
                        "Overlapping-group-lasso dual FISTA received a "
                        "non-finite point; the optimization candidate has "
                        "numerically diverged."
                    )
                if gap < best_gap:
                    best_gap = gap
                    best_point = candidate.detach().clone()
                    best_dual = next_dual.detach().clone()
                if gap <= self.accuracy * max(1.0, abs(primal)):
                    break

                next_momentum = (
                    1.0 + (1.0 + 4.0 * momentum**2) ** 0.5
                ) / 2.0
                accelerated = next_dual + (
                    (momentum - 1.0) / next_momentum
                ) * (next_dual - dual)
                dual = next_dual
                momentum = next_momentum

        result = flat.detach().clone()
        stop = self.weight_start + self.weight_size
        result[self.weight_start:stop] = best_point.reshape(-1)
        self.proximal_calls += 1
        self.total_dual_iterations += iterations
        self.last_iterations = iterations
        self.last_gap = best_gap
        if self.warm_start:
            self._dual_cache[key] = best_dual
        return result.reshape(point.shape)

    def __call__(
        self,
        point: torch.Tensor | FloatArray,
        parameter: float,
        constant: float,
        onlyValue: bool,
    ) -> torch.Tensor | FloatArray | Float:
        if onlyValue not in (0, 1, False, True):
            raise ValueError("onlyValue must be True (1) or False (0).")
        if isinstance(point, torch.Tensor):
            return self._torch_call(
                point, float(parameter), float(constant), bool(onlyValue)
            )

        array = np.asarray(point)
        tensor = torch.as_tensor(array, dtype=torch.float64)
        result = self._torch_call(
            tensor, float(parameter), float(constant), bool(onlyValue)
        )
        if bool(onlyValue):
            return Float(result)
        return result.cpu().numpy().astype(array.dtype, copy=False)


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
        
