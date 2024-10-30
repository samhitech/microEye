from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
from scipy import interpolate, optimize, stats


class CurveFitMethod(Enum):
    '''Methods for curve fitting'''

    CSPLINE = 'cspline'
    LINEAR = 'linear'
    SIGMOID = 'sigmoid'
    POLYNOMIAL = 'polynomial'
    PIECEWISE = 'piecewise'
    ASTIGMATIC_PSF = 'astigmatic_psf'

    def __str__(self):
        return self.value

    def __reduce__(self):
        '''__reduce__ function for pickle'''
        return (CurveFitMethod, (self.value,))


@dataclass
class CurveResult:
    '''Results from curve analysis'''

    method: CurveFitMethod
    parameters: dict
    r_squared: float
    derivative_max: float
    inflection_point: Optional[float] = None
    significant_region: tuple[float, float] = None

    def to_dict(self):
        return {
            'Method': [self.method.value],
            'R²': [self.r_squared],
            'Max dY/dZ': [self.derivative_max],
            'Inflection Point': [self.inflection_point],
            'Significant Region': list(self.significant_region)
            if self.significant_region
            else None,
            **{f'Parameter {k}': [v] for k, v in self.parameters.items()},
        }

    def get_spline_func(self) -> Optional[interpolate.UnivariateSpline]:
        '''Get the spline function if method is CSPLINE'''
        if self.method == CurveFitMethod.CSPLINE:
            return interpolate.UnivariateSpline(
                self.parameters['x_data'],
                self.parameters['y_data'],
                s=self.parameters['smoothing_factor'],
                k=self.parameters['degree'],
            )

        return None

    def get_data(self, x_data: np.ndarray):
        return CurveAnalyzer.get_data(x_data, self)


@dataclass
class PSFParameters:
    '''Parameters for astigmatic Gaussian PSF model'''

    sigma0x: float  # Base sigma x
    Ax: float  # First coefficient x
    Bx: float  # Second coefficient x
    sigma0y: float  # Base sigma y
    Ay: float  # First coefficient y
    By: float  # Second coefficient y
    gamma: float  # Scale factor
    d: float  # Characteristic distance

    def to_dict(self):
        return {
            'σ₀ₓ': self.sigma0x,
            'Aₓ': self.Ax,
            'Bₓ': self.Bx,
            'σ₀y': self.sigma0y,
            'Ay': self.Ay,
            'By': self.By,
            'γ': self.gamma,
            'd': self.d,
        }


class CurveAnalyzer:
    @staticmethod
    def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
        '''Generalized sigmoid function'''
        return L / (1 + np.exp(-k * (x - x0))) + b

    @staticmethod
    def sigmoid_derivative(
        x: np.ndarray, L: float, k: float, x0: float, b: float
    ) -> np.ndarray:
        '''Derivative of sigmoid function'''
        exp_term = np.exp(-k * (x - x0))
        return (L * k * exp_term) / (1 + exp_term) ** 2

    @staticmethod
    def sigma_model(z, sigma0, A0, B0, gamma, d):
        '''
        Gaussian PSF model for σ(z).

        Parameters
        ----------
        z : np.ndarray
            Z positions
        sigma0 : float
            σ at z=0
        A0 : float
            Cubic term coefficient
        B0 : float
            Quartic term coefficient
        gamma : float
            Z-offset
        d : float
            Characteristic length scale

        Returns
        -------
        np.ndarray
            σ values
        '''
        # Center and normalize z
        z_norm = (z - gamma) / d

        # Calculate σ(z)
        sigma = sigma0 * np.sqrt(1 + z_norm**2 + A0 * z_norm**3 + B0 * z_norm**4)

        return sigma

    @staticmethod
    def xy_sigma_model(z, sigma0x, Ax, Bx, sigma0y, Ay, By, gamma, d):
        '''
        Astigmatic Gaussian PSF model for σx(z) and σy(z).

        Parameters
        ----------
        z : np.ndarray
            Z positions
        sigma0x : float
            σx at z=0
        Ax : float
            Cubic term coefficient for σx
        Bx : float
            Quartic term coefficient for σx
        sigma0y : float
            σy at z=0
        Ay : float
            Cubic term coefficient for σy
        By : float
            Quartic term coefficient for σy
        gamma : float
            Z-offset
        d : float
            Characteristic length scale

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            σx and σy values
        '''
        # Center and normalize z
        z_norm = (z - gamma) / d

        # Calculate σx(z) and σy(z)
        sigma_x = sigma0x * np.sqrt(1 + z_norm**2 + Ax * z_norm**3 + Bx * z_norm**4)
        sigma_y = sigma0y * np.sqrt(1 + z_norm**2 + Ay * z_norm**3 + By * z_norm**4)

        return sigma_x, sigma_y

    @staticmethod
    def fit_stat_curve(
        selected_stat: str,
        get_stats: Callable,
        region: tuple[int, int],
        method: CurveFitMethod = CurveFitMethod.CSPLINE,
        derivative_threshold: float = 0.01,
        smoothing_factor: Optional[float] = 0,
    ) -> Optional[CurveResult]:
        '''
        Analyze the curve characteristics using specified method.

        Parameters
        ----------
        selected_stat : str
            The statistic to analyze
        get_stats : Callable
            Function to get the statistical data
        region : tuple[int, int]
            The z-region to analyze (start, end)
        method : CurveFitMethod
            Method to fit the curve
        derivative_threshold : float
            Threshold for considering dY/dZ significant
        smoothing_factor : Optional[float]
            Smoothing factor for cubic spline (None for auto)

        Returns
        -------
        Optional[CurveResult]
            Curve analysis results including fit parameters and characteristics
        '''
        if (
            selected_stat not in ['Sigma (diff)', 'Sigma (x/y)', 'Sigma² (diff)']
            or method == CurveFitMethod.ASTIGMATIC_PSF
        ):
            return None

        # Get data
        z_indices, stat_data, lower_bounds, upper_bounds = get_stats()

        # Select data within region
        mask = (z_indices >= region[0]) & (z_indices <= region[1])
        x_data = z_indices[mask]
        y_data = stat_data[mask]

        # Remove NaN values
        valid_mask = ~np.isnan(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]

        if len(x_data) < 4:
            return None

        if method == CurveFitMethod.SIGMOID:
            # Estimate initial parameters
            L = np.max(y_data) - np.min(y_data)
            b = np.min(y_data)
            x0 = x_data[len(x_data) // 2]  # Initial guess for midpoint
            k = 0.01  # Initial guess for steepness

            try:
                # Fit sigmoid
                popt, _ = optimize.curve_fit(
                    CurveAnalyzer.sigmoid,
                    x_data,
                    y_data,
                    p0=[L, k, x0, b],
                    bounds=([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
                )

                # Calculate R²
                y_fit = CurveAnalyzer.sigmoid(x_data, *popt)
                residuals = y_data - y_fit
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Find inflection point (maximum of derivative)
                x_fine = np.linspace(x_data[0], x_data[-1], 1000)
                dy_dx = CurveAnalyzer.sigmoid_derivative(x_fine, *popt)
                inflection_idx = np.argmax(dy_dx)
                inflection_point = x_fine[inflection_idx]
                max_derivative = dy_dx[inflection_idx]

                # Find region where derivative is significant
                significant_mask = dy_dx > derivative_threshold * max_derivative
                significant_region = (
                    x_fine[np.where(significant_mask)[0][0]],
                    x_fine[np.where(significant_mask)[0][-1]],
                )

                parameters = {'L': popt[0], 'k': popt[1], 'x0': popt[2], 'b': popt[3]}

                parameters['x_data'] = x_data.tolist()
                parameters['y_data'] = y_data.tolist()

                return CurveResult(
                    method=method,
                    parameters=parameters,
                    r_squared=r_squared,
                    derivative_max=max_derivative,
                    inflection_point=inflection_point,
                    significant_region=significant_region,
                )

            except (RuntimeError, optimize.OptimizeWarning):
                return None

        elif method == CurveFitMethod.POLYNOMIAL:
            # Implement polynomial fitting if sigmoid doesn't work well
            degree = 3  # Cubic polynomial by default
            coeffs = np.polyfit(x_data, y_data, degree)
            y_fit = np.polyval(coeffs, x_data)

            # Calculate R²
            residuals = y_data - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Find maximum derivative
            deriv_coeffs = np.polyder(coeffs)
            x_fine = np.linspace(x_data[0], x_data[-1], 1000)
            dy_dx = np.polyval(deriv_coeffs, x_fine)
            max_derivative = np.max(np.abs(dy_dx))

            # Find region where derivative is significant
            significant_mask = np.abs(dy_dx) > derivative_threshold * max_derivative
            significant_region = (
                x_fine[np.where(significant_mask)[0][0]],
                x_fine[np.where(significant_mask)[0][-1]],
            )

            parameters = {f'c{i}': c for i, c in enumerate(coeffs[::-1])}

            parameters['x_data'] = x_data.tolist()
            parameters['y_data'] = y_data.tolist()

            return CurveResult(
                method=method,
                parameters=parameters,
                r_squared=r_squared,
                derivative_max=max_derivative,
                significant_region=significant_region,
            )
        elif method == CurveFitMethod.CSPLINE:
            try:
                # Create cubic spline
                cs = interpolate.UnivariateSpline(
                    x_data, y_data, s=smoothing_factor, k=3
                )

                # Generate fine grid for analysis
                x_fine = np.linspace(x_data[0], x_data[-1], 1000)
                y_fit = cs(x_fine)

                # Calculate first derivative
                dy_dx = cs.derivative(1)(x_fine)
                max_derivative = np.max(np.abs(dy_dx))

                # Find inflection points (zeros of second derivative)
                d2y_dx2 = cs.derivative(2)(x_fine)
                inflection_indices = np.where(np.diff(np.signbit(d2y_dx2)))[0]

                if len(inflection_indices) > 0:
                    # Find inflection point with maximum absolute derivative
                    max_deriv_at_inflection = np.max(np.abs(dy_dx[inflection_indices]))
                    inflection_point = x_fine[
                        inflection_indices[np.argmax(np.abs(dy_dx[inflection_indices]))]
                    ]
                else:
                    inflection_point = None

                # Find region where derivative is significant
                significant_mask = np.abs(dy_dx) > derivative_threshold * max_derivative
                if np.any(significant_mask):
                    significant_region = (
                        x_fine[np.where(significant_mask)[0][0]],
                        x_fine[np.where(significant_mask)[0][-1]],
                    )
                else:
                    significant_region = None

                # Calculate R² on original data points
                y_fit_original = cs(x_data)
                residuals = y_data - y_fit_original
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Assuming `cs` is already created
                parameters = {
                    'x_data': x_data.tolist(),
                    'y_data': y_data.tolist(),
                    'degree': 3,
                    'smoothing_factor': smoothing_factor,
                }

                return CurveResult(
                    method=method,
                    parameters=parameters,
                    r_squared=r_squared,
                    derivative_max=max_derivative,
                    inflection_point=inflection_point,
                    significant_region=significant_region,
                )

            except (ValueError, RuntimeError):
                return None

        return None

    @staticmethod
    def get_z_from_y(
        y_value: float, curve_result: CurveResult, tolerance: float = 1e-6
    ) -> Optional[float]:
        '''
        Reverse extract Z value from Y using the fitted curve.

        Parameters
        ----------
        y_value : float
            The Y value to find Z for
        curve_result : CurveResult
            The curve fitting results
        tolerance : float
            Tolerance for numerical solution

        Returns
        -------
        Optional[float]
            The corresponding Z value if found
        '''
        if curve_result.method == CurveFitMethod.CSPLINE:
            # For spline, use numerical root finding on (f(x) - y_value)
            x_range = np.linspace(
                curve_result.significant_region[0],
                curve_result.significant_region[1],
                1000,
            )
            cs = curve_result.get_spline_func()
            y_spline = cs(x_range)

            # Find closest points to target y_value
            idx = np.argmin(np.abs(y_spline - y_value))
            z_approx = x_range[idx]

            # Refine using optimization
            def objective(x):
                return abs(cs(x) - y_value)

            result = optimize.minimize(
                objective,
                z_approx,
                bounds=[
                    (
                        curve_result.significant_region[0],
                        curve_result.significant_region[1],
                    )
                ],
            )

            if result.success and objective(result.x[0]) < tolerance:
                return result.x[0]

            return None
        elif curve_result.method == CurveFitMethod.SIGMOID:
            # For sigmoid, we can solve analytically
            L = curve_result.parameters['L']
            k = curve_result.parameters['k']
            x0 = curve_result.parameters['x0']
            b = curve_result.parameters['b']

            try:
                z = x0 - (1 / k) * np.log((L / (y_value - b)) - 1)
                # Check if z is within significant region
                if (
                    curve_result.significant_region
                    and curve_result.significant_region[0]
                    <= z
                    <= curve_result.significant_region[1]
                ):
                    return z
            except (ValueError, RuntimeWarning):
                return None

        elif curve_result.method == CurveFitMethod.POLYNOMIAL:
            # For polynomial, use numerical root finding
            coeffs = [
                curve_result.parameters[f'c{i}']
                for i in range(len(curve_result.parameters))
            ]
            poly = np.polynomial.Polynomial(coeffs) - y_value

            try:
                roots = poly.roots()
                real_roots = roots[np.abs(roots.imag) < tolerance].real

                # Filter roots within significant region
                valid_roots = [
                    r
                    for r in real_roots
                    if (
                        curve_result.significant_region
                        and curve_result.significant_region[0]
                        <= r
                        <= curve_result.significant_region[1]
                    )
                ]

                return valid_roots[0] if valid_roots else None

            except (ValueError, IndexError):
                return None

        return None

    @staticmethod
    def get_z_from_y_array(
        y_values: np.ndarray,
        curve_result: CurveResult,
        region: tuple[float, float] = None,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        '''
        Get Z values from array of Y values using the fitted curve.

        Parameters
        ----------
        y_values : np.ndarray
            Array of Y values to find Z for
        curve_result : CurveResult
            The curve fitting results
        region : tuple[float, float], optional
            The significant region for the curve
        tolerance : float
            Tolerance for numerical solution

        Returns
        -------
        np.ndarray
            Array of corresponding Z values,
            with np.nan for values where no solution is found
        '''
        z_values = np.full_like(y_values, np.nan)
        if region is None:
            region = curve_result.significant_region

        if curve_result.method == CurveFitMethod.CSPLINE:
            cs = curve_result.get_spline_func()
            x_range = np.linspace(
                region[0],
                region[1],
                1000,
            )
            y_spline = cs(x_range)

            # Vectorized initial approximation
            for i, y_value in enumerate(y_values):
                idx = np.argmin(np.abs(y_spline - y_value))
                z_approx = x_range[idx]

                # Refine using optimization
                def objective(x, y):
                    return abs(cs(x) - y)

                result = optimize.minimize(
                    objective,
                    z_approx,
                    args=(y_value,),
                    bounds=[
                        (
                            region[0],
                            region[1],
                        )
                    ],
                )

                if result.success and objective(result.x[0], y_value) < tolerance:
                    z_values[i] = result.x[0]

        elif curve_result.method == CurveFitMethod.SIGMOID:
            # Vectorized analytical solution for sigmoid
            L = curve_result.parameters['L']
            k = curve_result.parameters['k']
            x0 = curve_result.parameters['x0']
            b = curve_result.parameters['b']

            with np.errstate(divide='ignore', invalid='ignore'):
                z_values = x0 - (1 / k) * np.log((L / (y_values - b)) - 1)

            # Mask values outside significant region
            mask = (z_values >= region[0]) & (
                z_values <= region[1]
            )
            z_values[~mask] = np.nan

        return z_values

    @staticmethod
    def get_z_from_y_array_optimized(
        y_values: np.ndarray,
        curve_result: CurveResult,
        region: tuple[float, float] = None,
        tolerance: float = 1e-6,
        num_points: int = 10000,
    ) -> np.ndarray:
        '''
        Optimized version to get Z values from array of Y values using the fitted curve.
        Creates a dense lookup table for faster interpolation.

        Parameters
        ----------
        y_values : np.ndarray
            Array of Y values to find Z for
        curve_result : CurveResult
            The curve fitting results
        region : tuple[float, float], optional
            The significant region for the curve
        tolerance : float
            Tolerance for numerical solution
        num_points : int
            Number of points in the lookup table

        Returns
        -------
        np.ndarray
            Array of corresponding Z values,
            with np.nan for values where no solution is found
        '''
        z_values = np.full_like(y_values, np.nan)
        if region is None:
            region = curve_result.significant_region

        if curve_result.method == CurveFitMethod.CSPLINE:
            # Create dense lookup table
            z_lookup = np.linspace(
                region[0],
                region[1],
                num_points,
            )
            cs = curve_result.get_spline_func()
            y_lookup = cs(z_lookup)

            # Sort lookup table by y values for faster searching
            sort_idx = np.argsort(y_lookup)
            y_lookup = y_lookup[sort_idx]
            z_lookup = z_lookup[sort_idx]

            # Find nearest points using binary search
            indices = np.searchsorted(y_lookup, y_values)

            # Handle edge cases
            indices = np.clip(indices, 1, len(y_lookup) - 1)

            # Linear interpolation between nearest points
            y_low = y_lookup[indices - 1]
            y_high = y_lookup[indices]
            z_low = z_lookup[indices - 1]
            z_high = z_lookup[indices]

            # Avoid division by zero in interpolation
            valid_mask = np.abs(y_high - y_low) > tolerance
            t = np.zeros_like(y_values)
            t[valid_mask] = (y_values[valid_mask] - y_low[valid_mask]) / (
                y_high[valid_mask] - y_low[valid_mask]
            )

            z_values = z_low + t * (z_high - z_low)

            # Mask values outside the valid range
            valid_y_range = (y_values >= np.min(y_lookup)) & (
                y_values <= np.max(y_lookup)
            )
            z_values[~valid_y_range] = np.nan

        elif curve_result.method == CurveFitMethod.SIGMOID:
            L = curve_result.parameters['L']
            k = curve_result.parameters['k']
            x0 = curve_result.parameters['x0']
            b = curve_result.parameters['b']

            with np.errstate(divide='ignore', invalid='ignore'):
                z_values = x0 - (1 / k) * np.log((L / (y_values - b)) - 1)

            # Mask values outside significant region
            mask = (z_values >= region[0]) & (z_values <= region[1])
            z_values[~mask] = np.nan

        return z_values

    @staticmethod
    def get_data(x_data: np.ndarray, curve_result: CurveResult) -> np.ndarray:
        '''
        Get the fitted curve data for the specified X values.

        Parameters
        ----------
        x_data : np.ndarray
            The X values to evaluate the curve at
        curve_result : CurveResult
            The curve fitting results

        Returns
        -------
        np.ndarray
            The fitted curve data
        '''
        if isinstance(x_data, list):
            x_data = np.array(x_data)

        if curve_result.method == CurveFitMethod.CSPLINE:
            return curve_result.get_spline_func()(x_data)
        elif curve_result.method == CurveFitMethod.SIGMOID:
            return CurveAnalyzer.sigmoid(
                x_data,
                curve_result.parameters['L'],
                curve_result.parameters['k'],
                curve_result.parameters['x0'],
                curve_result.parameters['b'],
            )
        elif curve_result.method == CurveFitMethod.POLYNOMIAL:
            coeffs = [
                curve_result.parameters[f'c{i}']
                for i in range(len(curve_result.parameters))
            ]
            return np.polyval(coeffs, x_data)

        return np.zeros_like(x_data)
