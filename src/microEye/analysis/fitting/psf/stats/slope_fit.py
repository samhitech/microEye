from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import stats

from microEye.analysis.fitting.psf.stats.core import ConfidenceMethod
from microEye.analysis.fitting.psf.stats.curve_fit import CurveFitMethod


@dataclass
class SlopeResult:
    '''Results from slope analysis'''
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    slope_ci: tuple[float, float]
    data : dict
    zero_crossing: Optional[float] = None
    zero_crossing_ci: Optional[tuple[float, float]] = None
    method: CurveFitMethod = CurveFitMethod.LINEAR

class SlopeAnalyzer:
    @staticmethod
    def fit_stat_slope(
        selected_stat: str,
        get_stats: Callable,
        region: tuple[int, int],
        confidence_method: ConfidenceMethod = ConfidenceMethod.T_DIST,
        confidence_level: float = 0.95,
    ) -> Optional[SlopeResult]:
        '''
        Calculate slope and related statistics for the
        selected statistic in the given region.

        Parameters
        ----------
        selected_stat : str
            The statistic to analyze
        stats_calculator : StatsCalculator
            Calculator for the statistic
        region : tuple[int, int]
            The z-region to analyze (start, end)
        confidence_method : ConfidenceMethod
            Method to calculate confidence intervals
        confidence_level : float
            Confidence level for intervals (0-1)

        Returns
        -------
        Optional[SlopeResult]
            Slope analysis results including confidence intervals and zero crossing
        '''
        if selected_stat not in ['Sigma (diff)', 'Sigma (x/y)', 'Sigma² (diff)']:
            return None


        # Get z values and stat data
        z_indices, stat_data, lower_bounds, upper_bounds = get_stats()

        # Select data within region
        mask = (z_indices >= region[0]) & (z_indices <= region[1])
        x_data = z_indices[mask]
        y_data = stat_data[mask]

        # Remove any NaN values
        valid_mask = ~np.isnan(y_data)
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]

        if len(x_data) < 2:
            return None

        # Perform linear regression
        slope, intercept, r_value, p_value, slope_std_err = stats.linregress(
            x_data, y_data
        )
        r_squared = r_value**2

        # Calculate slope confidence intervals
        if confidence_method == ConfidenceMethod.T_DIST:
            # t-distribution based CI
            dof = len(x_data) - 2
            t_val = stats.t.ppf((1 + confidence_level) / 2, dof)
            slope_ci = (slope - t_val * slope_std_err, slope + t_val * slope_std_err)
        elif confidence_method == ConfidenceMethod.BOOTSTRAP:
            # Bootstrap confidence intervals for slope
            n_bootstrap = 1000
            bootstrap_slopes = []
            for _ in range(n_bootstrap):
                indices = np.random.randint(0, len(x_data), size=len(x_data))
                x_boot = x_data[indices]
                y_boot = y_data[indices]
                boot_slope = np.polyfit(x_boot, y_boot, 1)[0]
                bootstrap_slopes.append(boot_slope)
            slope_ci = np.percentile(
                bootstrap_slopes,
                [(1 - confidence_level) * 100 / 2, (1 + confidence_level) * 100 / 2],
            )
        else:
            # Simple min/max based on data uncertainty
            lower_slope = np.polyfit(x_data, lower_bounds[mask][valid_mask], 1)[0]
            upper_slope = np.polyfit(x_data, upper_bounds[mask][valid_mask], 1)[0]
            slope_ci = (lower_slope, upper_slope)

        # Calculate zero crossing
        target = 0 if selected_stat == 'Sigma (diff)' else 1
        zero_crossing = (target - intercept) / slope if abs(slope) > 1e-10 else None

        # Calculate zero crossing confidence intervals
        if zero_crossing is not None:
            if confidence_method == ConfidenceMethod.T_DIST:
                # Use error propagation for zero crossing uncertainty
                intercept_std_err = stats.linregress(x_data, y_data).intercept_stderr
                zero_crossing_err = np.sqrt(
                    (intercept_std_err / slope) ** 2
                    + ((target - intercept) * slope_std_err / slope**2) ** 2
                )
                t_val = stats.t.ppf((1 + confidence_level) / 2, len(x_data) - 2)
                zero_crossing_ci = (
                    zero_crossing - t_val * zero_crossing_err,
                    zero_crossing + t_val * zero_crossing_err,
                )
            elif confidence_method == ConfidenceMethod.BOOTSTRAP:
                # Bootstrap zero crossings
                bootstrap_crossings = []
                for _ in range(n_bootstrap):
                    indices = np.random.randint(0, len(x_data), size=len(x_data))
                    x_boot = x_data[indices]
                    y_boot = y_data[indices]
                    boot_slope, boot_intercept = np.polyfit(x_boot, y_boot, 1)
                    if abs(boot_slope) > 1e-10:
                        boot_crossing = (target - boot_intercept) / boot_slope
                        bootstrap_crossings.append(boot_crossing)
                if bootstrap_crossings:
                    zero_crossing_ci = np.percentile(
                        bootstrap_crossings,
                        [
                            (1 - confidence_level) * 100 / 2,
                            (1 + confidence_level) * 100 / 2,
                        ],
                    )
                else:
                    zero_crossing_ci = None
            else:
                # Simple interval based on slope confidence intervals
                crossings = [
                    (target - intercept) / s for s in slope_ci if abs(s) > 1e-10
                ]
                zero_crossing_ci = (
                    (min(crossings), max(crossings)) if crossings else None
                )
        else:
            zero_crossing_ci = None

        return SlopeResult(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            slope_ci=slope_ci,
            zero_crossing=zero_crossing,
            zero_crossing_ci=zero_crossing_ci,
            data = {
                'z': x_data.tolist(),
                'ratio': y_data.tolist(),
                'lower_bounds': lower_bounds[mask][valid_mask].tolist(),
                'upper_bounds': upper_bounds[mask][valid_mask].tolist()
            }
        )
