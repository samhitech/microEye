from enum import Enum

from microEye import __version__

ABOUT_HTML = f'''
<h2>Dark Calibration Tool</h2>
<p><strong>Version:</strong> {__version__}</p>

<p>
This tool analyzes dark calibration data by fitting pixel <em>mean</em> and
<em>variance</em> across exposure times.
</p>

<h3>Data Requirements</h3>
<p>Each directory should contain:</p>

<ul>
    <li>
        <strong><code>mean.npy</code></strong><br>
        3D array: <code>(n_exposures, height, width)</code><br>
        Mean pixel intensities for each exposure time.
    </li>
    <li>
        <strong><code>variance.npy</code></strong><br>
        3D array: <code>(n_exposures, height, width)</code><br>
        Pixel intensity variance for each exposure time.
    </li>
    <li>
        <strong><code>exposure_times.npy</code></strong><br>
        1D array: <code>(n_exposures,)</code><br>
        Exposure times in milliseconds.
    </li>
    <li>
        <strong><code>temps.npy</code></strong> (optional)<br>
        2D array: <code>(n_exposures, m)</code><br>
        Temperature readings for each exposure time.
    </li>
</ul>

<h3>Notes</h3>
<ul>
    <li>Use <code>Standard</code> mode for trend/fitting plots.</li>
    <li>Use <code>Histograms</code> mode for distribution-based diagnostics.</li>
</ul>

<h3>Dataset JSON Format (Import)</h3>
<p>
You can import datasets from a JSON file. A minimal example is:
</p>

<pre><code>{{
    "dataset_01": {{
        "dark_calibration_directory": "C:/data/dark/dataset_01"
    }}
}}
</code></pre>

<p>
Optional fields per dataset: <code>name</code>, <code>gain</code>,
<code>responsivity</code>, <code>quantum_efficiency</code>.
</p>

<h3>Current Capabilities</h3>
<ul>
    <li>
        Import datasets from JSON with optional metadata fields:
        <code>dark_calibration_directory</code>, <code>gain</code>,
        <code>responsivity</code>, and <code>quantum_efficiency</code>.
    </li>
    <li>
        Dataset table shows loaded metadata (name, gain source, responsivity, QE,
        and source directory).
    </li>
    <li>
        <strong>Compare Datasets</strong> provides baseline, dark current,
        noise intercept, and noise slope comparisons with absolute/percent deltas.
    </li>
    <li>
        Results cache (<code>.npz</code>) preserves imported dataset metadata,
        including gain-related fields, and restores them on load.
    </li>
</ul>
'''


class DataTypes(Enum):
    MEAN = 'mean'
    VARIANCE = 'variance'
    EXPOSURE = 'exposure'
    TEMPERATURE = 'temperature'
    BASELINE = 'baseline'
    DARK_CURRENT = 'dark_current'
    DARK_NOISE = 'dark_noise'
    THERMAL_NOISE = 'thermal_noise'

    DARK_VARIANCE = 'dark_variance'
    THERMAL_VARIANCE = 'thermal_variance'

    def to_symbol(self) -> str:
        symbols = {
            DataTypes.MEAN: '$\\mu_d$',
            DataTypes.VARIANCE: '$\\sigma_d^2$',
            DataTypes.BASELINE: 'Baseline',
            DataTypes.DARK_CURRENT: 'Dark Current',
            DataTypes.DARK_NOISE: '$\\sigma_{d,~0}$',
            DataTypes.THERMAL_NOISE: '$\\sigma_{{thermal}}$',
            DataTypes.DARK_VARIANCE: '$\\sigma_{{d,~0}}^2$',
            DataTypes.THERMAL_VARIANCE: '$\\sigma_{{thermal}}^2$',
        }
        return symbols.get(self, self.value.replace('_', ' ').title())

    def to_unit(self, gain: float) -> str:
        unit = 'ADU' if gain == 1.0 else 'e-'
        units = {
            DataTypes.MEAN: unit,
            DataTypes.VARIANCE: f'{unit}$^2$',
            DataTypes.BASELINE: unit,
            DataTypes.DARK_CURRENT: f'{unit}/s',
            DataTypes.DARK_NOISE: unit,
            DataTypes.THERMAL_NOISE: f'{unit}/s$^{{0.5}}$',
            DataTypes.DARK_VARIANCE: f'{unit}$^2$',
            DataTypes.THERMAL_VARIANCE: f'{unit}$^2$/s',
        }
        return units.get(self, '')


FILE_NAMES = {
    DataTypes.MEAN: 'mean.npy',
    DataTypes.VARIANCE: 'variance.npy',
    DataTypes.EXPOSURE: 'exposure_times.npy',
    DataTypes.TEMPERATURE: 'temps.npy',
}

RESULTS_SCHEMA_VERSION = 2

HISTOGRAM_DATA_TYPES = (
    DataTypes.BASELINE,
    DataTypes.DARK_CURRENT,
    DataTypes.DARK_NOISE,
    DataTypes.THERMAL_NOISE,
)
