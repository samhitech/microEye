from enum import Enum

from microEye import __version__

ABOUT_HTML = f'''
<h2>Photon Transfer Curve Analysis Tool</h2>
<p><strong>Version:</strong> {__version__}</p>

<p>
This tool analyzes photon transfer curves (PTC) to characterize camera
sensor performance, including gain, read noise, quantum efficiency (QE),
and signal-to-noise ratio (SNR).
</p>

<h3>Data Requirements</h3>

<h4>Fixed Irradiance, Variable Exposure Method:</h4>


<p>
The tool uses the fixed irradiance method, where the laser power is constant and
exposure time varies to achieve different photon flux levels.</p>

<p>Each dataset requires:</p>

<ul>
    <li>
        <strong><code>Signal .npy</code></strong><br>
        4D array: <code>(n_exposures, 2, height, width)</code><br>
        Signal frames with 2 samples per exposure time for temporal
        variance calculation.
    </li>
    <li>
        <strong><code>Dark .npy</code></strong><br>
        4D array: <code>(n_exposures, 2, height, width)</code><br>
        Dark frames with 2 samples per exposure time for read noise
        estimation.
    </li>
</ul>

<p>
Where <code>n_exposures</code> is
<code>[min_exposure, 0.001, 0.002, ..., 0.200]</code> seconds (201 points).</p>

<p>
<strong>Note:</strong> the 2 frames for dark and signal are extracted from the middle
of an acquisition series to avoid any transient effects.
</p>

<h3>Sphere Power Calibration</h3>
<p>
Before running PTC analysis, you must perform sphere power calibration to
establish the irradiance at the sensor plane. The calibration requires:
</p>

<ul>
    <li>
        <strong>Calibration Data File</strong><br>
        CSV/HDF file with columns: <code>laser_power</code> (mW),
        <code>port</code> (mW), <code>camera</code> (µW)
    </li>
    <li>
        <strong>Photodiode Diameter</strong><br>
        Reference photodiode diameter (default 10 mm) for irradiance
        calculation: <code>irradiance = power / (π × (diameter/2)²)</code>
    </li>
</ul>

<p>The calibration establishes two linear models:</p>
<ul>
    <li><strong>Power → Port Irradiance:</strong>
    Laser power (mW) to port sensor reading converted to irradiance (mW/cm²)</li>
    <li><strong>Port → Camera Irradiance:</strong>
    Sphere port irradiance (mW/cm²) to camera plane irradiance (mW/cm²)</li>
</ul>

<p>
The tool automatically finds the best linear segment with high R² (≥0.9999)
to avoid saturation, then uses these models to convert measurement data to
calibrated irradiance values for PTC analysis.
</p>

<h3>Dataset JSON Format (Import)</h3>
<p>
Optionally import multiple datasets from a JSON file with the following
structure:
</p>

<pre><code>{{
    "dataset_name_1": {{
        "signal": "/path/to/signal.npy",
        "dark": "/path/to/dark.npy",
        "port_power": 0.05, # in mW (from sphere port meter)
        "pixel_size_um": 6.5,
        "wavelength_nm": 488.0
    }},
    "dataset_name_2": {{
        "signal": "/path/to/signal2.npy",
        "dark": "/path/to/dark2.npy",
        "power": 1.5, # in mW (direct laser power, legacy)
        "pixel_size_um": 6.5,
        "wavelength_nm": 488.0
    }}
}}
</code></pre>

<p>
<strong>Note:</strong> Use <code>port_power</code> (mW from sphere port measurement)
or <code>power</code> (direct laser power, legacy). The tool converts these to
camera irradiance using the sphere calibration models.
</p>

<h3>Analysis Output</h3>
<ul>
    <li>
        <strong>Gain (e⁻/DN):</strong> Conversion factor from digital
        numbers to electrons (from PTC: slope of variance vs. mean)
    </li>
    <li>
        <strong>Read Noise (e⁻):</strong> Camera's dark-limited noise
        floor (from PTC: intercept of variance vs. mean)
    </li>
    <li>
        <strong>Responsivity (ADU/photon):</strong> Signal response per
        incident photon (linear fit of mean vs. photon flux)
    </li>
    <li>
        <strong>Quantum Efficiency (%):</strong> Fraction of incident
        photons converted to electrons (responsivity × gain / wavelength)
    </li>
    <li>
        <strong>SNR (dB):</strong> Signal-to-noise ratio as a function of
        photon flux
    </li>
</ul>

<h3>Workflow</h3>
<ol>
    <li>
        Go to <strong>Sphere Power Calibration</strong> tab
        <ul>
            <li>Load calibration data file
            (CSV with laser_power, port, camera columns)</li>
            <li>Set photodiode diameter</li>
            <li>Run calibration to fit power→port and port→camera models</li>
            <li>Verify linear fits in plots (R² should be ≥0.9999)</li>
        </ul>
    </li>
    <li>
        Return to <strong>General</strong> tab
        <ul>
            <li>Add datasets manually or import from JSON</li>
            <li>Ensure datasets have port_power (mW) or laser power specified</li>
        </ul>
    </li>
    <li>Click <strong>Run Analysis</strong> to compute PTC metrics</li>
    <li>
        Review results in the <strong>Analysis Results</strong> tabs
        <ul>
            <li>Mean vs. Variance plot (gain and read noise from intercept/slope)</li>
            <li>Photon Flux vs. Mean Signal (responsivity)</li>
            <li>Quantum Efficiency per dataset</li>
            <li>SNR vs. Photon Flux</li>
        </ul>
    </li>
</ol>

<h3>Notes</h3>
<ul>
    <li>
        Analysis cache (<code>.npz</code>) stores plot-ready results and can be
        reloaded without recomputing.
    </li>
    <li>
        Gain-fit variance source (<code>shot</code> or <code>total</code>) is
        persisted in cache and restored on load.
    </li>
    <li>
        Two analysis cache files can be compared directly from the UI.
    </li>
    <li>
        Export <strong>Dark Cal JSON</strong> writes per-dataset
        <code>dark_calibration_directory</code> (blank by default),
        <code>gain</code>, and available <code>responsivity</code>/<code>qe</code>
        from cached analysis results.
    </li>
    <li>PTC analysis uses exposure times: 0.001 to 0.200 s (1 ms increments)</li>
    <li>Photon flux calculated from:</li>
    <ul>
        <li>Calibrated irradiance at camera plane (from sphere calibration)</li>
        <li>Laser wavelength and pixel size</li>
        <li>Exposure time</li>
    </ul>
    <li>Pixel-level analysis is spatially averaged
    (over height/width) for robustness</li>
    <li>
        Linearity limit defaults to 70% of saturation capacity (typically
        4096 DN) to ensure valid PTC region
    </li>
    <li>
        Temporal variance method (difference between paired frames) reduces
        fixed-pattern noise effects
    </li>
    <li>
        Sphere calibration finds optimal linear regions automatically with
        data cleaning and outlier rejection
    </li>
</ul>
'''

RESULTS_CACHE_SCHEMA_VERSION = 1


class PlotType(Enum):
    GAIN = 'gain'
    RESPONSE = 'response'
    QE = 'qe'
    QE_BAR = 'qe_bar'
    SNR = 'snr'


PLOTS_META = {
    PlotType.GAIN: {
        'title': 'Variance [ADU²] vs Mean Signal [ADU]',
        'xlabel': 'Mean Signal [ADU]',
        'ylabel': 'Variance [ADU²]',
        'legend': True,
    },
    PlotType.RESPONSE: {
        'title': 'Mean Signal [ADU] vs Photon Flux [photons/cm²/s]',
        'xlabel': 'Photon Flux [photons/cm²/s]',
        'ylabel': 'Mean Signal [ADU]',
        'legend': True,
    },
    PlotType.QE: {
        'title': 'Mean Signal [e-] vs Photon Flux [photons/cm²/s]',
        'xlabel': 'Photon Flux [photons/cm²/s]',
        'ylabel': 'Mean Signal [e-]',
        'legend': True,
    },
    PlotType.QE_BAR: {
        'title': 'Quantum Efficiency',
        'xlabel': 'Dataset',
        'ylabel': 'QE [%]',
        'legend': False,
        'grid': True,
    },
    PlotType.SNR: {
        'title': 'SNR vs Photon Flux',
        'xlabel': 'Photon Flux [photons/cm²/s]',
        'ylabel': 'SNR (dB)',
        'legend': True,
    },
}
