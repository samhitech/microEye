# The microEye

The **`microEye`** is a Python toolkit for fluorescence microscopy that supports super-resolution single-molecule localization microscopy and single-particle tracking. It features hardware control, data analysis, and visualization.

This toolkit is compatible with the [hardware](#hardware) used in our microscope. For further details, refer to the [miEye microscope paper](https://doi.org/10.1016/j.ohx.2022.e00368) and [OSF project](http://doi.org/10.17605/osf.io/j2fqy).

![Package Health](https://snyk.io/advisor/python/microEye/badge.svg)
![Python Version](https://img.shields.io/badge/Python-version_3.9-blue)
![Package Version](https://img.shields.io/badge/version-2.0.0-gold)
![Package Version](https://img.shields.io/badge/GUI_Platform-PyQt5-navy)
![Package Version](https://img.shields.io/badge/OS-Windows-cyan)
[![DOI](https://img.shields.io/badge/HardwareX-10.1016/j.ohx.2022.e00368-orange)](https://doi.org/10.1016/j.ohx.2022.e00368)

```bash
   __  ____              ____                ___   ___   ___ 
  /  |/  (_)__________  / __/_ _____   _  __|_  | / _ \ / _ \
 / /|_/ / / __/ __/ _ \/ _// // / -_) | |/ / __/_/ // // // /
/_/  /_/_/\__/_/  \___/___/\_, /\__/  |___/____(_)___(_)___/ 
                          /___/
```

## Modules

### The miEye Module

The `miEye_module` provides the primary graphical user interface (GUI) for microscope control and data acquisition, combining the functionalities of the deprecated *Acquisition* and *Control* modules.

| miEye module | Acquisition Camera |
|----------------|----------------|
| ![miEye](https://github.com/samhitech/microEye/assets/89871015/20c5573a-e489-478e-adfc-29410bc6d4c2) | ![CamStack](https://github.com/samhitech/microEye/assets/89871015/ead95989-54ce-4643-b5a3-4461c36f6b14) |

**How to use:**

For Vimba SDK to work, the script should be executed as an administrator on Windows and wrapped in a `with` statement:

```python
from microEye.hardware import miEye_module

try:
    import vimba as vb
except Exception:
    vb = None

if vb:
    with vb.Vimba.get_instance() as vimba:
        app, window = miEye_module.StartGUI()
        app.exec_()
else:
    app, window = miEye_module.StartGUI()
    app.exec_()
```

### The Multi Viewer Module

The `multi_viewer` Module is an improved GUI that replaces the deprecated `tiff_viewer` module. It allows users to process multiple files and provides data analysis and visualization tools for super-resolution single-molecule localization microscopy and single-particle tracking.

| Raw Data | Localizations |
|----------------|----------------|
| ![imagesStack](https://github.com/samhitech/microEye/assets/89871015/17421117-a633-4a20-ba38-e7adffcd7332) | ![locStack](https://github.com/samhitech/microEye/assets/89871015/713d9fc7-7f92-4341-adb2-17b83ac8fc34) |

**How to use:**

```python
from microEye import multi_viewer

app, window = multi_viewer.StartGUI('')

app.exec_()
```

## Uses Packages

The `microEye` uses the following Python packages:

| Data Analysis and Visualization | GUI and UI Development | Code Quality and Formatting | Image and Video Processing | File and Data Storage | Other Utilities |
|--------------------------------|------------------------|------------------------------|---------------------------|-----------------------|-----------------|
| dask                           | PyQt5                  | autopep8                     | opencv-python | ome-types             | hidapi          |
| h5py                           | pyqtdarktheme          | pyflakes                    |                           | tables                | pyfiglet        |
| matplotlib                     | pyqtgraph              |                              |              | zarr                  | pyserial        |
| numba                          | QDarkStyle             |                              |                           |                       | pyueye          |
| numpy                          | QScintilla             |                              |                           |                       | setuptools      |
| pandas                         |                        |                              |                           |                       | tabulate        |
| scikit-image                   |                        |                              |                           |                       | VimbaPython     |
| scikit-learn                   |                        |                              |                           |                       |                 |
| scipy                          |                        |                              |                           |                       |                 |
| tifffile                       |                        |                              |                           |                       |                 |
| vispy                          |                        |                              |                           |                       |                 |

Note: VimbaPython is included in Vimba SDK and needs to be installed manually.

## How to Install [microEye](https://pypi.org/project/microEye/)

1. **Install Python:**

   Download and install the latest [Python 3.9 stable release](https://www.python.org/downloads/).

2. **Install microEye package:**

   Open a terminal and execute the following command to install microEye using pip:

    ```powershell
    pip install microEye
    ```

3. **Install required packages:**

   Download the [requirements.txt](https://github.com/samhitech/microEye/blob/main/requirements.txt) file. Navigate to the directory containing the requirements file in your terminal and run:

    ```powershell
    pip install -r requirements.txt
    ```

   Note: This step might take a while.

4. **Install specific hardware drivers: (Optional)**
   - For Integrated Optics: Download and install [Laser control software](https://integratedoptics.com/downloads).
   - For IDS uEye CMOS cameras: Install [IDS Software Suite 4.96.1](https://en.ids-imaging.com/download-details/AB00604.html?os=windows&version=win10&bus=64&floatcalc=) for Windows 32/64-bit.
   - For Allied Vision CMOS cameras: Install [Vimba SDK](https://www.alliedvision.com/en/products/vimba-sdk) 5.0 or 6.0 outside the Program Files. Navigate to the directory containing setup.py and run:

        ```powershell
        python -m pip install .
        ```

   - For Thorlabs CMOS cameras: Install [Thorcam](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam) in its default directory. Note: Some Thorlabs cameras may be identified as IDS uEye cameras by Windows and may run without Thorcam.

   - For Thorlabs hardware, install [Kinesis® Software](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285) and [Elliptec™ Software](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ELL).

5. **Download and run examples:**
   Download examples to start using microEye!

:partying_face: **Note:** Ensure all necessary drivers are installed for microEye to function properly.

## Microscope Scheme

Schematic overview of the miEye instrument:

- **A) Single-Mode Fiber (SMF):** Excitation path for TIRF-, HILO-, and Epi-mode.
- **B) Multi-Mode Fiber (MMF):** Excitation path for Epi-mode when imaging MMF output on the sample plane.
- **C) Fluorescence Emission:** Path for fluorescence emission.
- **D) Automatic Focus Stabilization:** the automatic focus stabilization path using an IR laser in TIRF setting.

| Scheme 1 | Scheme 2 |
|----------|---------|
| ![Quad Scheme](https://user-images.githubusercontent.com/89871015/182302644-9fdf8615-75c3-4702-9913-d1a535f60e22.png) | ![Scheme GIT](https://user-images.githubusercontent.com/89871015/182302694-3f70d058-b1b6-4ef5-9cc2-aec9b58a05f0.png) |

**Key Components:** *AC:* Achromat lens, *AS:* Aspheric lens, *BFP:* Back-focal plane, *TL:* Tube lens, *B:* B-coated N-BK7 optics, *BS:* Beamsplitter.

## Hardware

### Supported Cameras

| Camera | Description | Link |
|--------|-------------|------|
| IDS uEye UI-3060CP Rev. 2 | IDS industrial-grade CMOS cameras | [Link](https://en.ids-imaging.com/store/products/cameras/ui-3060cp-rev-2.html) |
| Thorlabs DCC1545M | DCx camera using UC480 driver | [Link](https://www.thorlabs.com/thorProduct.cfm?partNumber=DCC1545M) |
| Allied Vision Alvium 1800 | Allied Vision industrial-grade CMOS cameras (U-158m, U-511m)  | [Link](https://www.alliedvision.com/en/products/alvium-configurator/alvium-1800-u/158/#_configurator) |

### Additional Hardware

| Hardware | Description | Link |
|----------|-------------|------|
| Integrated Optics MatchBox | Multi-wavelength Laser Combiner, Single Laser MatchBox | [Link](https://integratedoptics.com/products/wavelength-combiners) |
| Piezo Concept FOC | Nanopositioner for microscope objectives | [Link](https://piezoconcept-store.squarespace.com/1-axis/p/foc) |
| Thorlabs Elliptec ELL6/ELL9 | Dual/Four-Position Support | [ELL6](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL6), [ELL9](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL9) |
| Thorlabs KDC101 | Kinesis Controller for Z825B/[Z925B](https://www.thorlabs.com/thorproduct.cfm?partnumber=Z925B) actuators (Activate USB VCP to access the COM port in device manager) | [Link](https://www.thorlabs.com/thorproduct.cfm?partnumber=KDC101) |
| Parallax TSL1401-DB (#28317) | Linescan Camera Module | [Link](https://eu.mouser.com/ProductDetail/Parallax/28317?qs=%2Fha2pyFaduiCRhuOAXMuCmQIeG1Q3R01m6Y1EH%252BmN80%3D) |
| RelayBox Arduino | For laser control using camera GPIO signals | [RelayBox](https://github.com/samhitech/RelayBox) |
| miEye OSF Project Parts List | Parts list of miEye OSF Project | [Link](https://osf.io/j2fqy/) |

## Authors

[Mohammad Nour Alsamsam](https://tutkuslab.github.io/team/MNA/), PhD student @ Vilnius University.

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/samhightech.svg?style=social&label=Follow%20%40samhightech)](https://twitter.com/samhightech)

## People Involved

**PhD supervision:** [Dr. Marijonas Tutkus](https://tutkuslab.github.io/team/MT/)

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/MTutkus.svg?style=social&label=Follow%20%40MTutkus)](https://twitter.com/MTutkus)

**Sample preparation, experiments and testing:**

- [Aurimas Kopūstas](https://tutkuslab.github.io/team/AK/)
- [Tutkus Lab Team](https://tutkuslab.github.io/team/)

## Acknowledgement

Research and access to intruments and samples is credited to:

- Vilnius University, Lithuania.
- Center For Physical Sciences and Technology, Vilnius, Lithuania.
- Research Council of Lithuania.
