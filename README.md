# The microEye

![microEye](https://github.com/samhitech/microEye/assets/89871015/37ca0859-0b4b-402d-8652-9a01d0bf0f29)

The **`microEye`** is a Python toolkit for fluorescence microscopy that supports super-resolution single-molecule localization microscopy and single-particle tracking. It features hardware control, data analysis, and visualization.

This toolkit is compatible with the [hardware](#hardware) used in our microscope. For further details, refer to the [miEye microscope paper](https://doi.org/10.1016/j.ohx.2022.e00368) and [OSF project](http://doi.org/10.17605/osf.io/j2fqy).

![Package Health](https://snyk.io/advisor/python/microEye/badge.svg)
![Python Version](https://img.shields.io/badge/Python-version_>=3.9-blue)
![Package Version](https://img.shields.io/badge/version-2.1.1-gold)
![Package Version](https://img.shields.io/badge/GUI_Platform-PySide6|PyQt6|PyQt5-navy)
[![DOI](https://img.shields.io/badge/HardwareX-10.1016/j.ohx.2022.e00368-orange)](https://doi.org/10.1016/j.ohx.2022.e00368)
![Package Version](https://img.shields.io/badge/Windows-Tested-cyan)
![Package Version](https://img.shields.io/badge/MacOS-Errors-red)
![Package Version](https://img.shields.io/badge/Linux-NotTested-lightgray)

```bash
   __  ____              ____                ___   ___ ___
  /  |/  (_)__________  / __/_ _____   _  __|_  | <  // _ \
 / /|_/ / / __/ __/ _ \/ _// // / -_) | |/ / __/_ / // // /
/_/  /_/_/\__/_/  \___/___/\_, /\__/  |___/____(_)_(_)___/
                          /___/
```

## Table of Contents

- [The microEye](#the-microeye)
  - [Table of Contents](#table-of-contents)
  - [How to Install microEye](#how-to-install-microeye)
    - [Troubleshooting Installation](#troubleshooting-installation)
  - [microEye Launcher](#microeye-launcher)
    - [Usage](#usage)
  - [Modules](#modules)
    - [The miEye Module](#the-mieye-module)
      - [Experiment Designer (Beta)](#experiment-designer-beta)
    - [The Multi Viewer Module](#the-multi-viewer-module)
  - [Uses Packages](#uses-packages)
  - [Microscope Scheme](#microscope-scheme)
  - [Hardware](#hardware)
    - [Supported Cameras](#supported-cameras)
    - [Additional Hardware](#additional-hardware)
  - [Authors](#authors)
  - [People Involved](#people-involved)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## How to Install [microEye](https://pypi.org/project/microEye/)

1. **Install Python:**

   Download and install the latest [Python](https://www.python.org/downloads/)  ≥3.9 stable release.

2. **Install microEye package:**

   Open a terminal and execute the following command to install microEye using pip:

    ```powershell
    pip install microEye --upgrade
    ```

   Or for a specific version:

    ```powershell
    pip install microEye==version
    ```

3. **Install required packages: (Optional)**

   Download the [requirements.txt](https://github.com/samhitech/microEye/blob/main/requirements.txt) file. Navigate to the directory containing the requirements file in your terminal and run:

    ```powershell
    pip install -r requirements.txt
    ```

   Note: This step is optional as dependecies are installed with the package.

4. **Install specific hardware drivers: (Optional)**
   - For Integrated Optics: Download and install [Laser control software](https://integratedoptics.com/downloads).
   - For IDS uEye CMOS cameras: Install [IDS Software Suite 4.96.1](https://en.ids-imaging.com/download-details/AB00604.html?os=windows&version=win10&bus=64&floatcalc=) for Windows 32/64-bit.
   - For Allied Vision CMOS cameras: Install [Vimba SDK](https://www.alliedvision.com/en/products/vimba-sdk) 5.0 or 6.0 outside the Program Files. Navigate to the directory containing setup.py and run:

        ```powershell
        python -m pip install .
        ```

   - For Thorlabs CMOS cameras: Install [Thorcam](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam) in its default directory. Note: Some Thorlabs cameras may be identified as IDS uEye cameras by Windows and may run without Thorcam.

   - For Thorlabs hardware, install [Kinesis® Software](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285) and [Elliptec™ Software](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ELL).

5. **Open a terminal and execute microEye:** :partying_face:

   ```powershell
   usage: microEye.exe [-h] [--module MODULE] [--QT_API QT_API] [--theme THEME]

   optional arguments:
    -h, --help            show this help message and exit
    --module {mieye,viewer}
                          The module to launch [mieye|viewer], If not specified, launcher is executed.
    --QT_API {PySide6,PyQT6,PyQt5}
                          Select QT API [PySide6|PyQT6|PyQt5], If not specified, the environment variable QT_API is used.
    --theme {None,qdarktheme,qdarkstyle,...}
                          The theme of the app, if not specified, the environment variable MITHEME is used.
   ```

> **Note:** Ensure all necessary drivers are installed for microEye to function properly.

### Troubleshooting Installation

If you encounter any issues during the installation process, please check the following:

- Ensure that you have the latest version of Python installed (≥3.9).
- Try installing the required packages repeating step (2) or individually using `pip install <package_name>`.
- Check the system requirements for specific hardware drivers and follow the installation instructions provided by the manufacturers.
- Refer to the project's issue tracker on GitHub for any known installation issues and solutions.

If the issue persists, feel free to open a new issue on the project's GitHub repository, providing detailed information about the problem and any error messages you encountered.

## microEye Launcher

The microEye Launcher allows you to launch either the `miEye Module` or the `Viewer Module`, and provides options to select the Qt API and theme for the application.

![Launcher Screenshot](https://github.com/samhitech/microEye/assets/89871015/385cc3d6-e3b8-44a4-a288-2471b6b34f7f)

### Usage

Upon running the launcher, you will be presented with the following interface:

- **miEye Module**: Launches the miEye module for microscope control and acquisition.
- **Viewer Module**: Launches the viewer module for image/data anlysis and visualization.
- **QT API (dropdown)**: Select the Qt API to use. Options are PySide6, PyQt6, or PyQt5.
- **Theme (dropdown)**: Select the theme for the application. Options are None (default), qdarktheme, or qdarkstyle.

To launch a module, simply click on the respective button (`miEye Module` or `Viewer Module`). If you wish to change the Qt API or theme, select the desired option from the dropdown menus before launching.
Sounds good, here's the updated section with the simplified commands:

## Modules

### The miEye Module

The `miEye_module` provides the primary graphical user interface (GUI) for microscope control and data acquisition, combining the functionalities of the deprecated *Acquisition* and *Control* modules.

| miEye module                                                                                         | Acquisition Camera                                                                                      |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ![miEye](https://github.com/samhitech/microEye/assets/89871015/20c5573a-e489-478e-adfc-29410bc6d4c2) | ![CamStack](https://github.com/samhitech/microEye/assets/89871015/ead95989-54ce-4643-b5a3-4461c36f6b14) |

**How to use:**

To launch the miEye module, run the following command:

```powershell
microEye --module mieye
```

#### Experiment Designer (Beta)

![Experiment Designer (Beta)](https://github.com/samhitech/microEye/assets/89871015/0693a620-6aed-4bf4-a723-7b0c7b143d1f)

The Experiment Designer is a new feature within the `miEye Module` that allows users to create and manage complex acquisition protocols through a graphical interface.

Key features include:

- GUI-based design of acquisition protocols;
- Support for loops and device parameter adjustments/actions;
- Export and import capabilities for protocols;
- Threaded execution with pause functionality;
- Configurable delays between protocol steps
- Context menu for easy adjustments
- Nested protocol structure
- Acquisition action wait option
- Cell reordering and deletion via keyboard shortcuts
- Real-time execution progress view

To access the Experiment Designer:

1. Launch the `miEye module`.
2. Look for the Experiment Designer (Protocols) view in the interface.

Note: This feature is currently in beta. We value your feedback to enhance its functionality and improve the user experience.

### The Multi Viewer Module

The `multi_viewer` Module is an improved GUI that replaces the deprecated `tiff_viewer` module. It allows users to process multiple files and provides data analysis and visualization tools for super-resolution single-molecule localization microscopy and single-particle tracking.

| Raw Data                                                                                                   | Localizations                                                                                           |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ![imagesStack](https://github.com/samhitech/microEye/assets/89871015/17421117-a633-4a20-ba38-e7adffcd7332) | ![locStack](https://github.com/samhitech/microEye/assets/89871015/713d9fc7-7f92-4341-adb2-17b83ac8fc34) |

**How to use:**

To launch the Multi Viewer module, run the following command:

```powershell
microEye --module viewer
```

## Uses Packages

The `microEye` uses the following Python packages:

| Data Analysis and Visualization | GUI and UI Development | Code Quality and Formatting | Image and Video Processing | File and Data Storage | Other Utilities |
| ------------------------------- | ---------------------- | --------------------------- | -------------------------- | --------------------- | --------------- |
| dask                            | PySide6                | autopep8                    | opencv-python              | ome-types             | hidapi          |
| matplotlib                      | pyqtdarktheme          | pyflakes                    |                            | tables                | pyfiglet        |
| numba                           | pyqtgraph              |                             |                            | zarr                  | pyserial        |
| numpy                           | QDarkStyle             |                             |                            |                       | pyjokes         |
| pandas                          | QScintilla             |                             |                            |                       | setuptools      |
| scikit-image                    | PyQt5 (optional)       |                             |                            |                       | tabulate        |
| scikit_learn                    | PyQt6 (optional)       |                             |                            |                       | pyueye          |
| scipy                           |                        |                             |                            |                       |                 |
| tifffile                        |                        |                             |                            |                       |                 |
| vispy                           |                        |                             |                            |                       |                 |

Note: VimbaPython is included in Vimba SDK and needs to be installed manually.

## Microscope Scheme

Schematic overview of the miEye instrument:

- **A) Single-Mode Fiber (SMF):** Excitation path for TIRF-, HILO-, and Epi-mode.
- **B) Multi-Mode Fiber (MMF):** Excitation path for Epi-mode when imaging MMF output on the sample plane.
- **C) Fluorescence Emission:** Path for fluorescence emission.
- **D) Automatic Focus Stabilization:** the automatic focus stabilization path using an IR laser in TIRF setting.

| Scheme 1                                                                                                              | Scheme 2                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| ![Quad Scheme](https://user-images.githubusercontent.com/89871015/182302644-9fdf8615-75c3-4702-9913-d1a535f60e22.png) | ![Scheme GIT](https://user-images.githubusercontent.com/89871015/182302694-3f70d058-b1b6-4ef5-9cc2-aec9b58a05f0.png) |

**Key Components:** *AC:* Achromat lens, *AS:* Aspheric lens, *BFP:* Back-focal plane, *TL:* Tube lens, *B:* B-coated N-BK7 optics, *BS:* Beamsplitter.

## Hardware

### Supported Cameras

| Camera                    | Description                                                  | Link                                                                                                  |
| ------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| IDS uEye UI-3060CP Rev. 2 | IDS industrial-grade CMOS cameras                            | [Link](https://en.ids-imaging.com/store/products/cameras/ui-3060cp-rev-2.html)                        |
| Thorlabs DCC1545M         | DCx camera using UC480 driver                                | [Link](https://www.thorlabs.com/thorProduct.cfm?partNumber=DCC1545M)                                  |
| Allied Vision Alvium 1800 | Allied Vision industrial-grade CMOS cameras (U-158m, U-511m) | [Link](https://www.alliedvision.com/en/products/alvium-configurator/alvium-1800-u/158/#_configurator) |

### Additional Hardware

| Hardware                     | Description                                                                                                                                                           | Link                                                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Integrated Optics MatchBox   | Multi-wavelength Laser Combiner, Single Laser MatchBox                                                                                                                | [Link](https://integratedoptics.com/products/wavelength-combiners)                                                                 |
| Piezo Concept FOC            | Nanopositioner for microscope objectives                                                                                                                              | [Link](https://piezoconcept-store.squarespace.com/1-axis/p/foc)                                                                    |
| Thorlabs Elliptec ELL6/ELL9  | Dual/Four-Position Support                                                                                                                                            | [ELL6](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL6), [ELL9](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL9) |
| Thorlabs KDC101              | Kinesis Controller for Z825B/[Z925B](https://www.thorlabs.com/thorproduct.cfm?partnumber=Z925B) actuators (Activate USB VCP to access the COM port in device manager) | [Link](https://www.thorlabs.com/thorproduct.cfm?partnumber=KDC101)                                                                 |
| Parallax TSL1401-DB (#28317) | Linescan Camera Module                                                                                                                                                | [Link](https://eu.mouser.com/ProductDetail/Parallax/28317?qs=%2Fha2pyFaduiCRhuOAXMuCmQIeG1Q3R01m6Y1EH%252BmN80%3D)                 |
| RelayBox Arduino             | For laser control using camera GPIO signals                                                                                                                           | [RelayBox](https://github.com/samhitech/RelayBox)                                                                                  |
| miEye OSF Project Parts List | Parts list of miEye OSF Project                                                                                                                                       | [Link](https://osf.io/j2fqy/)                                                                                                      |

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

Special thanks to the following projects and libraries that make this work possible:

- **SMAP/fit3Dcspline**: The original code, provided as a part of the Ries group [SMAP software](https://github.com/jries/SMAP/tree/master/fit3Dcspline). The `pyfit3Dcspline` module in our project is a Python adaptation of this functionality, offering both CPU and GPU accelerated fitting of Single-Molecule Localization Microscopy (SMLM) data. For more details, refer to the pyfit3Dcspline [README](https://github.com/samhitech/microEye/tree/main/src/microEye/analysis/fitting/pyfit3Dcspline/README.md).

- **ACCéNT**: a partial implementation of the photon free calibration within the acquisition pipeline which generates pixel-wise mean and variance images.
  > Robin Diekmann, Joran Deschamps, Yiming Li, Aline Tschanz, Maurice Kahnwald, Ulf Matti, Jonas Ries, "Photon-free (s)CMOS camera characterization for artifact reduction in high- and super-resolution microscopy", bioRxiv 2021.04.16.440125. [doi: 2021.04.16.440125](https://doi.org/10.1101/2021.04.16.440125)

- **Phasor Fit**: We have implemented the 2D phasor fitting algorithm in Python for fast pre-fitting visualization of localizations.
  > K.J.A. Martens, A.N. Bader, S. Baas, B. Rieger, J. Hohlbein. "Phasor based single-molecule localization microscopy in 3D (pSMLM-3D): an algorithm for MHz localization rates using standard CPUs," bioRxiv, 2017. [DOI: 10.1101/191957](https://doi.org/10.1101/191957).

- **Endesfelder Lab/SMLMComputational**: A numba accelerated adaptation of Drift Correction, Fourier Ring Correlation (FRC) structural resolution and Nearest Neighbour Analysis (NeNA) for localization precision from [Endesfelder Lab](https://github.com/Endesfelder-Lab/SMLMComputational).
  > Raw data to results: a hands-on introduction and overview of computational analysis for single-molecule localization microscopy", Martens et al., (2022), Frontiers in Bioinformatics. [Paper](https://www.frontiersin.org/articles/10.3389/fbinf.2021.817254)

- **TARDIS (Temporal Analysis of Relative Distances)**: We have developed a partial Python implementation of TARDIS without fitting for now. For more information, refer to the [TARDIS software releases](https://github.com/kjamartens/TARDIS-public). The underlying algorithms and scientific details of TARDIS are detailed in the manuscript:
  > Martens et al., “Temporal analysis of relative distances (TARDIS) is a robust, parameter-free alternative to single-particle tracking”, Nature Methods (2024). [Link](https://rdcu.be/dv1sr)

**Note**: I'm committed to maintaining an accurate acknowledgment list for our project. However, if I inadvertently miss acknowledging your work, please don't hesitate to reach out to us. I appreciate your understanding, and I'm doing my best to manage all acknowledgments amidst my other responsibilities.
> I make it a standard practice to cite and provide credit within a function's docstring whenever I draw inspiration from any external reference.

## Citation

If you find our work or software helpful in your research or project, we kindly request that you cite it appropriately. Here is the suggested citation format:

> M.N. Alsamsam, A. Kopūstas, M. Jurevičiūtė, and M. Tutkus, “The miEye: Bench-top super-resolution microscope with cost-effective equipment,” HardwareX 12, e00368 (2022). [Paper](https://doi.org/10.1016/j.ohx.2022.e00368)

Additionally, we would appreciate it if you could provide a link to our GitHub repository or any relevant publication associated with the software.
> Alsamsam, M. N. microEye, <https://github.com/samhitech/microEye> [Computer software]

Thank you for your support!
