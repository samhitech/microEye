# microEye

**microEye** is a python toolkit for fluorescence microscopy that features hardware control, data analysis and vizualization for super-resolution single-molecule localization microscopy and single-partical tracking.

## Modules

- **miEye Module:** Provides the primary GUI for microscope control and data acquisition, combining previous functionalities implemented within *Acquisition* and *Control* modules.

- **Multi Viewer Module:** An improved GUI replacing *Tiff Viewer* module that allows processing multiple files providing data analysis and visualization tools.

**Note: The following modules are deprecated, and support has been halted. Please consider the alternatives above.**

- **Acquisition Module:** Enables multi-camera image acquisition within a unified graphical user interface.

- **Control Module:** Allows users to set laser excitation presets, manually focus, and perform automatic focus stabilization by monitoring the peak position of a totally internally reflected IR beam and adjusting the piezo stage accordingly.

- **Tiff Viewer Module:** Provides access to TIFF images (single files and sequences) of order TYX (2D SMLM) and facilitates visualizing the filtering-localization process (*WYSIWYG*).

This toolkit is compatible with the [hardware](#hardware) used in our microscope. For further details, refer to the [miEye microscope paper](https://doi.org/10.1016/j.ohx.2022.e00368) and [OSF project](http://doi.org/10.17605/osf.io/j2fqy).

## Uses Packages

- autopep8
- dask
- h5py
- hidapi
- matplotlib
- numba
- numpy
- ome_types
- opencv_python
- pandas
- pyflakes
- PyQt5
- PyQt5_sip
- pyqtgraph
- pyserial
- pyueye
- QDarkStyle
- QScintilla
- scikit_image
- scikit_learn
- scipy
- setuptools
- tables
- tifffile
- VimbaPython (Included in [Vimba SDK](https://www.alliedvision.com/en/products/vimba-sdk/) and installed manually)
- zarr

## How to Install [Package](https://pypi.org/project/microEye/)

1. Download and install the latest [*Python*](https://www.python.org/downloads/) 3.9 stable release.

2. Open a terminal and install the [*microEye*](https://pypi.org/project/microEye/) package using *pip*.

    ```bash
    pip install microEye
    ```

3. Download the [requirements.txt](https://github.com/samhitech/microEye/blob/main/requirements.txt) file and install the required packages for *microEye* by executing: (This may take a while)

    ```bash
    pip install -r /path/to/requirements.txt
    ```

4. Install the specific hardware drivers for the cameras:

   - **Integrated Optics** [Laser control software](https://integratedoptics.com/downloads).

   - **IDS uEye CMOS cameras:** Install [IDS Software Suite 4.96.1](https://en.ids-imaging.com/download-details/AB00604.html?os=windows&version=win10&bus=64&floatcalc=) for Windows 32/64-bit.

   - **Allied Vision CMOS cameras:** Install [Vimba SDK](https://www.alliedvision.com/en/products/vimba-sdk) 5.0 or 6.0 outside the Program Files to skip the run-as-admin requirement. In a terminal navigate to directory [*.../Allied Vision/Vimba_5.0/VimbaPython/*] where [*setup.py*] is located and execute:

        ```bash
        python -m pip install .
        ```

   - **Thorlabs CMOS cameras:** Install [Thorcam](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam) in its default directory. Note, some *Thorlabs* cameras can get identified by *Windows* as *IDS uEye* cameras and run without this software.

5. Other hardware used by the miEye microscope:

   - Thorlabs [Kinesis® Software](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285) and [Elliptec™ Software](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ELL).

6. Download and run examples to start using microEye! :partying_face:

**Note:** *microEye* might not function as expected in case drivers are missing.

## Microscope Scheme

Schematic overview of the miEye instrument. A) The excitation path via single-mode fiber (SMF) for TIRF-, HILO-, and Epi-mode. B) The excitation path via multi-mode fiber (MMF) for Epi-mode when imaging MMF output on the sample plane. C) The fluorescence emission path. D) IR laser-based autofocusing path. AC: achromat lens, AS: aspheric lens, BFP: back-focal plane, TL: tube lens, B: B-coated N-BK7 optics, BS: beamsplitter.

| Scheme 1 | Scheme 2 |
|----------| ---------|
| ![quadScheme_4](https://user-images.githubusercontent.com/89871015/182302644-9fdf8615-75c3-4702-9913-d1a535f60e22.png) | ![scheme_git_0](https://user-images.githubusercontent.com/89871015/182302694-3f70d058-b1b6-4ef5-9cc2-aec9b58a05f0.png)|

## Hardware

### Supported Cameras

- IDS uEye industrial-grade CMOS cameras, specifically [UI-3060CP Rev. 2](https://en.ids-imaging.com/store/products/cameras/ui-3060cp-rev-2.html).

- Thorlabs DCx cameras using the UC480 driver, specifically [DCC1545M](https://www.thorlabs.com/thorProduct.cfm?partNumber=DCC1545M).

- Allied Vision cameras using Vimba SDK, specifically [Alvium 1800 U-158m](https://www.alliedvision.com/en/products/alvium-configurator/alvium-1800-u/158/#_configurator).

### Additional Hardware

- Integrated Optics Multi-wavelength Laser Combiner [MatchBox](https://integratedoptics.com/products/wavelength-combiners).

- Piezo Concept nanopositioner for microscope objectives [FOC](https://piezoconcept-store.squarespace.com/1-axis/p/foc).

- Thorlab's Elliptec Dual-Position/Four-Position ([ELL6](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL6)/[ELL9](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL9)) support.

- Thorlab's Kinesis [KDC101](https://www.thorlabs.com/thorproduct.cfm?partnumber=KDC101) controller for [Z825B](https://www.thorlabs.com/thorproduct.cfm?partnumber=Z825B) 25mm motorized actuators used for the XY stage instead of manual micrometers. (Requires activating USB VCP to access the COM port from device manager)

- Parallax Linescan Camera Module used for IR autofocus stabilization tracking [TSL1401-DB (#28317)](https://eu.mouser.com/ProductDetail/Parallax/28317?qs=%2Fha2pyFaduiCRhuOAXMuCmQIeG1Q3R01m6Y1EH%252BmN80%3D) acquisition done by an Arduino [LineScanner](https://github.com/samhitech/LineScanner).

- [RelayBox](https://github.com/samhitech/RelayBox) Arduino for laser control using the camera flash signal with different presets.

- Parts list of our [miEye OSF Project](https://osf.io/j2fqy/) – an iteration of [hohlbeinlab miCube](https://hohlbeinlab.github.io/miCube/index.html).

## miEye Module

| miEye module | Acquisition Camera |
|----------------|----------------|
| ![miEye](https://github.com/samhitech/microEye/assets/89871015/73211153-6811-4acc-816f-154b1a2c5a32) | ![Cam_acq](https://github.com/samhitech/microEye/assets/89871015/a114fde6-a847-4be5-ab10-95f82190c830) |

### How to use

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

## Multi Viewer Module

| Raw Data | Localizations |
|----------------|----------------|
| ![Stack_images](https://github.com/samhitech/microEye/assets/89871015/07479e66-77c6-4d2d-b47f-ade89239dc49) | ![Stack_loc](https://github.com/samhitech/microEye/assets/89871015/beb0981d-a6d9-46d8-ae1c-e0d267f1d10e) |

### How to use

```python
from microEye import multi_viewer

app, window = multi_viewer.StartGUI('')

app.exec_()
```

## Acquisition Module (Deprecated)

### How to use

For Vimba SDK to work, the script should be executed as an administrator on Windows and wrapped in a `with` statement:

```python
from microEye.hardware import acquisition_module

try:
    import vimba as vb
except Exception:
    vb = None

if vb:
    with vb.Vimba.get_instance() as vimba:
        app, window = acquisition_module.StartGUI()
        app.exec_()
else:
    app, window = acquisition_module.StartGUI()
    app.exec_()
```

## Control Module (Deprecated)

### How to use

```python
from microEye.hardware import control_module

app, window = control_module.StartGUI()
app.exec_()
```

## Tiff Viewer Module (Deprecated)

| File System | Data Fitting | Localizations Visualization & Analysis |
| --------------- | ----------------- | ----------------- |
| ![Capture Viewer](https://user-images.githubusercontent.com/89871015/150964047-ba4521fa-1ffa-4f76-9e4e-6759fbdc3b8f.PNG) | ![Capture Viewer 2](https://user-images.githubusercontent.com/89871015/150964062-9560b228-5052-40cb-86fc-5617f760a1b1.PNG) | ![Capture Viewer 3](https://user-images.githubusercontent.com/89871015/151448086-0799926d-a4a4-420b-ad12-177145a90c78.png) |

### How to use

```python
from microEye import tiff_viewer

app, window = tiff_viewer.StartGUI('')

app.exec_()
```

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
