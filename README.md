# microEye
A python toolkit for fluorescence microscopy that features IDS uEye industrial-grade CMOS cameras.

The *Acquisition Module* allows multi-cam image acquisition within one graphical user interface.

The *Control Module* allows setting the laser excitation presets, manual focus and automatic focus stabilization by monitoring the peak position of a totally internally reflected IR beam and moving the piezo stage accordingly. 

The *Tiff Viewer Module* allows accessing tiff images (single file and sequecnces) of order TYX (2D SMLM), also allows for visualizing the filtering-localization process *WYSIWYG*.  

This toolkit is compatible with the [hardware](#hardware) we are using in our microscope. For further details check our *miEye* microscope's [paper](https://doi.org/10.1016/j.ohx.2022.e00368) and [OSF project](http://doi.org/10.17605/osf.io/j2fqy).

## Uses Packages
 
- Numpy
- scipy
- pandas
- Zarr
- dask
- pyueye
- cv2
- tifffile
- PyQt5
- pyqtgraph
- qdarkstyle
- ome-types
- hidapi
- pyqode.python ([Jedi Fix](https://github.com/pyQode/pyqode.python/blob/6cc30087dab69d334a48c716d8d19fc1546ff0c6/pyqode/python/backend/workers.py))
- VimbaPython (Included in [Vimba SDK](https://www.alliedvision.com/en/products/vimba-sdk/) and installed manually)

## How to Install [Package](https://pypi.org/project/microEye/)

1. Download and install the latest [*Python*](https://www.python.org/downloads/) 3.9 stable release. (We tested with 3.9.4 and 3.9.7)

2. Open a terminal and install *microEye* package using *pip*.

    > pip install microEye

3. Download the file [*requirements.txt*](https://github.com/samhitech/microEye/blob/main/requirements.txt) and install the *microEye* required packages by executing the following line. (This may take a while) 

    > pip install -r /path/to/requirements.txt

4. Install the dependent hardware specific drivers for the cameras if you intend to use the hardware control or acquisition modules.
(*microEye* might not function as expected in case drivers are missing)
      * *Integrated Optics* [Laser control software](https://integratedoptics.com/downloads).
 
      * *IDS uEye* CMOS cameras: install [*IDS Software Suite 4.96.1*](https://en.ids-imaging.com/download-details/AB00604.html?os=windows&version=win10&bus=64&floatcalc=) for *Windows 32/64-bit*. (We did not test it with Linux)

      * *Allied Vision* CMOS cameras: install the [*Vimba SDK*](https://www.alliedvision.com/en/products/vimba-sdk) 5.0 or 6.0 outside the *Program Files* to skip the run as admin requirement, then in a terminal navigate to  directory [*.../Allied Vision/Vimba_5.0/VimbaPython/*] where [*setup.py*] is located and execute:

    > python -m pip install .

      * *Thorlabs* CMOS cameras: install [Thorcam](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam) in its default directory as it is initial to have *uc480_64.dll*  in path 'C:\Program Files\Thorlabs\Scientific Imaging\ThorCam\uc480_64.dll'. (Note, *Thorlabs* cameras can get identified by *Windows* as *IDS uEye* cameras and run without this software)  

5. Other hardware used by the *miEye* microscope include:
      * *Thorlabs* [*Kinesis® Software*](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285) and [*Elliptec™ Software*](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ELL).

6. Download and run examples to start using *microEye*! :partying_face:

## Microscope Scheme

Schematic overview of the miEye instrument. A) The excitation path via single-mode fiber (SMF) for TIRF-, HILO-, and Epi-mode. B) The excitation path via multi-mode fiber (MMF) for Epi-mode when imaging MMF output on the sample plane. C) The fluorescence emission path. D) IR laser-based autofocusing path. AC: achromat lens, AS: aspheric lens, BFP: back-focal plane, TL: tube lens, B: B-coated N-BK7 optics, BS: beamsplitter.

![quadScheme_4](https://user-images.githubusercontent.com/89871015/182302644-9fdf8615-75c3-4702-9913-d1a535f60e22.png)

![scheme_git_0](https://user-images.githubusercontent.com/89871015/182302694-3f70d058-b1b6-4ef5-9cc2-aec9b58a05f0.png)

## Hardware 

- Supported Cameras:
  - IDS uEye industrial-grade CMOS cameras, specifically [UI-3060CP Rev. 2](https://en.ids-imaging.com/store/products/cameras/ui-3060cp-rev-2.html).
  - Thorlabs DCx cameras using the UC480 driver, specifically [DCC1545M](https://www.thorlabs.com/thorProduct.cfm?partNumber=DCC1545M).
  - Allied Vision cameras using Vimba SDK, specifically [Alvium 1800 U-158m](https://www.alliedvision.com/en/products/alvium-configurator/alvium-1800-u/158/#_configurator).
- Integrated Optics Multi-wavelength Laser Combiner [MatchBox](https://integratedoptics.com/products/wavelength-combiners).
- Piezo Concept nanopositioner for microscope objectives [FOC](https://piezoconcept-store.squarespace.com/1-axis/p/foc).
- Thorlab's Elliptec Dual-Position/Four-Position ([ELL6](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL6)/[ELL9](https://www.thorlabs.com/thorproduct.cfm?partnumber=ELL9)) support.
- Thorlab's Kinesis [KDC101](https://www.thorlabs.com/thorproduct.cfm?partnumber=KDC101) controller for [Z825B](https://www.thorlabs.com/thorproduct.cfm?partnumber=Z825B) 25mm motorized actuators used for the XY stage instead of manual micrometers. (Requires activating USB VCP to access the COM port from device manger)
- Parallax Linescan Camera Module used for IR autofocus stabilization tracking [TSL1401-DB (#28317)](https://eu.mouser.com/ProductDetail/Parallax/28317?qs=%2Fha2pyFaduiCRhuOAXMuCmQIeG1Q3R01m6Y1EH%252BmN80%3D) acquisition done by an Arduino [LineScanner](https://github.com/samhitech/LineScanner).
- [RelayBox](https://github.com/samhitech/RelayBox) arduino for laser control using the camera flash signal with different presets.
- Parts list of our [miEye OSF Project](https://osf.io/j2fqy/) an iteration of [hohlbeinlab miCube](https://hohlbeinlab.github.io/miCube/index.html).

## Acquisition Module

![acquisition_module](https://user-images.githubusercontent.com/89871015/135764990-b9ac0062-4710-4a10-b2f8-a16f34d77ee1.png)

### How to use

For Vimba SDK to work the script should be executed as administrator on Windows and wrapped in a with statement:

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

## Control Module

![control_module](https://user-images.githubusercontent.com/89871015/141841883-d37c4979-c8aa-4e1f-b1b9-84bdd819c828.png)

### How to use

    from microEye.hardware import control_module
    
    app, window = control_module.StartGUI()
    app.exec_()


## miEye Module

TBA

## Data Viewer / Processor

![Capture_viewer](https://user-images.githubusercontent.com/89871015/150964047-ba4521fa-1ffa-4f76-9e4e-6759fbdc3b8f.PNG) 
![Capture_viewer_2](https://user-images.githubusercontent.com/89871015/150964062-9560b228-5052-40cb-86fc-5617f760a1b1.PNG) 
![Capture_viewer_3](https://user-images.githubusercontent.com/89871015/151448086-0799926d-a4a4-420b-ad12-177145a90c78.png)


### How to use

    from microEye import tiff_viewer

    app, window = tiff_viewer.StartGUI('D:/')

    app.exec_()


## Authors

Mohammad Nour Alsamsam

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/samhightech.svg?style=social&label=Follow%20%40samhightech)](https://twitter.com/samhightech)
    
## People Involved

Dr. Marijonas Tutkus (supervision)

[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/MTutkus.svg?style=social&label=Follow%20%40MTutkus)](https://twitter.com/MTutkus)


Aurimas Kopūstas (sample preparation and experiments)

## Acknowledgement

![ack](https://user-images.githubusercontent.com/89871015/135897106-12656072-932e-45ea-abeb-54e86ba60eb0.png)
