# microEye
A python toolkit for fluorescence microscopy that features IDS uEye industrial-grade CMOS cameras.

The *Acquisition Module* allows multi-cam image acquisition within one graphical user interface.

The *Control Module* allows setting the laser excitation presets, manual focus and automatic focus stabilization by monitoring the peak position of a totally internally reflected IR beam and moving the piezo stage accordingly. 

This toolkit is compatible with the [hardware](#hardware) we are using in our microscope. For further details check our microscope's github (TBA)

## Uses Packages
 
- Numpy 
- scipy
- pyueye
- cv2
- tifffile
- PyQt5
- pyqtgraph
- qdarkstyle

## How to Install [Package](https://pypi.org/project/microEye/)

    > pip install microEye

## Microscope Scheme

![scheme](https://user-images.githubusercontent.com/89871015/135764774-8c2dbc12-bff1-4325-97bc-7f1fc356f517.png)

## Hardware 

- IDS uEye industrial-grade CMOS cameras, specifically [UI-3060CP Rev. 2](https://en.ids-imaging.com/store/products/cameras/ui-3060cp-rev-2.html) 
- Integrated Optics Multi-wavelength Laser Combiner [MatchBox](https://integratedoptics.com/products/wavelength-combiners)
- Piezo Concept nanopositioner for microscope objectives [FOC](https://piezoconcept-store.squarespace.com/1-axis/p/foc)
- Parallax Linescan Camera Module used for IR autofocus stabilization tracking [TSL1401-DB (#28317)](https://eu.mouser.com/ProductDetail/Parallax/28317?qs=%2Fha2pyFaduiCRhuOAXMuCmQIeG1Q3R01m6Y1EH%252BmN80%3D) acquisition done by an Arduino [LineScanner](https://github.com/samhitech/LineScanner)
- [RelayBox](https://github.com/samhitech/RelayBox) arduino for laser control using the camera flash signal with different presets.
- Parts list related to our iteration of [hohlbeinlab miCube](https://hohlbeinlab.github.io/miCube/index.html) (TBA)

## Acquisition Module

![acquisition_module](https://user-images.githubusercontent.com/89871015/135764990-b9ac0062-4710-4a10-b2f8-a16f34d77ee1.png)

### How to use

    from microEye.acquisition_module import acquisition_module
    
    try:
        app, window = acquisition_module.StartGUI()
        app.exec_()
    except Exception as e:
        traceback.print_exc()
    finally:
        # dispose camera adapters
        for cam in window.ids_cams:
            cam.dispose()

        # Destroys the OpenCv windows
        cv2.destroyAllWindows()

## Control Module

![control_module](https://user-images.githubusercontent.com/89871015/135764911-7a83217a-d1ab-480d-bb57-c82f9d4a6624.png)

### How to use

    from microEye.control_module import control_module
    
    app, window = control_module.StartGUI()
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
