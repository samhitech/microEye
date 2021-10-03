# microEye
A python toolkit for fluorescence microscopy that features IDS uEye industrial-grade CMOS cameras.

## Uses Packages
 
- Numpy 
- scipy
- pyueye
- cv2
- tifffile
- PyQt5
- pyqtgraph
- qdarkstyle

## Hardware 

- IDS uEye industrial-grade CMOS cameras, specifically [UI-3060CP Rev. 2](https://en.ids-imaging.com/store/products/cameras/ui-3060cp-rev-2.html) 
- Integrated Optics Multi-wavelength Laser Combiner [MatchBox](https://integratedoptics.com/products/wavelength-combiners)
- Piezo Concept nanopositioner for microscope objectives [FOC](https://piezoconcept-store.squarespace.com/1-axis/p/foc)
- Parallax Linescan Camera Module used for IR autofocus stabilization tracking [TSL1401-DB (#28317)](https://eu.mouser.com/ProductDetail/Parallax/28317?qs=%2Fha2pyFaduiCRhuOAXMuCmQIeG1Q3R01m6Y1EH%252BmN80%3D) acquisition done by an arduino (code TBA)
- RelayBox arduino for laser control using the camera flash signal with different presets (code TBA)
- Parts list related to our iteration of [hohlbeinlab miCube](https://hohlbeinlab.github.io/miCube/index.html) (TBA)

## Microscope Scheme

![scheme](https://user-images.githubusercontent.com/89871015/135764774-8c2dbc12-bff1-4325-97bc-7f1fc356f517.png)

## Control Module

![control_module](https://user-images.githubusercontent.com/89871015/135764911-7a83217a-d1ab-480d-bb57-c82f9d4a6624.png)

### How to use

    app, window = control_module.StartGUI()
    app.exec_()

## Acquisition Module

![acquisition_module](https://user-images.githubusercontent.com/89871015/135764990-b9ac0062-4710-4a10-b2f8-a16f34d77ee1.png)

### How to use

    app, window = acquisition_module.StartGUI()
    app.exec_()
