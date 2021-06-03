# rat-2p-area-characterizations 
Code for characterizing neural responses in rat cortex to a battery of visual stimuli. Neural data acquired with 2-photon imaging and epifluorescent wide-field imaging. 

Written by Juliana Rhee (Cox Lab, Harvard University).

## Data sources
Data Acquisition
* Wide-field data acquired and analyzed with custom code available at [julianarhee/retinotopy-mapper](https://github.com/julianarhee/retinotopic_mapping). 
* Visual stimuli for 2p experiments were presented with [MWorks](https://mworks.github.io/) and custom Python code. Custom  protocols available at [coxlab/protocols](https://github.com/julianarhee/protocols/tree/2p-dev). 
* 2p neural data was acquired using [ScanImage](http://scanimage.vidriotechnologies.com/display/SIH/ScanImage+Home) (5). 
* Additional custom code for data acquisition and syncing between stimulus presentation, neural data, and pupil/face-tracking at [julianarhee/acquisition-tools](https://github.com/julianarhee/acquisition-tools).

Data Processing
* Custom code for the basic data processing pipeline can be found at [julianarhee/2p-pipeline](https://github.com/julianarhee/2p-pipeline).
* Automated cell detection and trace extraction tested with [CaImAn](https://github.com/flatironinstitute/CaImAn) and [suite2p](https://github.com/MouseLand/suite2p).
* Face data (pupil and whisker tracking) was extracted with [DeepLabCut] (https://github.com/DeepLabCut/DeepLabCut) (6) or [FaceMap] (https://github.com/MouseLand/facemap).

## Visual Stimuli
Visual stimuli include a cycling bar (retinotopy), tiled gratings (receptive field mapping), drifting gratings (direction tuning), and objects (shape selectivity, transformation tolerance). 

**Wide-field retinotopy**
Retinotopic preferences are estimated using a phase-encoding protocol (adapted from 1-4).
TODO: some examples

**2-photon retinotopy**
Sub-portions of the cortex mapped with wide-field methods are targeted for 2-photon (2p) imaging, and retintopic preferences of a given field-of-view (FOV) are estimated with the same phase-encoding protocol. This method allows for fine-scale characterizations of retinotopic organization and validation of visual area assignment of a given FOV.  
TODO: some examples

**Receptive field mapping**
Receptive field characteristics of single neurons are measured using a tiling protocol that presents a dynamic stimulus one small square or tile at a time across the whole screen. 
TODO: some examples

**Drifting gratings**
Drifting gratings that vary in direction of motion, size, spatial frequency, and speed measure single neuron preferences for low-level visual features (*e.g.*, direction-tuning). 
TODO: some examples

**Objects**
Object stimuli are complex shapes that vary along more than one dimension (unlike gratings, for example). Two axes of transformation are tested: identity-changing transformations and identity-preserving transformations. Stimuli are adapted from Zoccolan et al., 2009 (8). 
TODO: examples

## Getting Started
Create the environment (conda).
```
$ conda env create -f rat2p.yml 
$ source activate rat2p
```

## References
1. Kalatsky VA, Stryker MP (2003) New paradigm for optical imaging: temporally encoded maps of intrinsic signal. _Neuron_ 38:529-545.

2. Garrett ME, Nauhaus I, Marshel JH, Callaway EM (2014) Topography and areal organization of mouse visual cortex. _J Neurosci_ 34:12587-12600.

3. Juavinett AL, Nauhaus I, Garrett ME, Zhuang J, Callaway EM (2017). Automated identification of mouse visual areas with intrinsic signal imaging. _Nature Protocols_. 12: 32-43.

4. Zhuang J, Ng L, Williams D, Valley M, Li Y, Garrett M, Waters J (2017) An extended retinotopic map of mouse cortex. _eLife_ 6: e18372.

5. Pologruto TA, Sabatini BL, Svoboda K. ScanImage: flexible software for operating laser scanning microscopes. _Biomed Eng Online_. 2003 May 17;2:13.

6. Nath, T., Mathis, A., Chen, A.C. _et al._ Using DeepLabCut for 3D markerless pose estimation across species and behaviors. _Nat Protoc_  14, 2152–2176 (2019).

7. Zoccolan D, Oertelt N, DiCarlo JJ, Cox DD. A rodent model for the study of invariant visual object recognition. Proc Natl Acad Sci U S A. 2009 May 26;106(21):8748-53.
