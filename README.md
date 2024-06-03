# Computer vision projects

This repository includes various CV projects that I've completed, now it consists of various competition models I have worked on.

## Folders
### CAGS_segmentation_and_classification
  - Image classfication and segmentation models for the CAGS (Cats and dogs) dataset (examples) `https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/demos/cags_train.html`
  -  Preprocessing done akin to RandAugmnent on both tasks
  -  Using Keras
  - Classification
    - Classification of color images into 34 classes
    - Uses an EfficientV2 as a base model and a classification head on top
    - Final predictions done via ensembling
 - Segmentation
    - Prediction of a boolean mask on the input image, representing the location of the Cat/Dog on the image
    - U-Net based architecture using EfficientNetV2 as a backbone on the way down, and training the upscaling layers maually
    - Final predictions via ensembling
    
## CIFAR_classification
- CIFAR-10 classification model in Keras
- Various architectures based on WideNet, EfficientNet and Unet
- Affine image augmentations and grayscaling

## SVHN_object_detection
- StreetView House Number dataset (link) `http://ufldl.stanford.edu/housenumbers/`
- Prediction of bounding boxes and classification of the numbers in the bounding boxes
- RetinaNet-like single stage detector

## Modelnet_3d_recognition
- ModelNet 3D classification based on `https://modelnet.cs.princeton.edu/`, however with fewer classes
- Using 3D convolutions with WideNet-like architecture
