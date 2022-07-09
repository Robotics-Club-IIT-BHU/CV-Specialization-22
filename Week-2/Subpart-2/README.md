<p align="center" width="100%">
    <img width="33%" src="https://user-images.githubusercontent.com/76533398/177035566-a4e29c2e-8224-46d0-9525-7788ef559471.jpeg">
</p>

# Image Classification

Since you people are here, we hope you have understood basics of machine learning and computer vision. Now let's "<i>skip to the good part</i>", i.e, its implementation. The most common use of computer vision and convolutional neural networks is image classification, similar to what you did in the second assignment of week-1.

Image classification is one of the most important applications of Deep Learning and Artificial Intelligence. It refers to assigning labels to images based on certain characteristics or features present in them. The algorithm identifies these features and uses them to differentiate between different images and assign labels to them.

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/76533398/177038899-884486e6-0602-4682-8db2-8d5217969936.png">
</p>

We will see what how Deep Convolutional Neural Networks are made and some existing models, but before that, let's talk about one of the most important task in computer vision, Data Preprocessing.

## Data(Image) Preprocessing

The acquired data are usually messy and come from different sources. To feed them to the ML model (or neural network), they need to be standardized and cleaned up. More often than not, preprocessing is used to conduct steps that reduce the complexity and increase the accuracy of the applied algorithm.

Other than cleaning, we aso do Data Augmentation, in which we apply several operation on images, like rotation, flipping, etc., to bring a variety in the dataset and making it more generalised, which helps in better learning of model and generates better accuracy. Here we list some commonly used operations on images:

- Padding
- Random Rotating
- Re-Scaling,
- Vertical and Horizontal Flipping
- Translation ( Image is moved along X, Y direction)
- Cropping
- Zooming
- Darkening & Brightening/Color Modification
- Grayscaling
- Changing Contrast
- Adding Noise
- Random Erasing

Augmentation techniques are also used to enlarge the size of dataset, so that model gets trained on more number of and generalised images.

<p align="center" width="100%">
    <img width="40%" src="https://user-images.githubusercontent.com/76533398/177040007-04e0d09b-82a5-429d-854c-8c10a8c82702.jpg">
</p>

Check out this [tutorial](https://www.youtube.com/watch?v=Zvd276j9sZ8) for implementing data augmentation in PyTorch.

Before moving forward,watch [this](https://www.youtube.com/watch?v=dXB-KQYkzNU&t=389s) video to get to know about **Batch Normalization**.

## Deep Convolutional Neural Networks

Deep Convolutional Neural Networks consits of two sections:

- **Feature Extraction**: For feature extraction, we place many convolution units in the network. A convolutional unit generally consists four layers:
  - Convolution
  - Batch Normalization
  - Activation
  - Pooling<br>
These feature extractors, first extract low-level features (say edges, lines), then mid-level features as shapes or combinations from several low-level   features, and eventually high-level features, say an ear/nose/eyes in the case of a cat.
  
- **Classification**: The convolution units are followed by fully connected layers for flattening and an activation layer for producing output.

<br>You can check out this online [tutorial](https://www.youtube.com/watch?v=YRhxdVk_sIs) for better understanding and this [tutorial](https://youtu.be/wnK3uWv_WkU) for PyTorch implementation

![dcnn](https://user-images.githubusercontent.com/76533398/177041340-a2ce57ac-f604-40f6-9aa3-2482335897b2.jpeg)

# Task

<p align="center" width="100%">
    <img width="30%" src="https://user-images.githubusercontent.com/76533398/177055867-53f8c56c-1849-40b0-986f-02107a87083d.jpeg">
</p>
