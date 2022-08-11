<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/76533398/177053936-9cff3e29-5f3a-48e1-8a00-bd7be8398ad6.jpg">
</p>

# Inception Net/Google Net

Glad to see you here. Now since you are here, lets begin implementing Advanced Computer Vision Architectures in Robotics.

<p align="center" width="100%">
    <img width="25%" src="https://user-images.githubusercontent.com/76533398/177054078-ef2f8dcb-a495-45be-9456-b807b05c4279.jpg">
</p>

Many people believe that increasing the depth of network(number of layers) or the number of neurons in each layer makes a better network. But this is partially correct. Deeper models brings these difficulties:
- **Overfitting**: It specially occurs when the training dataset is small
- **Computational Resources**: More number of parameters, more computation is required

Lets see how Inception Net tackle these problems. Inception Net uses sparsely connected architecture instead of fully connected architecture. See this image:

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/76533398/177054729-af94b73f-6d1a-44a9-b8f9-5068b5218937.png">
</p>
(<i>First image depicts fully connected network and second image depicts sparsely connected network</i>)

Inception Net basically consists of 27 layers. Here is the model summary:

<p align="center" width="100%">
    <img width="15%" src="https://user-images.githubusercontent.com/76533398/177054848-eb815db1-9685-48bf-9b8a-91d5bb86a4a2.png">
</p>

Notice in the above image that there is a layer called inception layer. This is actually the core concept of a sparsely connected architecture.

<p align="center" width="100%">
    <img width="40%" src="https://user-images.githubusercontent.com/76533398/177054931-6b9b9c6e-8367-443b-9943-0cec470314a9.png">
</p>

“(Inception Layer) is a combination of all those layers (namely, 1×1 Convolutional layer, 3×3 Convolutional layer, 5×5 Convolutional layer) with their output concatenated into a single output, forming the input of the next stage.”

Along with the above-mentioned layers, there are two major add-ons in the original inception layer:

- 1×1 Convolutional layer before applying another layer, which is mainly used for dimensionality reduction
- A parallel Max Pooling layer, which provides another option to the inception layer

<p align="center" width="100%">
    <img width="65%" src="https://user-images.githubusercontent.com/76533398/177055229-c51835dc-b51b-44db-92c4-5639484e2531.png">
</p>

---

<p align="center" width="100%">
    <img width="40%" src="https://user-images.githubusercontent.com/76533398/177055305-20cb112b-1ffc-4034-9558-e56e4137f538.jpg">
</p>

To understand the importance of the inception layer’s structure, lets the Hebbian principle from human learning. This says that <b>“neurons that fire together, wire together”</b>. When creating a subsequent layer in a deep learning model, one should pay attention to the learnings of the previous layer.

Suppose, for example, a layer in our deep learning model has learned to focus on individual parts of a face. The next layer of the network would probably focus on the overall face in the image to identify the different objects present there. Now to actually do this, the layer should have the appropriate filter sizes to detect different objects.

<p align="center" width="100%">
    <img width="35%" src="https://user-images.githubusercontent.com/76533398/177055413-4d8bd47a-a127-4d16-93ae-745504266fe3.png">
</p>

This is where the inception layer comes to the force. It allows the internal layers to pick and choose which filter size will be relevant to learn the required information. So even if the size of the face in the image is different (as seen in the images below), the layer works accordingly to recognize the face. For the first image, it would probably take a higher filter size, while it’ll take a lower one for the second image.

<p align="center" width="100%">
    <img width="35%" src="https://user-images.githubusercontent.com/76533398/177055464-13786bad-c7c0-4298-a989-5d37196fb12b.jpg">
</p>

Checkout this [tutotial](https://www.youtube.com/watch?v=uQc4Fs7yx5I) for implementing Inception Net in PyTorch. You can also look on official [PyTorch](https://pytorch.org/hub/pytorch_vision_inception_v3/) website. To implement this model, we will be using Transfer Learning Technique, which is exlained below.


## Transfer learning (TL)
<br><br>
<p align="center" width="100%">
    <img width="33%" src="https://user-images.githubusercontent.com/76533398/177052964-86672c98-d1b0-457b-b889-f22a28c811ed.jpg">
</p>

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.

As it takes lot of time to train the model using CNN but with the help of TL one one can train the model fastly.
It has the benefit of decreasing the training time for a neural network model and can result in lower generalization error.
<br><br>

<p align="center" width="100%">
    <img width="33%" src="https://user-images.githubusercontent.com/76533398/177053008-bfe8eb07-37de-40b5-b872-6d7963217da5.jpg">
</p>

A range of high-performing models have been developed for image classification and demonstrated on the annual ImageNet Large Scale Visual Recognition Challenge, or ILSVRC.
These models can be used as the basis for transfer learning in computer vision applications.

For implementation refer to this <a href="https://www.youtube.com/watch?v=K0lWSB2QoIQ" target="_blank">tutorial</a>.

For more knowledge <a href="https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/" target="_blank">reading</a>
<!--
# Task

<p align="center" width="100%">
    <img width="33%" src="https://user-images.githubusercontent.com/76533398/177055837-1e81e11f-f955-4193-bad8-4da81ce9ba5b.jpg">
</p>
-->
