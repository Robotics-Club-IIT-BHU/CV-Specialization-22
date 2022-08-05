# Semantic segmentation

Here we will show how to use convolutional neural networks for the task of semantic image segmentation. Image segmentation is a computer vision task in which we label specific regions of an image according to what's being shown.

More specifically, the goal of semantic image segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we're predicting for every pixel in the image, this task is commonly referred to as dense prediction.

![image](https://user-images.githubusercontent.com/77875542/182614415-4a5b6ace-ba0b-4548-8765-b4fe9a871388.png)



One important thing to note is that we're not separating instances of the same class; we only care about the category of each pixel. In other words, if you have two objects of the same category in your input image, the segmentation map does not inherently distinguish these as separate objects. There exists a different class of models, known as instance segmentation models, which do distinguish between separate objects of the same class.

Segmentation models are useful for a variety of tasks, including:

 - Autonomous vehicles

We need to equip cars with the necessary perception to understand their environment so that self-driving cars can safely integrate into our existing roads.

![image](https://user-images.githubusercontent.com/77875542/182614559-210266be-3198-4b19-ad72-688438862292.png)


Medical image diagnostics
Machines can augment analysis performed by radiologists, greatly reducing the time required to run diagnositic tests.
chest xray

![image](https://user-images.githubusercontent.com/77875542/182614628-f992be14-c938-4e38-b4ba-961ba60720b7.png)

But how to do semantic segmentation?? Here comes UNET.


# Understanding the UNet architecture

UNet, evolved from the traditional convolutional neural network, was first designed and applied in 2015 to process biomedical images. As a general convolutional neural network focuses its task on image classification, where input is an image and output is one label, but in biomedical cases, it requires us not only to distinguish whether there is a disease, but also to localise the area of abnormality.

UNet is dedicated to solving this problem. The reason it is able to localise and distinguish borders is by doing classification on every pixel, so the input and output share the same size. For example, for an input image of size 2x2:

```bash
[[255, 230], [128, 12]]  # each number is a pixel
```

the output will have the same size of 2x2:

```bash
[[1, 0], [1, 1]]  # could be any number between [0, 1]
```

Now let’s get to the detail implementation of UNet. I will:

  - Show the overview of UNet
  - Breakdown the implementation line by line and further explain it
  
## Overview

The network has basic foundation looks like:

![image](https://user-images.githubusercontent.com/77875542/181906159-a98434b7-fe5d-4c72-9069-d73b08965cb2.png)



First sight, it has a “U” shape. The architecture is symmetric and consists of two major parts — the left part is called contracting path, which is constituted by the general convolutional process; the right part is expansive path, which is constituted by transposed 2d convolutional layers(you can think it as an upsampling technic for now).

Now let’s have a quick look at the implementation:

```python
def build_model(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

input_layer = Input((img_size_target, img_size_target, 1))
output_layer = build_model(input_layer, 16)
```



Now let’s break down the implementation line by line and maps to the corresponding parts on the image of UNet architecture.

## Line by Line Explanation

### Contracting Path

The contracting path follows the formula:

```bash
conv_layer1 -> conv_layer2 -> max_pooling -> dropout(optional)
```

So the first part of our code is:

```python
conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
pool1 = Dropout(0.25)(pool1)
```


which matches to:

![image](https://user-images.githubusercontent.com/77875542/181906233-26a0234a-dba9-42b9-a6d9-6a4122c31467.png)



Notice that each process constitutes two convolutional layers, and the number of channel changes from 1 → 64, as convolution process will increase the depth of the image. The red arrow pointing down is the max pooling process which halves down size of image(the size reduced from 572x572 → 568x568 is due to padding issues, but the implementation here uses padding= “same”).

The process is repeated 3 more times:

![image](https://user-images.githubusercontent.com/77875542/181906262-04727e6b-9ba8-4d0b-ac56-4a603265bb17.png)



with code:

```python
conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)
pool2 = Dropout(0.5)(pool2)

conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)
pool3 = Dropout(0.5)(pool3)

conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(0.5)(pool4)
```

and now we reaches at the bottommost:

![image](https://user-images.githubusercontent.com/77875542/181906333-e16521a4-edbb-47ae-9d07-e4a6b7166b67.png)



still 2 convolutional layers are built, but with no max pooling:

```python
# Middle
convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
```


The image at this moment has been resized to 28x28x1024. Now let’s get to the expansive path.

Expansive Path
In the expansive path, the image is going to be upsized to its original size. The formula follows:

```bash
conv_2d_transpose -> concatenate -> conv_layer1 -> conv_layer2
```

![image](https://user-images.githubusercontent.com/77875542/182107163-afb83fef-1043-41d5-bdf6-d0ac48a906a3.png)

```python
deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
uconv4 = concatenate([deconv4, conv4])
uconv4 = Dropout(0.5)(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
```

Transposed convolution is an upsampling technic that expands the size of images. There is a visualised demo here and an explanation here. Basically, it does some padding on the original image followed by a convolution operation.

After the transposed convolution, the image is upsized from 28x28x1024 → 56x56x512, and then, this image is concatenated with the corresponding image from the contracting path and together makes an image of size 56x56x1024. The reason here is to combine the information from the previous layers in order to get a more precise prediction.

In line 4 and line 5, 2 other convolution layers are added.

Same as before, this process is repeated 3 more times:

```python
deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = concatenate([deconv3, conv3])
uconv3 = Dropout(0.5)(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = concatenate([deconv2, conv2])
uconv2 = Dropout(0.5)(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = concatenate([deconv1, conv1])
uconv1 = Dropout(0.5)(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
```


Now we’ve reached the uppermost of the architecture, the last step is to reshape the image to satisfy our prediction requirements.

![image](https://user-images.githubusercontent.com/77875542/182107376-c8e6f7e3-072d-429e-aefb-a78cc2c62c12.png)

```python
output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
```



The last layer is a convolution layer with 1 filter of size 1x1(notice that there is no dense layer in the whole network). And the rest left is the same for neural network training.

For more understanding of UNet and its PyTorch implementation, follow this [video](https://www.youtube.com/watch?v=IHq1t7NxS8k).

## Conclusion

UNet is able to do image localisation by predicting the image pixel by pixel and the author of UNet claims in his paper that the network is strong enough to do good prediction based on even few data sets by using excessive data augmentation techniques. There are many applications of image segmentation using UNet and it also occurs in lots of competitions. One should try out on yourself and I hope this post could be a good starting point for you.

# Assignment

Now it's time for you to solve the assignment. 

You have to implement U-net architecture using U-net in the following Kaggle dataset

https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge

You don't need to download the whole dataset for the assignment but you can use the kaggle kernel for the same

Here are some tutorials for implementing semantic segmentation using unet in pytorch

https://www.youtube.com/watch?v=IHq1t7NxS8k

https://www.youtube.com/watch?v=T0BiFBaMLDQ&t=216s

https://www.youtube.com/watch?v=u1loyDCoGbE




