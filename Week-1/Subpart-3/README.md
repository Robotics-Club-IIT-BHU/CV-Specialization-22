# Neural Networks

You’ve probably already been using neural networks on a daily basis. When you ask your mobile assistant to perform a search for you—say, Google or Siri or Amazon Web—or use a self-driving car, these are all neural network-driven. Computer games also use neural networks on the back end, as part of the game system and how it adjusts to the players, and so do map applications, in processing map images and helping you find the quickest way to get to your destination.

A neural network is a system or hardware that is designed to operate like a human brain.

Neural Networks are based on a collection of connected units (neurons), which, just like the synapses in a brain, can transmit a signal to other neurons, so that, acting like interconnected brain cells, they can learn and make decisions in a more human-like manner.

## Working of Neural Networks


A neural network is usually described as having different layers. The first layer is the input layer, it picks up the input signals and passes them to the next layer. The next layer does all kinds of calculations and feature extractions—it’s called the hidden layer. Often, there will be more than one hidden layer. And finally, there’s an output layer, which delivers the final result.

Therefore the 3 parts of a nueral network can be grouped as :-

  - Input layer has the job to pass the input vector to the Neural Network. If we have a matrix of 3 features (shape N x 3), this layer takes 3 numbers as the input and passes the same 3 numbers to the next layer.
  - Hidden layers represent the intermediary nodes, they apply several transformations to the numbers in order to improve the accuracy of the final result, and the output is defined by the number of neurons.
  - Output layer that returns the final output of the Neural Network. If we are doing a simple binary classification or regression, the output layer shall have only 1 neuron (so that it returns only 1 number). In the case of a multiclass classification with 5 different classes, the output layer shall have 5 neurons.

<details>
  <summary><h1>A single Nueron</h1></summary>

## The Linear Unit

So let's begin with the fundamental component of a neural network: the individual neuron. As a diagram, a neuron (or unit) with one input looks like:

<p align="center">
  <img src="https://i.imgur.com/mfOlDR6.png">
</p>
<br> 

The input is x. Its connection to the neuron has a weight which is w. Whenever a value flows through a connection, you multiply the value by the connection's weight. For the input x, what reaches the neuron is w * x. A neural network "learns" by modifying its weights.

The b is a special kind of weight we call the bias. The bias doesn't have any input data associated with it; instead, we put a 1 in the diagram so that the value that reaches the neuron is just b (since 1 * b = b). The bias enables the neuron to modify the output independently of its inputs.

The y is the value the neuron ultimately outputs. To get the output, the neuron sums up all the values it receives through its connections. This neuron's activation is y = w * x + b, or as a formula  y=wx+b.
  
  ## Example The Linear Unit as a Model
  
  Let us try to compute the calories by consuming a product which has many ingredients like sugar, protien, etc. First we will only consider sugar
  
  Training a model with 'sugars' (grams of sugars per serving) as input and 'calories' (calories per serving) as output, we might find the bias is b=90 and the weight is w=2.5. We could estimate the calorie content of a cereal with 5 grams of sugar per serving like this:
  
  <p align="center">
  <img src="https://i.imgur.com/yjsfFvY.png">
</p>
<br> 
  
  And, checking against our formula, we have  calories=2.5×5+90=102.5 , just like we expect.
  
  ## Multiple Inputs
  
  Now the Product contain not just sugar but multiple ingredients. What if we wanted to expand our model to include things like fiber or protein content? That's easy enough. We can just add more input connections to the neuron, one for each additional feature. To find the output, we would multiply each input to its connection weight and then add them all together.
  
  <p align="center">
  <img src="https://i.imgur.com/vyXSnlZ.png">
</p>
<br> 
  
  The formula for this neuron would be  y=w0x0+w1x1+w2x2+b . A linear unit with two inputs will fit a plane, and a unit with more inputs than that will fit a hyperplane.
  
  ## Linear Unit in Keras
  
  The easiest way to create a model in Keras is through keras.Sequential, which creates a neural network as a stack of layers. We can create models like those above using a dense layer (which we'll learn more about in the next lesson).

We could define a linear model accepting three input features ('sugars', 'fiber', and 'protein') and producing a single output ('calories') like so:
  
  ```python
  from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
  ```
  
  With the first argument, units, we define how many outputs we want. In this case we are just predicting 'calories', so we'll use units=1.

With the second argument, input_shape, we tell Keras the dimensions of the inputs. Setting input_shape=[3] ensures the model will accept three features as input ('sugars', 'fiber', and 'protein').

This model is now ready to be fit to training data!
  
</details>

<details>
  <summary><h1>Deep Nueral Networks</h1></summary>

One could say that all the Deep Learning models are Neural Networks but not all the Neural Networks are Deep Learning models. Generally speaking, “Deep” Learning applies when the algorithm has at least 2 hidden layers (so 4 layers in total including input and output). 

  ## Layers 
  
  Neural networks typically organize their neurons into layers. When we collect together linear units having a common set of inputs we get a dense layer.
  
  <p align="center">
  <img src="https://i.imgur.com/2MA4iMV.png">
</p>
<br> 
  
  You could think of each layer in a neural network as performing some kind of relatively simple transformation. Through a deep stack of layers, a neural network can transform its inputs in more and more complex ways. In a well-trained neural network, each layer is a transformation getting us a little bit closer to a solution.
  
  ## The Activation Function 
  
  It turns out, however, that two dense layers with nothing in between are no better than a single dense layer by itself. Dense layers by themselves can never move us out of the world of lines and planes. What we need is something nonlinear. What we need are activation functions.
  
  <p align="center">
  <img src="https://i.imgur.com/OLSUEYT.png">
</p>
<br> 
  
  An activation function is simply some function we apply to each of a layer's outputs (its activations). The most common is the rectifier function or ReLu. It is defined as max(0, x) i.e. if the output is > 0 than the answer will be x else it will be 0
  
  <p align="center">
  <img src="https://i.imgur.com/aeIyAlF.png">
</p>
<br> 
  
  The rectifier function has a graph that's a line with the negative part "rectified" to zero. Applying the function to the outputs of a neuron will put a bend in the data, moving us away from simple lines.

When we attach the rectifier to a linear unit, we get a rectified linear unit or ReLU. (For this reason, it's common to call the rectifier function the "ReLU function".) Applying a ReLU activation to a linear unit means the output becomes max(0, w * x + b), which we might draw in a diagram like:
  
  <p align="center">
  <img src="https://i.imgur.com/eFry7Yu.png">
</p>
<br> 
  
  ![image](https://user-images.githubusercontent.com/77875542/173996150-678b5ff0-6a27-44e3-ae33-3d62efc3f1b6.png)

  
  ## Stacking Dense layers
  
  Now that we have some nonlinearity, let's see how we can stack layers to get complex data transformations.

  <p align="center">
  <img src="https://i.imgur.com/Y5iwFQZ.png">
</p>
<br> 
  
  The layers before the output layer are sometimes called hidden since we never see their outputs directly.
  
  ![WhatsApp Image 2022-06-16 at 7 35 50 PM](https://user-images.githubusercontent.com/77875542/174237023-40646a0f-20a3-4460-a31b-9cbd7a4a0829.jpeg)


Now, notice that the final (output) layer is a linear unit (meaning, no activation function). That makes this network appropriate to a regression task, where we are trying to predict some arbitrary numeric value. Other tasks (like classification) might require an activation function on the output.
  
  <p align="center">
  <img src="Assest/ZomboMeme%2009062022150752.jpg">
</p>
<br> 

## Building Sequential Models
  
The Sequential model we've been using will connect together a list of layers in order from first to last: the first layer gets the input, the last layer produces the output. This creates the model in the figure above:
  
  ```python
  from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
  ```
  
  You Can use the above model to train any data 

</details>

<details>
  <summary><h1>Stochastic Gradient Descent</h1></summary>
  
  ## Introduction
In the first two lessons, we learned how to build fully-connected networks out of stacks of dense layers. When first created, all of the network's weights are set randomly -- the network doesn't "know" anything yet. In this lesson we're going to see how to train a neural network; we're going to see how neural networks learn.

As with all machine learning tasks, we begin with a set of training data. Each example in the training data consists of some features (the inputs) together with an expected target (the output). Training the network means adjusting its weights in such a way that it can transform the features into the target. In the 80 Cereals dataset, for instance, we want a network that can take each cereal's 'sugar', 'fiber', and 'protein' content and produce a prediction for that cereal's 'calories'. If we can successfully train a network to do that, its weights must represent in some way the relationship between those features and that target as expressed in the training data.

In addition to the training data, we need two more things:

- A "loss function" that measures how good the network's predictions are.
- An "optimizer" that can tell the network how to change its weights.
  
## The Loss Function
  
We've seen how to design an architecture for a network, but we haven't seen how to tell a network what problem to solve. This is the job of the loss function.

The loss function measures the disparity between the the target's true value and the value the model predicts.

Different problems call for different loss functions. We have been looking at regression problems, where the task is to predict some numerical value -- calories in 80 Cereals, rating in Red Wine Quality. Other regression tasks might be predicting the price of a house or the fuel efficiency of a car.

A common loss function for regression problems is the mean absolute error or MAE. For each prediction y_pred, MAE measures the disparity from the true target y_true by an absolute difference abs(y_true - y_pred).

The total MAE loss on a dataset is the mean of all these absolute differences.
  
  ![image](https://user-images.githubusercontent.com/77875542/173977809-542cdc5f-9db6-4dc0-9a4b-3f08fb6123f7.png)


A graph depicting error bars from data points to the fitted line..
The mean absolute error is the average length between the fitted curve and the data points.
Besides MAE, other loss functions you might see for regression problems are the mean-squared error (MSE) or the Huber loss (both available in Keras).

During training, the model will use the loss function as a guide for finding the correct values of its weights (lower loss is better). In other words, the loss function tells the network its objective.

The Optimizer - Stochastic Gradient Descent
We've described the problem we want the network to solve, but now we need to say how to solve it. This is the job of the optimizer. The optimizer is an algorithm that adjusts the weights to minimize the loss.

Virtually all of the optimization algorithms used in deep learning belong to a family called stochastic gradient descent. They are iterative algorithms that train a network in steps. One step of training goes like this:
  
   - Sample some training data and run it through the network to make predictions.
   - Measure the loss between the predictions and the true values.
   - Finally, adjust the weights in a direction that makes the loss smaller.
  
  

  
Then just do this over and over until the loss is as small as you like (or until it won't decrease any further.)
  
  ![image](https://user-images.githubusercontent.com/77875542/173977968-d7d6d674-ff72-4a9a-a3fe-5775bdf091c7.png)


Each iteration's sample of training data is called a minibatch (or often just "batch"), while a complete round of the training data is called an epoch. The number of epochs you train for is how many times the network will see each training example.
  
  The animation shows the linear model from Lesson 1 being trained with SGD. The pale red dots depict the entire training set, while the solid red dots are the minibatches. Every time SGD sees a new minibatch, it will shift the weights (w the slope and b the y-intercept) toward their correct values on that batch. Batch after batch, the line eventually converges to its best fit. You can see that the loss gets smaller as the weights get closer to their true values.

  ## Learning Rate and Batch Size
  
Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way). The size of these shifts is determined by the learning rate. A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values.

The learning rate and the size of the minibatches are the two parameters that have the largest effect on how the SGD training proceeds. Their interaction is often subtle and the right choice for these parameters isn't always obvious. (We'll explore these effects in the exercise.)

Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results. Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.

  ## Adding the Loss and Optimizer
  
After defining a model, you can add a loss function and optimizer with the model's compile method:
  
  ```python
  model.compile(
    optimizer="adam",
    loss="mae",
)
  ```
  
  Notice that we are able to specify the loss and optimizer with just a string. You can also access these directly through the Keras API -- if you wanted to tune parameters, for instance -- but for us, the defaults will work fine.
  
  ## Example - Red Wine Quality
  
Now we know everything we need to start training deep learning models. So let's see it in action! We'll use the Red Wine Quality dataset.

This dataset consists of physiochemical measurements from about 1600 Portuguese red wines. Also included is a quality rating for each wine from blind taste-tests. How well can we predict a wine's perceived quality from these measurements?

We've put all of the data preparation into this next hidden cell. It's not essential to what follows so feel free to skip it. One thing you might note for now though is that we've rescaled each feature to lie in the interval  [0,1] . As we'll discuss more in Lesson 5, neural networks tend to perform best when their inputs are on a common scale.
  
  ```python
  import pandas as pd
from IPython.display import display

red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']
  ```
  
  How many inputs should this network have? We can discover this by looking at the number of columns in the data matrix. Be sure not to include the target ('quality') here -- only the input features
  
  ```python
  print(X_train.shape)
  ```
  
  you will see the output (1119, 11)
  
  Eleven columns means eleven inputs.

We've chosen a three-layer network with over 1500 neurons. This network should be capable of learning fairly complex relationships in the data.
  
  ```python
  from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
  ```
  
  Deciding the architecture of your model should be part of a process. Start simple and use the validation loss as your guide. You'll learn more about model development in the exercises.

After defining the model, we compile in the optimizer and loss function.

```python
  model.compile(
    optimizer='adam',
    loss='mae',
)
  ```
  
Now we're ready to start the training! We've told Keras to feed the optimizer 256 rows of the training data at a time (the batch_size) and to do that 10 times all the way through the dataset (the epochs).

  ```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)
  ```
  
  You can see that Keras will keep you updated on the loss as the model trains.

Often, a better way to view the loss though is to plot it. The fit method in fact keeps a record of the loss produced during training in a History object. We'll convert the data to a Pandas dataframe, which makes the plotting easy.
  
  ```python
  import pandas as pd

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot()
  ```
  
  ![image](https://user-images.githubusercontent.com/77875542/173979411-78b0bff9-7867-45dc-bd7f-f94beb20df68.png)

  Notice how the loss levels off as the epochs go by. When the loss curve becomes horizontal like that, it means the model has learned all it can and there would be no reason continue for additional epochs.

  
</details>

<details>
  <summary><h1>PyTorch</h1></summary>
  
  We first taught you how to code in tensorflow but most of the industries uses pytorch and tensorflow is rarely used in real life examples.
  
  ![image](https://user-images.githubusercontent.com/77875542/173994207-2e102563-130d-43a8-af4f-3fe29ead527e.png)

  
  So let's Learn Pytorch
  
  <p align="center">
  <img src="https://github.com/sherlockholmes1603/computer-vision-workshop/blob/master/Week-1/Subpart-3/Assest/pytorch_.jpeg">
</p>
<br> 
  
  ## Introduction 
  
  - It’s a Python based scientific computing package targeted at two sets of audiences:
      - A replacement for NumPy to use the power of GPUs
      - Deep learning research platform that provides maximum flexibility and speed
  - pros:
      - Interactively debugging PyTorch. Many users who have used both frameworks would argue that makes pytorch significantly easier to debug and visualize.
      - Clean support for dynamic graphs
      - Organizational backing from Facebook
      - Blend of high level and low level APIs
  - cons:
    - Much less mature than alternatives
    - Limited references / resources outside of the official documentation
  
  See the three jupyter notebooks on Pytorch Tutorials
  
  Also we really have many good playlist on YouTube on PyTorch
  
  [Tutorial 1](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4) Only till 13th tutorial
  
  [Tutorial 2](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN) Only till 6th tutorial
  
  
  
  

</details>
  
  
  
# Task
Implement your knowledge of neural networks in this [assignment](https://colab.research.google.com/drive/1GMRM-OThrEAwJ98_4IZodsM4jXh4lO_H?usp=sharing). Submit your notebook(.ipynb file) [here](https://forms.gle/ow94wi18vVogabRLA).
