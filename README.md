# Learn-Keras

## What is Keras?
>Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
## Why use Keras?
>The Keras API is easier to use than TensorFlow, allowing us to create, train, and evaluate a deep learning model with considerably less code.
## Sequential Model
In Keras, every neural network model is an instance of the **Sequential** object. This acts as the container of the neural network, allowing us to build the model by stacking multiple layers inside the **Sequential** object.
### Building a multilayer perceptron (MLP) using the sequential model
When building a model, we start off by initializing a Sequential object. We can initialize an empty Sequential object and add layers onto the model using the add function, or we can directly initialize the Sequential object with a list of layers.

The most commonly used Keras neural network layer is the Dense layer. This represents a fully-connected layer in the neural network.

* Using the add() function
  ```python  
    model = Sequential()
    layer1 = Dense(5, input_dim=4)
    model.add(layer1)
    layer2 = Dense(3, activation='relu')
    model.add(layer2)
  ```  
* Using the constructor
    ```python
    layer1 = Dense(5, input_dim=4)
    layer2 = Dense(3, activation='relu')
    model = Sequential([layer1, layer2])
    ```