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
    layer3 = Dense(3, activation='softmax')
    model.add(layer3)
  ```  
* Using the constructor
    ```python
    layer1 = Dense(5, input_dim=4)
    layer2 = Dense(3, activation='relu')
    layer3 = Dense(3, activation='softmax')
    model = Sequential([layer1, layer2,layer3])
    ```
#### Model configuration for training
Configuring the model for training is done using the **compile()** function.The function takes in a single required argument, which is the optimizer to use, e.g. ADAM. A shorthand way to specify the optimizer is to use a (lowercase) string of the optimizer's name (e.g. `'adam'`, `'sgd'`, `'adagrad'`).

The two main keyword arguments to know are loss and metrics. The loss keyword argument specifies the loss function to use. For binary classification, we set the value to `'binary_crossentropy'`, which is the binary cross-entropy function. For multiclass classification, we set the value to `'categorical_crossentropy'`, which is the multiclass cross-entropy function.
The metrics keyword argument takes in a list of strings, representing metrics we want to track during training and evaluation.

* For binary-classification 
```python
model.compile('adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

* For multiclass-classification
```python
model.compile('adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
```

#### Model training 
After configuring a Keras model for training, it only takes a single line of code to actually perform the training. We use the `Sequential` model's `fit` function to train the model on input data and labels.

The first two arguments of the `fit` function are the input data and labels, respectively.The training batch size can be specified using the `batch_size` keyword argument.We can also specify the number of epochs.

```python
train_output = model.fit(data, labels,
                         batch_size=20, epochs=5)
```

The output of the `fit` function is a **History** object, which records the training metrics. The object's `history` attribute is a dictionary that contains the metric values at each epoch of training.

```python
print(train_output.history)
```

#### Evaluation
We use the Sequential model's evaluate function, which also takes in data and labels (NumPy arrays) as its first two arguments.
Calling the `evaluate` function will evaluate the model over the entire input dataset and labels. The function returns a list containing the evaluation loss as well as the values for each metric specified during model configuration.

```python
print(model.evaluate(eval_data, eval_labels))
```

#### Prediction
Finally, we can make predictions with a Keras model using the `predict` function. The function takes in a NumPy array dataset as its required argument, which represents the data observations that the model will make predictions for.
The output of the `predict` function is the output of the model. That means for classification, the `predict` function is the model's class probabilities for each data observation.

```python
print(model.predict(new_data))
```


