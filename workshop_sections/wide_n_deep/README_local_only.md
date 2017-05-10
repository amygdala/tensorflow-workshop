# Wide & Deep with TensorFlow

This directory contains the code for running a Wide and Deep model. 
This code currently only works on Python 2.7.

# Installation
This code runs on both Python 2.7, so please use Python 2.7 for your environment.
* Tensorflow: https://www.tensorflow.org/install

There are some Jupyter notebooks available for working through the code step by step.
* Jupyter: https://jupyter.org/install.html

# About the dataset and model
Wide and deep jointly trains wide linear models and deep neural networks -- to combine the benefits of memorization and generalization for recommender systems. See the [research paper](https://arxiv.org/abs/1606.07792) for more details. The code is based on the [TensorFlow wide and deep tutorial](https://www.tensorflow.org/tutorials/wide_and_deep/).

We will use the [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income) to predict the probability that the individual has an annual income of over 50,000 dollars. This data was extracted from the 1994 US Census by Barry Becker. 

The prediction task is to determine whether a person makes over 50K a year.

The dataset is downloaded as part of the code.

# Training and evaluation

The commands below assume you are in this directory (wide_n_deep). 

You should move to it with `cd workshop_sections/wide_n_deep`

### Local
You can run the Python module directly on your local environment using:
`python widendeep/model.py`

### Jupyter Notebook
Run the notebook, and step through the cells.

`jupyter notebook`

    
# Your trained model
Whichever path you chose, you should now have a set of files exported. 
It will be located in someplace similar to `models/model_WIDE_AND_DEEP_1234567890/exports/1234567890`. 

The trained model files that were exported are an optimized graph ready to be used for prediction. 
