# 3. Data Pre-Processing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pseudo-Lab/Tutorial-Book/blob/master/book/chapters/time-series/Ch3-preprocessing.ipynb)

from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/mgkUDA-V9oA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


In the previous chapter, we explored dataset features through an EDA. In this chapter, we will learn how to pre-process data for a time series task.

Pre-processing data into pairs of features and target variable is required in order to use a sequential dataset for supervised learning. In addition, it is necessary to unify data scales to stably train deep learning models. In chapter 3.1, we will transform the raw data of COVID-19 confirmed cases into data for supervised learning, and in chapter 3.2, we will examine how to perform data scaling. 

## 3.1 Preparing Data for Supervised Learning

We will load a dataset for data pre-processing, using code introduced in chapter 2.1.

!git clone https://github.com/Pseudo-Lab/Tutorial-Book-Utils
!python Tutorial-Book-Utils/PL_data_loader.py --data COVIDTimeSeries
!unzip -q COVIDTimeSeries.zip

Let's extract `daily_cases`, which shows the daily confirmed COVID-19 cases for South Korea, using code introduced in chapter 2.3.

import pandas as pd
confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
confirmed[confirmed['Country/Region']=='Korea, South']
korea = confirmed[confirmed['Country/Region']=='Korea, South'].iloc[:,4:].T
korea.index = pd.to_datetime(korea.index)
daily_cases = korea.diff().fillna(korea.iloc[0]).astype('int')
daily_cases

We need to convert the time series data shown above into pairs of input and output variables to use them for supervised learning. In a time series task, we call this kind of data as sequential data. Firstly, we need to define the sequence length in order to transform the data into sequential data. The sequence length is decided by how many days from the data we wish to use to predict future cases. For example, for a sequence length of 5, data in $t-1$, $t-2$, $t-3$, $t-4$, and $t-5$ are used to predict data in time $t$. Likewise, a task where we predict variable at time $t$ using data from $t-k$ to $t-1$ is called an one-step prediction task.

The `create_sequences` function defined below transforms time series data with size `N` into data with a <code>N - seq-length</code> size for supervised learning (See Figure 3-1).

![](https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch3img01.png?raw=true)


- Figure 3-1 Transforming Process of Times Series Data

import numpy as np

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data.iloc[i:(i+seq_length)]
        y = data.iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(daily_cases, seq_length)

With `seq_length` = 5, applying the `create_sequences` function to `daily_cases`, we got 327 samples in total for supervised learning. 

X.shape, y.shape

We will divide the transformed dataset into training, validation, and test datasets with an 8:1:1 ratio. The total number of data is 327, so the division of each dataset results in the following: 261 data for training, 33 data for validation, and 33 data for testing.

train_size = int(327 * 0.8)
print(train_size)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+33], y[train_size:train_size+33]
X_test, y_test = X[train_size+33:], y[train_size+33:]

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

## 3.2 Data Scaling

In chapter 3.2, we will perform data scaling. More specifically, we will perform MinMax scaling, which transforms the data range to between 0 and 1. Apply the following mathematical notation for MinMax scaling after calculating the minimum and maximum values of the data group.


> $x_{scaled} = \displaystyle\frac{x - x_{min}}{x_{max} - x_{min}}$

The data scaling for the training, validation, and test datasets must be processed based on the statistics of the training data. Input variables from the testing dataset should not be used, so we must perform training dataset scaling using the statitistics of the training data.<br>Since the model was trained with the statistics of the training data, the test data must also be scaled based on the same values in order to evaluate model performance later. Similarly, the validation data require data scaling based on the statistics of the training data, since validation data need to go through the same process of pre-processing as the test data.

We will get the minimum and maximum values from the `X_train` data in order to apply MinMax scaling.

MIN = X_train.min()
MAX = X_train.max()
print(MIN, MAX)

The minimum and maximum values are 0 and 851, respectively. Next, we will define the MinMax scaling function. 

def MinMaxScale(array, min, max):

    return (array - min) / (max - min)

Let's perform scaling using the `MinMaxScale` function.

X_train = MinMaxScale(X_train, MIN, MAX)
y_train = MinMaxScale(y_train, MIN, MAX)
X_val = MinMaxScale(X_val, MIN, MAX)
y_val = MinMaxScale(y_val, MIN, MAX)
X_test = MinMaxScale(X_test, MIN, MAX)
y_test = MinMaxScale(y_test, MIN, MAX)

Next, we will transform the data type from `np.array` into `torch.Tensor` in order for the data to be input in a PyTorch model. First, we will define the function for transforming the data type. 

import torch

def make_Tensor(array):
    return torch.from_numpy(array).float()

We will perform the transformation through the `make_Tensor` function.

X_train = make_Tensor(X_train)
y_train = make_Tensor(y_train)
X_val = make_Tensor(X_val)
y_val = make_Tensor(y_val)
X_test = make_Tensor(X_test)
y_test = make_Tensor(y_test)

So far, we have practiced transforming data into the correct format for the supervised learning of time series and data scaling. In the next chapter, we will build a prediction model for COVID-19 cases with the data we curated. 