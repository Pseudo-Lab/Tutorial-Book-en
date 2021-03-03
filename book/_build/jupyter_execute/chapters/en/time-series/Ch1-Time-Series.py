# 1. Introduction to Time Series

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pseudo-Lab/Tutorial-Book/blob/master/book/chapters/time-series/Ch1-Time-Series.ipynb)

from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/DRZFhCBsGQY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


Time series prediction is the prediction of future values based on observed values in the past. It can be defined as a supervised learning problem in that it requires finding patterns between observed data in the past and future values. Therefore, in this chapter, we will build a model that predicts future values through supervised learning based on a neural network structure.

Time series prediction is a skill that is required in many areas. One of the areas is in the energy domain. Electric power plants need to predict future power demand to secure sufficient amounts of reserved power, and city gas companies need a future usage prediction model to take preemptive measures against meter reader failure and meter reader cheating. In fact, these issues were presented for data science competitions ([electric power](https://dacon.io/competitions/official/235606/overview/), [city gas](https://icim.nims.re.kr/platform/question/16)) to facilitate the discovery of new models. In addition, stakeholders in the retail domain are interested in predicting the sales volumes of items for efficient product management, which was also the topic of a data science competition.([distribution](https://www.kaggle.com/c/m5-forecasting-accuracy/overview))

In this tutorial, we will build a model that predicts future confirmed cases of COVID-19 based on data from the past confirmed cases using [COVID-19 confirmed case data](https://github.com/CSSEGISandData/COVID-19) provided by Johns Hopkins University's Center for Systems Science and Engineering. In chapter 1, we will look at the neural network structures that can be used when building a time series prediction model, and check the metrics that can be used when evaluating model performance. In chapter 2, we will deepen our understanding of COVID-19 confirmed case data through exploratory data analysis, and in chapter 3 we will learn how to restructure time series data so it can be used for supervised learning. In chapters 4 and 5, we will use deep learning models to predict future confirmed cases.


## 1.1 Available Deep Learning Architectures

### 1.1.1 CNN

<p align="center"> <img align="center" src="https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch1img01.PNG?raw=true"></p>


- Figure 1-1. Example of CNN structure (Reference: Lim et al. 2020. Time Series Forecasting With Deep Learning: A Survey)


In general, CNN is a network structure that performs well in computer vision tasks. However, CNNs can also be applied to time series prediction. A weighted sum between input sequence data can be calculated using a one-dimensional convolution filter to calculate the predicted future value. However, the CNN structure does not take into account the temporal dependence between past and future data. 

### 1.1.2 RNN

<p align="center"> <img align="center" src="https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch1img02.PNG?raw=true"></p>


- Figure 1-2. Example of RNN Structure (Reference: Lim et al. 2020. Time Series Forecasting With Deep Learning: A Survey) 

RNN is a network structure that is frequently used in solving natural language processing tasks, and it utilizes hidden state information accumulated in previous states for future prediction. By doing so, it is possible to manipulate past information to calculate future forecasts. However, if the given input sequence is too large, a vanishing gradient problem may occur that adversely affects model training. Therefore, the LSTM structure, which solves the vanishing gradient problem, is used frequently. We will use the LSTM structure in this tutorial. 

### 1.1.3 Attention Mechanism

<p align="center"> <img align="center" src="https://github.com/Pseudo-Lab/Tutorial-Book/blob/master/book/pics/TS-ch1img03.PNG?raw=true"></p>


- Figure 1-3. Example of Attention Mechanism (Reference: Lim et al. 2020. Time Series Forecasting With Deep Learning: A Survey) 

When predicting the future based on past information, there can be information from the past that is helpful and information that isn't. For example, if a retailer wants to predict weekend sales, it may be helpful to consider weekend sales from the previous week rather than sales on the weekdays leading up to the prediction period. By using the attention mechanism, it is possible to manipulate these factors into the model. It calculates the importance of past points in time and uses them to inference future values. More accurate prediction is possible by assigning more weight to the value that is directly influencing the point to be predicted. 

## 1.2 Evaluation Metric

In this tutorial, we are going to build a model for predicting confirmed cases of COVID-19. Since the confirmed cases are measured with continuous values, the performance of the model can be evaluated through the gaps between the predicted and actual values. In this section, we'll look at various ways to calculate the gaps between the predicted and actual values. We will first define some symbols before diving in.


> $y_i$: Actual values that need to be predicted <br> $\hat{y}_i$: Values predicted by the model <br> $n$: Size of the test dataset


Chapters 1.2.1 to 1.2.4 will use the symbols above, but the definition of the symbols will change in chapter 1.2.5. Please be aware of this change.

### 1.2.1 MAE (Mean Absolute Error)

> $MAE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} |y_i-\hat{y}_i|$

MAE, also known as L1 Loss, can be calculated by dividing the sum of the absolute differences between the predicted values and the actual values by the number of samples(n). Since this is the process for calculating an average, from now on we will refer to this as 'calculating the mean'. Since the scale of MAE is the same scale as the target variable being predicted, the meaning of the value can be understood intuitively. The implemented code looks like this: 

import numpy as np #import numpy package

def MAE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs(true-pred))

TRUE = np.array([10, 20, 30, 40, 50])
PRED = np.array([30, 40, 50, 60, 70])

MAE(TRUE, PRED)

### 1.2.2 MSE (Mean Squared Error)

> $MSE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} (y_i-\hat{y}_i)^2$

> $RMSE=\sqrt{\frac{1}{n}\displaystyle\sum_{i=1}^{n} (y_i-\hat{y}_i)^2}$


MSE, also known as L2 Loss, is calculated by taking the mean of the squared differences between the predicted values and the actual values. The more the predicted values deviate from the actual values, the more the MSE value will increase. It will increase exponentially. Since the calculated value is squared, the scale of the target variable and MSE is different. In order to match the scale of the target value, we need to calculate the square root of the MSE. This value is called RMSE. The implemented code for MSE looks like this:

def MSE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.square(true-pred))

TRUE = np.array([10, 20, 30, 40, 50])
PRED = np.array([30, 40, 50, 60, 70])

MSE(TRUE, PRED)

### 1.2.3 MAPE (Mean Absolute Percentage Error)

> $MAPE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} |\frac{y_i-\hat{y}_i}{y_i}|$


(Reference: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)

In order to calculate MAPE, first calculate the relative size of the error compared to the actual values by dividing the difference between each of the actual values and the predicted value by each actual value. Then, take the absolute value of the relative size of the error for each actual value and calculate the mean. Since the size of the error is expressed as a percentage value, it can be used to understand the performance of the model. Also, it is a suitable metric for evaluating a model when there is more than one target variable because the scale of the calculated errors across the target variables will be similar.

However, if there is an actual value of 0, MAPE will be undefined. In addition, even if the absolute values of the errors are same, more penalties are added to a predicted value that overestimates([Makridakis, 1993](https://doi.org/10.1016/0169-2070(93)90079-3)). Let's look at the example below. 

def MAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs((true-pred)/true))

TRUE_UNDER = np.array([10, 20, 30, 40, 50])
PRED_OVER = np.array([30, 40, 50, 60, 70])
TRUE_OVER = np.array([30, 40, 50, 60, 70])
PRED_UNDER = np.array([10, 20, 30, 40, 50])


print('Comparison between MAE, MAPE when average error is 20 depending on the relationship between actual and predicted value \n')

print('When actual value is smaller than predicted value (Overestimating)')
print('MAE:', MAE(TRUE_UNDER, PRED_OVER))
print('MAPE:', MAPE(TRUE_UNDER, PRED_OVER))


print('\nWhen actual value is bigger than predicted value (Underestimating)')
print('MAE:', MAE(TRUE_OVER, PRED_UNDER))
print('MAPE:', MAPE(TRUE_OVER, PRED_UNDER))


MAPE divides the error by the actual value $y$ to convert it to a percentage. Therefore, the calculated value is dependent on $y$. Even if the numerators are the same, smaller denominators will increase the overall error.

We can observe this phenomenon by observing the two examples above where (`TRUE_UNDER`, `PRED_OVER`) predicts values that are more than the actual values by 20 and (`TRUE_OVER`, `PRED_UNDER`) predicts values that are less than the actual values by 20. On both examples, the MAE values are the same at 20. However, for the `TRUE_UNDER` case the MAPE value is calculated as 0.913 and `TRUE_OVER` case calculates the MAPE value as 0.437. 

### 1.2.4 SMAPE (Symmetric Mean Absolute Percentage Error)


> $SMAPE=\frac{100}{n}\displaystyle\sum_{i=1}^{n} \frac{|y_i-\hat{y}_i|}{|y_i| + |\hat{y}_i|}$


(Reference: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)


SMAPE has been created to deal with the limitations of MAPE for the examples above([Makridakis, 1993](https://doi.org/10.1016/0169-2070(93)90079-3)). Let's look at the example below.

def SMAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) #we won't include 100 in this code since it's a constant

print('Comparison between MAE, SMAPE when average error is 20 \n')

print('When actual value is smaller than predicted value (Overestimating)')
print('MAE:', MAE(TRUE_UNDER, PRED_OVER))
print('SMAPE:', SMAPE(TRUE_UNDER, PRED_OVER))


print('\nWhen actual value is bigger than predicted value (Underestimating)')
print('MAE:', MAE(TRUE_OVER, PRED_UNDER))
print('SMAPE:', SMAPE(TRUE_OVER, PRED_UNDER))


We can observe that MAPE produced different values of 0.91 and 0.43 respectively on the same example, but SMAPE yielded the same values of 0.29. However, SMAPE is dependent on $\hat{y}_i$ because the predicted value $\hat{y}_i$ is included in the denominator. When the predicted value is an underestimation, the denominator becomes smaller and the overall error increases. Let's look at the example below. 

TRUE2 = np.array([40, 50, 60, 70, 80])
PRED2_UNDER = np.array([20, 30, 40, 50, 60])
PRED2_OVER = np.array([60, 70, 80, 90, 100])

print('Comparison between MAE, SMAPE when average error is 20 \n')

print('When overestimating')
print('MAE:', MAE(TRUE2, PRED2_OVER))
print('SMAPE:', SMAPE(TRUE2, PRED2_OVER))

print('\nWhen underestimating')
print('MAE:', MAE(TRUE2, PRED2_UNDER))
print('SMAPE:', SMAPE(TRUE2, PRED2_UNDER))

`PRED2_UNDER` and `PRED2_OVER` both have an MAE of 20 compared with `TRUE2` , but SMAPE is calculated as 0.218 for `PRED2_UNDER` where underestimation occurred and 0.149 for `PRED2_OVER` where overestimation occurred.

### 1.2.5 RMSSE (Root Mean Squared Scaled Error)

> $RMSSE=\sqrt{\displaystyle\frac{\frac{1}{h}\sum_{i=n+1}^{n+h} (y_i-\hat{y}*i)^2}{\frac{1}{n-1}\sum*{i=2}^{n} (y_i-y_{i-1})^2}}$

We will proceed with the definition of the symbols used in the RMSSE formula. Each symbol has the following meaning.

> $y_i$: Actual value to be predicted
>
> $\hat{y}_i$: Value predicted by the model
>
> $n$: Size of the training dataset
>
> $h$: Size of the test dataset

RMSSE is a modified form of Mean Absolute Scaled Error ( [Hyndman, 2006](https://doi.org/10.1016/j.ijforecast.2006.03.001) ) and solves the problems of MAPE and SMAPE mentioned above. We have seen from above examples that MAPE and SMAPE result in an uneven overall error depending on the underestimation or overestimation of the model since they use the actual and predicted values of the test data to scale the MAE.

RMSSE avoids this problem by using the training data when scaling the MSE. It scales the error by dividing by the MSE that is calculated when applying naive forecasting on the training data, so the overall error is not affected by the underestimation or overestimation of the model. The naive forecast method is a method of forecasting from the most recent observation and is defined as follows:

> $\hat{y}*i = y*{i-1}$

This is a method of predicting the value at the time of $i$ as the actual value at the time of $i-1$. Since it is divided by the MSE value for the naive forecast method, if the RMSSE value is greater than 1, it means that the overall performance of the model is not better than the naive forecast method. On the other hand, if RMSSE is less than 1, it means that the performance of the model is better than the naive forecast method. Let's implement RMSSE with the code below. 

def RMSSE(true, pred, train): 
    '''
    true: np.array 
    pred: np.array
    train: np.array
    '''
    
    n = len(train)

    numerator = np.mean(np.sum(np.square(true - pred)))
    
    denominator = 1/(n-1)*np.sum(np.square((train[1:] - train[:-1])))
    
    msse = numerator/denominator
    
    return msse ** 0.5

TRAIN = np.array([10, 20, 30, 40, 50]) #create a random training dataset for calculating RMSSE

print(RMSSE(TRUE_UNDER, PRED_OVER, TRAIN))
print(RMSSE(TRUE_OVER, PRED_UNDER, TRAIN))
print(RMSSE(TRUE2, PRED2_OVER, TRAIN))
print(RMSSE(TRUE2, PRED2_UNDER, TRAIN))

On the examples where MAPE and SMAPE gave unequal penalties even if the MAE values were the same, RMSSE not only gave equal penalties, but also scaled the error.

So far, we have looked at the deep learning network structures and evaluation metrics that can be used for evaluating time series forecasting models. In the next chapter, we will perform exploratory data analysis on the COVID-19 data which will be used for building a forecasting model. 