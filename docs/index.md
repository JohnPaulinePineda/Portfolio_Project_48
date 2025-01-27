***
# Supervised Learning : Exploring Activation Functions And Backpropagation Gradient Updates for Neural Network Classification 

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *March 12, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Neural Network Classification Gradient and Weight Updates](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 Sigmoid Activation Function](#1.6.2)
        * [1.6.3 Rectified Linear Unit Activation Function](#1.6.3)
        * [1.6.4 Leaky Rectified Linear Unit Activation Function](#1.6.4)
        * [1.6.5 Exponential Linear Unit Activation Function](#1.6.5)
        * [1.6.6 Scaled Exponential Linear Unit Activation Function](#1.6.6)
        * [1.6.7 Randomized Leaky Rectified Linear Unit Activation Function](#1.6.7)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project manually implements the **Sigmoid**, **Rectified Linear Unit**, **Leaky Rectified Linear Unit**, **Exponential Linear Unit**, **Scaled Exponential Linear Unit** and **Randomized Leaky Rectified Linear Unit** activation functions using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> with fixed values applied for the learning rate and iteration count parameters to optimally update the gradients and weights of an artificial neural network classification model. The gradient, weight, cost function and classification accuracy optimization profiles of the different activation settings were compared. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document.

[Artificial Neural Network](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of categorical response prediction, consists of interconnected nodes called neurons organized in layers. The model architecture involves an input layer which receives the input data, with each neuron representing a feature or attribute of the data; hidden layers which perform computations on the input data through weighted connections between neurons and apply activation functions to produce outputs; and the output layer which produces the final predictions equal to the number of classes, each representing the probability of the input belonging to a particular class, based on the computations performed in the hidden layers. Neurons within adjacent layers are connected by weighted connections. Each connection has an associated weight that determines the strength of influence one neuron has on another. These weights are adjusted during the training process to enable the network to learn from the input data and make accurate predictions. Activation functions introduce non-linearities into the network, allowing it to learn complex relationships between inputs and outputs. The training process involves presenting input data along with corresponding target outputs to the network and adjusting the weights to minimize the difference between the predicted outputs and the actual targets which is typically performed through optimization algorithms such as gradient descent and backpropagation. The training process iteratively updates the weights until the model's predictions closely match the target outputs.

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Activation Functions](https://link.springer.com/book/10.1007/978-0-387-84858-7) play a crucial role in neural networks by introducing non-linearity into the network, enabling the model to learn complex patterns and relationships within the data. In the context of a neural network classification model, activation functions are applied to the output of each neuron in the hidden layers to introduce non-linear mappings between the input and output, allowing the network to approximate complex functions and make non-linear decisions. Activation functions are significant during model development by introducing non-linearity (without activation functions, the neural network would simply be a series of linear transformations, no matter how many layers it has. Activation functions introduce non-linearities to the model, enabling it to learn and represent complex patterns and relationships in the data); propagating back gradients (activation functions help in the backpropagation algorithm by providing gradients that indicate the direction and magnitude of adjustments to the weights during training. These gradients are necessary for optimizing the network's parameters through techniques like gradient descent}; and normalizing outputs (activation functions also help in normalizing the output of each neuron, ensuring that it falls within a specific range. This normalization prevents the activation values from becoming too large or too small, which can lead to numerical instability or saturation of gradients during training). The choice of activation function can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.


## 1.1. Data Background <a class="anchor" id="1.1"></a>

Datasets used for the analysis were separately gathered and consolidated from various sources including: 
1. Cancer Rates from [World Population Review](https://worldpopulationreview.com/country-rankings/cancer-rates-by-country)
2. Social Protection and Labor Indicator from [World Bank](https://data.worldbank.org/topic/social-protection-and-labor?view=chart)
3. Education Indicator from [World Bank](https://data.worldbank.org/topic/education?view=chart)
4. Economy and Growth Indicator from [World Bank](https://data.worldbank.org/topic/economy-and-growth?view=chart)
5. Environment Indicator from [World Bank](https://data.worldbank.org/topic/environment?view=chart)
6. Climate Change Indicator from [World Bank](https://data.worldbank.org/topic/climate-change?view=chart)
7. Agricultural and Rural Development Indicator from [World Bank](https://data.worldbank.org/topic/agriculture-and-rural-development?view=chart)
8. Social Development Indicator from [World Bank](https://data.worldbank.org/topic/social-development?view=chart)
9. Health Indicator from [World Bank](https://data.worldbank.org/topic/health?view=chart)
10. Science and Technology Indicator from [World Bank](https://data.worldbank.org/topic/science-and-technology?view=chart)
11. Urban Development Indicator from [World Bank](https://data.worldbank.org/topic/urban-development?view=chart)
12. Human Development Indices from [Human Development Reports](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI)
13. Environmental Performance Indices from [Yale Center for Environmental Law and Policy](https://epi.yale.edu/epi-results/2022/component/epi)

This study hypothesized that various global development indicators and indices influence cancer rates across countries.

The target variable for the study is:
* <span style="color: #FF0000">CANRAT</span> - Dichotomized category based on age-standardized cancer rates, per 100K population (2022)

The predictor variables for the study are:
* <span style="color: #FF0000">GDPPER</span> - GDP per person employed, current US Dollars (2020)
* <span style="color: #FF0000">URBPOP</span> - Urban population, % of total population (2020)
* <span style="color: #FF0000">PATRES</span> - Patent applications by residents, total count (2020)
* <span style="color: #FF0000">RNDGDP</span> - Research and development expenditure, % of GDP (2020)
* <span style="color: #FF0000">POPGRO</span> - Population growth, annual % (2020)
* <span style="color: #FF0000">LIFEXP</span> - Life expectancy at birth, total in years (2020)
* <span style="color: #FF0000">TUBINC</span> - Incidence of tuberculosis, per 100K population (2020)
* <span style="color: #FF0000">DTHCMD</span> - Cause of death by communicable diseases and maternal, prenatal and nutrition conditions,  % of total (2019)
* <span style="color: #FF0000">AGRLND</span> - Agricultural land,  % of land area (2020)
* <span style="color: #FF0000">GHGEMI</span> - Total greenhouse gas emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">RELOUT</span> - Renewable electricity output, % of total electricity output (2015)
* <span style="color: #FF0000">METEMI</span> - Methane emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">FORARE</span> - Forest area, % of land area (2020)
* <span style="color: #FF0000">CO2EMI</span> - CO2 emissions, metric tons per capita (2020)
* <span style="color: #FF0000">PM2EXP</span> - PM2.5 air pollution, population exposed to levels exceeding WHO guideline value,  % of total (2017)
* <span style="color: #FF0000">POPDEN</span> - Population density, people per sq. km of land area (2020)
* <span style="color: #FF0000">GDPCAP</span> - GDP per capita, current US Dollars (2020)
* <span style="color: #FF0000">ENRTER</span> - Tertiary school enrollment, % gross (2020)
* <span style="color: #FF0000">HDICAT</span> - Human development index, ordered category (2020)
* <span style="color: #FF0000">EPISCO</span> - Environment performance index , score (2022)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **177 rows** (observations)
    * **22 columns** (variables)
        * **1/22 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/22 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **19/22 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">PATRES</span>
             * <span style="color: #FF0000">RNDGDP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">RELOUT</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">ENRTER</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/22 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from scipy import stats
```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\original"
```


```python
##################################
# Loading the dataset
# from the DATASETS_ORIGINAL_PATH
##################################
cancer_rate = pd.read_csv(os.path.join("..", DATASETS_ORIGINAL_PATH, "CategoricalCancerRates.csv"))
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate.dtypes)
```

    Column Names and Data Types:
    


    COUNTRY     object
    CANRAT      object
    GDPPER     float64
    URBPOP     float64
    PATRES     float64
    RNDGDP     float64
    POPGRO     float64
    LIFEXP     float64
    TUBINC     float64
    DTHCMD     float64
    AGRLND     float64
    GHGEMI     float64
    RELOUT     float64
    METEMI     float64
    FORARE     float64
    CO2EMI     float64
    PM2EXP     float64
    POPDEN     float64
    ENRTER     float64
    GDPCAP     float64
    HDICAT      object
    EPISCO     float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COUNTRY</th>
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>PATRES</th>
      <th>RNDGDP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>...</th>
      <th>RELOUT</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>ENRTER</th>
      <th>GDPCAP</th>
      <th>HDICAT</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>High</td>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>2368.0</td>
      <td>NaN</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>...</td>
      <td>13.637841</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>110.139221</td>
      <td>51722.06900</td>
      <td>VH</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>High</td>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>348.0</td>
      <td>NaN</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>...</td>
      <td>80.081439</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>NaN</td>
      <td>19.331586</td>
      <td>75.734833</td>
      <td>41760.59478</td>
      <td>VH</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>High</td>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>75.0</td>
      <td>1.23244</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>...</td>
      <td>27.965408</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>74.680313</td>
      <td>85420.19086</td>
      <td>VH</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>High</td>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>269586.0</td>
      <td>3.42287</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>...</td>
      <td>13.228593</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>87.567657</td>
      <td>63528.63430</td>
      <td>VH</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>High</td>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>1261.0</td>
      <td>2.96873</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>...</td>
      <td>65.505925</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>82.664330</td>
      <td>60915.42440</td>
      <td>VH</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
##################################
# Setting the levels of the categorical variables
##################################
cancer_rate['CANRAT'] = cancer_rate['CANRAT'].astype('category')
cancer_rate['CANRAT'] = cancer_rate['CANRAT'].cat.set_categories(['Low', 'High'], ordered=True)
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].astype('category')
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].cat.set_categories(['L', 'M', 'H', 'VH'], ordered=True)
```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(cancer_rate.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GDPPER</th>
      <td>165.0</td>
      <td>45284.424283</td>
      <td>3.941794e+04</td>
      <td>1718.804896</td>
      <td>13545.254510</td>
      <td>34024.900890</td>
      <td>66778.416050</td>
      <td>2.346469e+05</td>
    </tr>
    <tr>
      <th>URBPOP</th>
      <td>174.0</td>
      <td>59.788121</td>
      <td>2.280640e+01</td>
      <td>13.345000</td>
      <td>42.432750</td>
      <td>61.701500</td>
      <td>79.186500</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>PATRES</th>
      <td>108.0</td>
      <td>20607.388889</td>
      <td>1.340683e+05</td>
      <td>1.000000</td>
      <td>35.250000</td>
      <td>244.500000</td>
      <td>1297.750000</td>
      <td>1.344817e+06</td>
    </tr>
    <tr>
      <th>RNDGDP</th>
      <td>74.0</td>
      <td>1.197474</td>
      <td>1.189956e+00</td>
      <td>0.039770</td>
      <td>0.256372</td>
      <td>0.873660</td>
      <td>1.608842</td>
      <td>5.354510e+00</td>
    </tr>
    <tr>
      <th>POPGRO</th>
      <td>174.0</td>
      <td>1.127028</td>
      <td>1.197718e+00</td>
      <td>-2.079337</td>
      <td>0.236900</td>
      <td>1.179959</td>
      <td>2.031154</td>
      <td>3.727101e+00</td>
    </tr>
    <tr>
      <th>LIFEXP</th>
      <td>174.0</td>
      <td>71.746113</td>
      <td>7.606209e+00</td>
      <td>52.777000</td>
      <td>65.907500</td>
      <td>72.464610</td>
      <td>77.523500</td>
      <td>8.456000e+01</td>
    </tr>
    <tr>
      <th>TUBINC</th>
      <td>174.0</td>
      <td>105.005862</td>
      <td>1.367229e+02</td>
      <td>0.770000</td>
      <td>12.000000</td>
      <td>44.500000</td>
      <td>147.750000</td>
      <td>5.920000e+02</td>
    </tr>
    <tr>
      <th>DTHCMD</th>
      <td>170.0</td>
      <td>21.260521</td>
      <td>1.927333e+01</td>
      <td>1.283611</td>
      <td>6.078009</td>
      <td>12.456279</td>
      <td>36.980457</td>
      <td>6.520789e+01</td>
    </tr>
    <tr>
      <th>AGRLND</th>
      <td>174.0</td>
      <td>38.793456</td>
      <td>2.171551e+01</td>
      <td>0.512821</td>
      <td>20.130276</td>
      <td>40.386649</td>
      <td>54.013754</td>
      <td>8.084112e+01</td>
    </tr>
    <tr>
      <th>GHGEMI</th>
      <td>170.0</td>
      <td>259582.709895</td>
      <td>1.118550e+06</td>
      <td>179.725150</td>
      <td>12527.487367</td>
      <td>41009.275980</td>
      <td>116482.578575</td>
      <td>1.294287e+07</td>
    </tr>
    <tr>
      <th>RELOUT</th>
      <td>153.0</td>
      <td>39.760036</td>
      <td>3.191492e+01</td>
      <td>0.000296</td>
      <td>10.582691</td>
      <td>32.381668</td>
      <td>63.011450</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>METEMI</th>
      <td>170.0</td>
      <td>47876.133575</td>
      <td>1.346611e+05</td>
      <td>11.596147</td>
      <td>3662.884908</td>
      <td>11118.976025</td>
      <td>32368.909040</td>
      <td>1.186285e+06</td>
    </tr>
    <tr>
      <th>FORARE</th>
      <td>173.0</td>
      <td>32.218177</td>
      <td>2.312001e+01</td>
      <td>0.008078</td>
      <td>11.604388</td>
      <td>31.509048</td>
      <td>49.071780</td>
      <td>9.741212e+01</td>
    </tr>
    <tr>
      <th>CO2EMI</th>
      <td>170.0</td>
      <td>3.751097</td>
      <td>4.606479e+00</td>
      <td>0.032585</td>
      <td>0.631924</td>
      <td>2.298368</td>
      <td>4.823496</td>
      <td>3.172684e+01</td>
    </tr>
    <tr>
      <th>PM2EXP</th>
      <td>167.0</td>
      <td>91.940595</td>
      <td>2.206003e+01</td>
      <td>0.274092</td>
      <td>99.627134</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>POPDEN</th>
      <td>174.0</td>
      <td>200.886765</td>
      <td>6.453834e+02</td>
      <td>2.115134</td>
      <td>27.454539</td>
      <td>77.983133</td>
      <td>153.993650</td>
      <td>7.918951e+03</td>
    </tr>
    <tr>
      <th>ENRTER</th>
      <td>116.0</td>
      <td>49.994997</td>
      <td>2.970619e+01</td>
      <td>2.432581</td>
      <td>22.107195</td>
      <td>53.392460</td>
      <td>71.057467</td>
      <td>1.433107e+02</td>
    </tr>
    <tr>
      <th>GDPCAP</th>
      <td>170.0</td>
      <td>13992.095610</td>
      <td>1.957954e+04</td>
      <td>216.827417</td>
      <td>1870.503029</td>
      <td>5348.192875</td>
      <td>17421.116227</td>
      <td>1.173705e+05</td>
    </tr>
    <tr>
      <th>EPISCO</th>
      <td>165.0</td>
      <td>42.946667</td>
      <td>1.249086e+01</td>
      <td>18.900000</td>
      <td>33.000000</td>
      <td>40.900000</td>
      <td>50.500000</td>
      <td>7.790000e+01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variable
##################################
print('Object Variable Summary:')
display(cancer_rate.describe(include='object').transpose())
```

    Object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COUNTRY</th>
      <td>177</td>
      <td>177</td>
      <td>Australia</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variables
##################################
print('Categorical Variable Summary:')
display(cancer_rate.describe(include='category').transpose())
```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT</th>
      <td>177</td>
      <td>2</td>
      <td>Low</td>
      <td>132</td>
    </tr>
    <tr>
      <th>HDICAT</th>
      <td>167</td>
      <td>4</td>
      <td>VH</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed.
2. Missing data noted for 20 variables with Null.Count>0 and Fill.Rate<1.0.
    * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
    * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
    * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
    * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">EPISCO</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">HDICAT</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">DTHCMD</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">METEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">CO2EMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GDPCAP</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GHGEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 4, Fill.Rate = 0.977
    * <span style="color: #FF0000">TUBINC</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">AGRLND</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPGRO</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPDEN</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">URBPOP</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">LIFEXP</span>: Null.Count = 3, Fill.Rate = 0.983
3. 120 observations noted with at least 1 missing data. From this number, 14 observations reported high Missing.Rate>0.2.
    * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
    * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227
4. Low variance observed for 1 variable with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">PM2EXP</span>: First.Second.Mode.Ratio = 53.000
5. No low variance observed for any variable with Unique.Count.Ratio>10.
6. High skewness observed for 5 variables with Skewness>3 or Skewness<(-3).
    * <span style="color: #FF0000">POPDEN</span>: Skewness = +10.267
    * <span style="color: #FF0000">GHGEMI</span>: Skewness = +9.496
    * <span style="color: #FF0000">PATRES</span>: Skewness = +9.284
    * <span style="color: #FF0000">METEMI</span>: Skewness = +5.801
    * <span style="color: #FF0000">PM2EXP</span>: Skewness = -3.141


```python
##################################
# Counting the number of duplicated rows
##################################
cancer_rate.duplicated().sum()
```




    np.int64(0)




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate)] * len(cancer_rate.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>category</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.610169</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.418079</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.864407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.977401</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.655367</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])
```




    20




```python
##################################
# Identifying the columns
# with Fill.Rate < 1.00
##################################
display(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)].sort_values(by=['Fill.Rate'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.418079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.610169</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.655367</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.864407</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.977401</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Fill.Rate < 0.90
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<0.90)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = cancer_rate["COUNTRY"].values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(cancer_rate.columns)] * len(cancer_rate))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(cancer_rate.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Identifying the rows
# with missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>22</td>
      <td>1</td>
      <td>0.045455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>172</th>
      <td>Congo Republic</td>
      <td>22</td>
      <td>3</td>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Bhutan</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>174</th>
      <td>Nepal</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Gambia</td>
      <td>22</td>
      <td>4</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Niger</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.00
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.00)])
```




    120




```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.20
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)])
```




    14




```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
row_high_missing_rate = all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)]
```


```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
display(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)].sort_values(by=['Missing.Rate'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>Guadeloupe</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Martinique</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>56</th>
      <td>French Guiana</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>13</th>
      <td>New Caledonia</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>French Polynesia</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Guam</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Puerto Rico</td>
      <td>22</td>
      <td>9</td>
      <td>0.409091</td>
    </tr>
    <tr>
      <th>85</th>
      <td>North Korea</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>168</th>
      <td>South Sudan</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Somalia</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Libya</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Venezuela</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Eritrea</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Yemen</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
cancer_rate_numeric = cancer_rate.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = cancer_rate_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = cancer_rate_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = cancer_rate_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = cancer_rate_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = cancer_rate_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[0] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[1] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = cancer_rate_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(cancer_rate_numeric)] * len(cancer_rate_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = cancer_rate_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDPPER</td>
      <td>1718.804896</td>
      <td>45284.424283</td>
      <td>34024.900890</td>
      <td>2.346469e+05</td>
      <td>98380.636010</td>
      <td>77541.764380</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>165</td>
      <td>177</td>
      <td>0.932203</td>
      <td>1.517574</td>
      <td>3.471992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>URBPOP</td>
      <td>13.345000</td>
      <td>59.788121</td>
      <td>61.701500</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>86.699000</td>
      <td>2</td>
      <td>1</td>
      <td>2.000000</td>
      <td>173</td>
      <td>177</td>
      <td>0.977401</td>
      <td>-0.210702</td>
      <td>-0.962847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PATRES</td>
      <td>1.000000</td>
      <td>20607.388889</td>
      <td>244.500000</td>
      <td>1.344817e+06</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>97</td>
      <td>177</td>
      <td>0.548023</td>
      <td>9.284436</td>
      <td>91.187178</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RNDGDP</td>
      <td>0.039770</td>
      <td>1.197474</td>
      <td>0.873660</td>
      <td>5.354510e+00</td>
      <td>1.232440</td>
      <td>3.422870</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>74</td>
      <td>177</td>
      <td>0.418079</td>
      <td>1.396742</td>
      <td>1.695957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POPGRO</td>
      <td>-2.079337</td>
      <td>1.127028</td>
      <td>1.179959</td>
      <td>3.727101e+00</td>
      <td>1.235701</td>
      <td>2.204789</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>-0.195161</td>
      <td>-0.423580</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LIFEXP</td>
      <td>52.777000</td>
      <td>71.746113</td>
      <td>72.464610</td>
      <td>8.456000e+01</td>
      <td>83.200000</td>
      <td>82.256098</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>-0.357965</td>
      <td>-0.649601</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TUBINC</td>
      <td>0.770000</td>
      <td>105.005862</td>
      <td>44.500000</td>
      <td>5.920000e+02</td>
      <td>12.000000</td>
      <td>4.100000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>131</td>
      <td>177</td>
      <td>0.740113</td>
      <td>1.746333</td>
      <td>2.429368</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DTHCMD</td>
      <td>1.283611</td>
      <td>21.260521</td>
      <td>12.456279</td>
      <td>6.520789e+01</td>
      <td>4.941054</td>
      <td>4.354730</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>0.900509</td>
      <td>-0.691541</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AGRLND</td>
      <td>0.512821</td>
      <td>38.793456</td>
      <td>40.386649</td>
      <td>8.084112e+01</td>
      <td>46.252480</td>
      <td>38.562911</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>0.074000</td>
      <td>-0.926249</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>179.725150</td>
      <td>259582.709895</td>
      <td>41009.275980</td>
      <td>1.294287e+07</td>
      <td>571903.119900</td>
      <td>80158.025830</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>9.496120</td>
      <td>101.637308</td>
    </tr>
    <tr>
      <th>10</th>
      <td>RELOUT</td>
      <td>0.000296</td>
      <td>39.760036</td>
      <td>32.381668</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>80.081439</td>
      <td>3</td>
      <td>1</td>
      <td>3.000000</td>
      <td>151</td>
      <td>177</td>
      <td>0.853107</td>
      <td>0.501088</td>
      <td>-0.981774</td>
    </tr>
    <tr>
      <th>11</th>
      <td>METEMI</td>
      <td>11.596147</td>
      <td>47876.133575</td>
      <td>11118.976025</td>
      <td>1.186285e+06</td>
      <td>131484.763200</td>
      <td>32241.937000</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>5.801014</td>
      <td>38.661386</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FORARE</td>
      <td>0.008078</td>
      <td>32.218177</td>
      <td>31.509048</td>
      <td>9.741212e+01</td>
      <td>17.421315</td>
      <td>37.570126</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>173</td>
      <td>177</td>
      <td>0.977401</td>
      <td>0.519277</td>
      <td>-0.322589</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CO2EMI</td>
      <td>0.032585</td>
      <td>3.751097</td>
      <td>2.298368</td>
      <td>3.172684e+01</td>
      <td>14.772658</td>
      <td>6.160799</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>2.721552</td>
      <td>10.311574</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>106</td>
      <td>2</td>
      <td>53.000000</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
    <tr>
      <th>15</th>
      <td>POPDEN</td>
      <td>2.115134</td>
      <td>200.886765</td>
      <td>77.983133</td>
      <td>7.918951e+03</td>
      <td>3.335312</td>
      <td>19.331586</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>10.267750</td>
      <td>119.995256</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ENRTER</td>
      <td>2.432581</td>
      <td>49.994997</td>
      <td>53.392460</td>
      <td>1.433107e+02</td>
      <td>110.139221</td>
      <td>75.734833</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>116</td>
      <td>177</td>
      <td>0.655367</td>
      <td>0.275863</td>
      <td>-0.392895</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GDPCAP</td>
      <td>216.827417</td>
      <td>13992.095610</td>
      <td>5348.192875</td>
      <td>1.173705e+05</td>
      <td>51722.069000</td>
      <td>41760.594780</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>2.258568</td>
      <td>5.938690</td>
    </tr>
    <tr>
      <th>18</th>
      <td>EPISCO</td>
      <td>18.900000</td>
      <td>42.946667</td>
      <td>40.900000</td>
      <td>7.790000e+01</td>
      <td>29.600000</td>
      <td>43.600000</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
      <td>137</td>
      <td>177</td>
      <td>0.774011</td>
      <td>0.641799</td>
      <td>0.035208</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    1




```python
##################################
# Identifying the numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>106</td>
      <td>2</td>
      <td>53.0</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])
```




    5




```python
##################################
# Identifying the numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))].sort_values(by=['Skewness'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>POPDEN</td>
      <td>2.115134</td>
      <td>200.886765</td>
      <td>77.983133</td>
      <td>7.918951e+03</td>
      <td>3.335312</td>
      <td>19.331586</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>10.267750</td>
      <td>119.995256</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>179.725150</td>
      <td>259582.709895</td>
      <td>41009.275980</td>
      <td>1.294287e+07</td>
      <td>571903.119900</td>
      <td>80158.025830</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>9.496120</td>
      <td>101.637308</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PATRES</td>
      <td>1.000000</td>
      <td>20607.388889</td>
      <td>244.500000</td>
      <td>1.344817e+06</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>97</td>
      <td>177</td>
      <td>0.548023</td>
      <td>9.284436</td>
      <td>91.187178</td>
    </tr>
    <tr>
      <th>11</th>
      <td>METEMI</td>
      <td>11.596147</td>
      <td>47876.133575</td>
      <td>11118.976025</td>
      <td>1.186285e+06</td>
      <td>131484.763200</td>
      <td>32241.937000</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>5.801014</td>
      <td>38.661386</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>106</td>
      <td>2</td>
      <td>53.000000</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object column only
##################################
cancer_rate_object = cancer_rate.select_dtypes(include='object')
```


```python
##################################
# Gathering the variable names for the object column
##################################
object_variable_name_list = cancer_rate_object.columns
```


```python
##################################
# Gathering the first mode values for the object column
##################################
object_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_object]
```


```python
##################################
# Gathering the second mode values for each object column
##################################
object_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of first mode values for each object column
##################################
object_first_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of second mode values for each object column
##################################
object_second_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object column
##################################
object_first_second_mode_ratio_list = map(truediv, object_first_mode_count_list, object_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object column
##################################
object_unique_count_list = cancer_rate_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object column
##################################
object_row_count_list = list([len(cancer_rate_object)] * len(cancer_rate_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
object_unique_count_ratio_list = map(truediv, object_unique_count_list, object_row_count_list)
```


```python
object_column_quality_summary = pd.DataFrame(zip(object_variable_name_list,
                                                 object_first_mode_list,
                                                 object_second_mode_list,
                                                 object_first_mode_count_list,
                                                 object_second_mode_count_list,
                                                 object_first_second_mode_ratio_list,
                                                 object_unique_count_list,
                                                 object_row_count_list,
                                                 object_unique_count_ratio_list), 
                                        columns=['Object.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(object_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Object.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>Australia</td>
      <td>New Zealand</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>177</td>
      <td>177</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of object columns
# with Unique.Count.Ratio > 10.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Formulating the dataset
# with categorical columns only
##################################
cancer_rate_categorical = cancer_rate.select_dtypes(include='category')
```


```python
##################################
# Gathering the variable names for the categorical column
##################################
categorical_variable_name_list = cancer_rate_categorical.columns
```


```python
##################################
# Gathering the first mode values for each categorical column
##################################
categorical_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the second mode values for each categorical column
##################################
categorical_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of first mode values for each categorical column
##################################
categorical_first_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of second mode values for each categorical column
##################################
categorical_second_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the first mode to second mode ratio for each categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each categorical column
##################################
categorical_unique_count_list = cancer_rate_categorical.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each categorical column
##################################
categorical_row_count_list = list([len(cancer_rate_categorical)] * len(cancer_rate_categorical.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)
```


```python
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                    categorical_first_mode_list,
                                                    categorical_second_mode_list,
                                                    categorical_first_mode_count_list,
                                                    categorical_second_mode_count_list,
                                                    categorical_first_second_mode_ratio_list,
                                                    categorical_unique_count_list,
                                                    categorical_row_count_list,
                                                    categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>Low</td>
      <td>High</td>
      <td>132</td>
      <td>45</td>
      <td>2.933333</td>
      <td>2</td>
      <td>177</td>
      <td>0.011299</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HDICAT</td>
      <td>VH</td>
      <td>H</td>
      <td>59</td>
      <td>39</td>
      <td>1.512821</td>
      <td>4</td>
      <td>177</td>
      <td>0.022599</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>


### 1.4.1 Data Cleaning <a class="anchor" id="1.4.1"></a>

1. Subsets of rows and columns with high rates of missing data were removed from the dataset:
    * 4 variables with Fill.Rate<0.9 were excluded for subsequent analysis.
        * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
        * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
        * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
        * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * 14 rows with Missing.Rate>0.2 were exluded for subsequent analysis.
        * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
        * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227  
2. No variables were removed due to zero or near-zero variance.
3. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **15/18 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Performing a general exploration of the original dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Filtering out the rows with
# with Missing.Rate > 0.20
##################################
cancer_rate_filtered_row = cancer_rate.drop(cancer_rate[cancer_rate.COUNTRY.isin(row_high_missing_rate['Row.Name'].values.tolist())].index)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_filtered_row.shape)
```

    Dataset Dimensions: 
    


    (163, 22)



```python
##################################
# Filtering out the columns with
# with Fill.Rate < 0.90
##################################
cancer_rate_filtered_row_column = cancer_rate_filtered_row.drop(column_low_fill_rate['Column.Name'].values.tolist(), axis=1)
```


```python
##################################
# Formulating a new dataset object
# for the cleaned data
##################################
cancer_rate_cleaned = cancer_rate_filtered_row_column
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_cleaned.shape)
```

    Dataset Dimensions: 
    


    (163, 18)


### 1.4.2 Missing Data Imputation <a class="anchor" id="1.4.2"></a>

[Iterative Imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) is based on the [Multivariate Imputation by Chained Equations](https://journals.sagepub.com/doi/10.1177/0962280206074463) (MICE) algorithm - an imputation method based on fully conditional specification, where each incomplete variable is imputed by a separate model. As a sequential regression imputation technique, the algorithm imputes an incomplete column (target column) by generating plausible synthetic values given other columns in the data. Each incomplete column must act as a target column, and has its own specific set of predictors. For predictors that are incomplete themselves, the most recently generated imputations are used to complete the predictors prior to prior to imputation of the target columns.

[Linear Regression](https://link.springer.com/book/10.1007/978-1-4757-3462-1) explores the linear relationship between a scalar response and one or more covariates by having the conditional mean of the dependent variable be an affine function of the independent variables. The relationship is modeled through a disturbance term which represents an unobserved random variable that adds noise. The algorithm is typically formulated from the data using the least squares method which seeks to estimate the coefficients by minimizing the squared residual function. The linear equation assigns one scale factor represented by a coefficient to each covariate and an additional coefficient called the intercept or the bias coefficient which gives the line an additional degree of freedom allowing to move up and down a two-dimensional plot.

1. Missing data for numeric variables were imputed using the iterative imputer algorithm with a  linear regression estimator.
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 1
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 1
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 5
2. Missing data for categorical variables were imputed using the most frequent value.
    * <span style="color: #FF0000">HDICAP</span>: Null.Count = 1


```python
##################################
# Formulating the summary
# for all cleaned columns
##################################
cleaned_column_quality_summary = pd.DataFrame(zip(list(cancer_rate_cleaned.columns),
                                                  list(cancer_rate_cleaned.dtypes),
                                                  list([len(cancer_rate_cleaned)] * len(cancer_rate_cleaned.columns)),
                                                  list(cancer_rate_cleaned.count()),
                                                  list(cancer_rate_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(cleaned_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>158</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='object')
```


```python
##################################
# Formulating the cleaned dataset
# with numeric columns only
##################################
cancer_rate_cleaned_numeric = cancer_rate_cleaned.select_dtypes(include='number')
```


```python
##################################
# Taking a snapshot of the cleaned dataset
##################################
cancer_rate_cleaned_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>46.252480</td>
      <td>5.719031e+05</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>51722.06900</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>38.562911</td>
      <td>8.015803e+04</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>NaN</td>
      <td>19.331586</td>
      <td>41760.59478</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>65.495718</td>
      <td>5.949773e+04</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>85420.19086</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>44.363367</td>
      <td>5.505181e+06</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>63528.63430</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>65.499675</td>
      <td>4.113555e+04</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>60915.42440</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the estimator to be used
# at each step of the round-robin imputation
##################################
lr = LinearRegression()
```


```python
##################################
# Defining the parameter of the
# iterative imputer which will estimate 
# the columns with missing values
# as a function of the other columns
# in a round-robin fashion
##################################
iterative_imputer = IterativeImputer(
    estimator = lr,
    max_iter = 10,
    tol = 1e-10,
    imputation_order = 'ascending',
    random_state=88888888
)
```


```python
##################################
# Implementing the iterative imputer 
##################################
cancer_rate_imputed_numeric_array = iterative_imputer.fit_transform(cancer_rate_cleaned_numeric)
```


```python
##################################
# Transforming the imputed data
# from an array to a dataframe
##################################
cancer_rate_imputed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array, 
                                           columns = cancer_rate_cleaned_numeric.columns)
```


```python
##################################
# Taking a snapshot of the imputed dataset
##################################
cancer_rate_imputed_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>46.252480</td>
      <td>5.719031e+05</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>51722.06900</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>38.562911</td>
      <td>8.015803e+04</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>65.867296</td>
      <td>19.331586</td>
      <td>41760.59478</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>65.495718</td>
      <td>5.949773e+04</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>85420.19086</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>44.363367</td>
      <td>5.505181e+06</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>63528.63430</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>65.499675</td>
      <td>4.113555e+04</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>60915.42440</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='category')
```


```python
##################################
# Imputing the missing data
# for categorical columns with
# the most frequent category
##################################
cancer_rate_cleaned_categorical['HDICAT'] = cancer_rate_cleaned_categorical['HDICAT'].fillna(cancer_rate_cleaned_categorical['HDICAT'].mode()[0])
cancer_rate_imputed_categorical = cancer_rate_cleaned_categorical.reset_index(drop=True)
```


```python
##################################
# Formulating the imputed dataset
##################################
cancer_rate_imputed = pd.concat([cancer_rate_imputed_numeric,cancer_rate_imputed_categorical], axis=1, join='inner')  
```


```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate_imputed.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate_imputed.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate_imputed)] * len(cancer_rate_imputed.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate_imputed.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate_imputed.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all imputed columns
##################################
imputed_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                                  data_type_list,
                                                  row_count_list,
                                                  non_null_count_list,
                                                  null_count_list,
                                                  fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(imputed_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CANRAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.3 Outlier Detection <a class="anchor" id="1.4.3"></a>

1. High number of outliers observed for 5 numeric variables with Outlier.Ratio>0.10 and marginal to high Skewness.
    * <span style="color: #FF0000">PM2EXP</span>: Outlier.Count = 37, Outlier.Ratio = 0.226, Skewness=-3.061
    * <span style="color: #FF0000">GHGEMI</span>: Outlier.Count = 27, Outlier.Ratio = 0.165, Skewness=+9.299
    * <span style="color: #FF0000">GDPCAP</span>: Outlier.Count = 22, Outlier.Ratio = 0.134, Skewness=+2.311
    * <span style="color: #FF0000">POPDEN</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+9.972
    * <span style="color: #FF0000">METEMI</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+5.688
2. Minimal number of outliers observed for 5 numeric variables with Outlier.Ratio<0.10 and normal Skewness.
    * <span style="color: #FF0000">TUBINC</span>: Outlier.Count = 12, Outlier.Ratio = 0.073, Skewness=+1.747
    * <span style="color: #FF0000">CO2EMI</span>: Outlier.Count = 11, Outlier.Ratio = 0.067, Skewness=+2.693
    * <span style="color: #FF0000">GDPPER</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+1.554
    * <span style="color: #FF0000">EPISCO</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+0.635
    * <span style="color: #FF0000">CANRAT</span>: Outlier.Count = 2, Outlier.Ratio = 0.012, Skewness=+0.910


```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
cancer_rate_imputed_numeric = cancer_rate_imputed.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = list(cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_imputed_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
cancer_rate_imputed_numeric_q1 = cancer_rate_imputed_numeric.quantile(0.25)
cancer_rate_imputed_numeric_q3 = cancer_rate_imputed_numeric.quantile(0.75)
cancer_rate_imputed_numeric_iqr = cancer_rate_imputed_numeric_q3 - cancer_rate_imputed_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((cancer_rate_imputed_numeric < (cancer_rate_imputed_numeric_q1 - 1.5 * cancer_rate_imputed_numeric_iqr)) | (cancer_rate_imputed_numeric > (cancer_rate_imputed_numeric_q3 + 1.5 * cancer_rate_imputed_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(cancer_rate_imputed_numeric)] * len(cancer_rate_imputed_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDPPER</td>
      <td>1.554457</td>
      <td>3</td>
      <td>163</td>
      <td>0.018405</td>
    </tr>
    <tr>
      <th>1</th>
      <td>URBPOP</td>
      <td>-0.212327</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POPGRO</td>
      <td>-0.181666</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LIFEXP</td>
      <td>-0.329704</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TUBINC</td>
      <td>1.747962</td>
      <td>12</td>
      <td>163</td>
      <td>0.073620</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DTHCMD</td>
      <td>0.930709</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AGRLND</td>
      <td>0.035315</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GHGEMI</td>
      <td>9.299960</td>
      <td>27</td>
      <td>163</td>
      <td>0.165644</td>
    </tr>
    <tr>
      <th>8</th>
      <td>METEMI</td>
      <td>5.688689</td>
      <td>20</td>
      <td>163</td>
      <td>0.122699</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FORARE</td>
      <td>0.563015</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CO2EMI</td>
      <td>2.693585</td>
      <td>11</td>
      <td>163</td>
      <td>0.067485</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PM2EXP</td>
      <td>-3.088403</td>
      <td>37</td>
      <td>163</td>
      <td>0.226994</td>
    </tr>
    <tr>
      <th>12</th>
      <td>POPDEN</td>
      <td>9.972806</td>
      <td>20</td>
      <td>163</td>
      <td>0.122699</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GDPCAP</td>
      <td>2.311079</td>
      <td>22</td>
      <td>163</td>
      <td>0.134969</td>
    </tr>
    <tr>
      <th>14</th>
      <td>EPISCO</td>
      <td>0.635994</td>
      <td>3</td>
      <td>163</td>
      <td>0.018405</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in cancer_rate_imputed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_imputed_numeric, x=column)
```


    
![png](output_122_0.png)
    



    
![png](output_122_1.png)
    



    
![png](output_122_2.png)
    



    
![png](output_122_3.png)
    



    
![png](output_122_4.png)
    



    
![png](output_122_5.png)
    



    
![png](output_122_6.png)
    



    
![png](output_122_7.png)
    



    
![png](output_122_8.png)
    



    
![png](output_122_9.png)
    



    
![png](output_122_10.png)
    



    
![png](output_122_11.png)
    



    
![png](output_122_12.png)
    



    
![png](output_122_13.png)
    



    
![png](output_122_14.png)
    


### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

[Pearson’s Correlation Coefficient](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) is a parametric measure of the linear correlation for a pair of features by calculating the ratio between their covariance and the product of their standard deviations. The presence of high absolute correlation values indicate the univariate association between the numeric predictors and the numeric response.

1. Majority of the numeric variables reported moderate to high correlation which were statistically significant.
2. Among pairwise combinations of numeric variables, high Pearson.Correlation.Coefficient values were noted for:
    * <span style="color: #FF0000">GDPPER</span> and <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient = +0.921
    * <span style="color: #FF0000">GHGEMI</span> and <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.905
3. Among the highly correlated pairs, variables with the lowest correlation against the target variable were removed.
    * <span style="color: #FF0000">GDPPER</span>: Pearson.Correlation.Coefficient = +0.690
    * <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.062
4. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **16 columns** (variables)
        * **1/16 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/16 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **13/16 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/16 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
def plot_correlation_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, 
                ax=ax,
                mask=mask,
                annot=True, 
                vmin=-1, 
                vmax=1, 
                center=0,
                cmap='coolwarm', 
                linewidths=1, 
                linecolor='gray', 
                cbar_kws={'orientation': 'horizontal'})  
```


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# among pairs of numeric columns
##################################
cancer_rate_imputed_numeric_correlation_pairs = {}
cancer_rate_imputed_numeric_columns = cancer_rate_imputed_numeric.columns.tolist()
for numeric_column_a, numeric_column_b in itertools.combinations(cancer_rate_imputed_numeric_columns, 2):
    cancer_rate_imputed_numeric_correlation_pairs[numeric_column_a + '_' + numeric_column_b] = stats.pearsonr(
        cancer_rate_imputed_numeric.loc[:, numeric_column_a], 
        cancer_rate_imputed_numeric.loc[:, numeric_column_b])
```


```python
##################################
# Formulating the pairwise correlation summary
# for all numeric columns
##################################
cancer_rate_imputed_numeric_summary = cancer_rate_imputed_numeric.from_dict(cancer_rate_imputed_numeric_correlation_pairs, orient='index')
cancer_rate_imputed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cancer_rate_imputed_numeric_summary.sort_values(by=['Pearson.Correlation.Coefficient'], ascending=False).head(20))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GDPPER_GDPCAP</th>
      <td>0.921010</td>
      <td>8.158179e-68</td>
    </tr>
    <tr>
      <th>GHGEMI_METEMI</th>
      <td>0.905121</td>
      <td>1.087643e-61</td>
    </tr>
    <tr>
      <th>POPGRO_DTHCMD</th>
      <td>0.759470</td>
      <td>7.124695e-32</td>
    </tr>
    <tr>
      <th>GDPPER_LIFEXP</th>
      <td>0.755787</td>
      <td>2.055178e-31</td>
    </tr>
    <tr>
      <th>GDPCAP_EPISCO</th>
      <td>0.696707</td>
      <td>5.312642e-25</td>
    </tr>
    <tr>
      <th>LIFEXP_GDPCAP</th>
      <td>0.683834</td>
      <td>8.321371e-24</td>
    </tr>
    <tr>
      <th>GDPPER_EPISCO</th>
      <td>0.680812</td>
      <td>1.555304e-23</td>
    </tr>
    <tr>
      <th>GDPPER_URBPOP</th>
      <td>0.666394</td>
      <td>2.781623e-22</td>
    </tr>
    <tr>
      <th>GDPPER_CO2EMI</th>
      <td>0.654958</td>
      <td>2.450029e-21</td>
    </tr>
    <tr>
      <th>TUBINC_DTHCMD</th>
      <td>0.643615</td>
      <td>1.936081e-20</td>
    </tr>
    <tr>
      <th>URBPOP_LIFEXP</th>
      <td>0.623997</td>
      <td>5.669778e-19</td>
    </tr>
    <tr>
      <th>LIFEXP_EPISCO</th>
      <td>0.620271</td>
      <td>1.048393e-18</td>
    </tr>
    <tr>
      <th>URBPOP_GDPCAP</th>
      <td>0.559181</td>
      <td>8.624533e-15</td>
    </tr>
    <tr>
      <th>CO2EMI_GDPCAP</th>
      <td>0.550221</td>
      <td>2.782997e-14</td>
    </tr>
    <tr>
      <th>URBPOP_CO2EMI</th>
      <td>0.550046</td>
      <td>2.846393e-14</td>
    </tr>
    <tr>
      <th>LIFEXP_CO2EMI</th>
      <td>0.531305</td>
      <td>2.951829e-13</td>
    </tr>
    <tr>
      <th>URBPOP_EPISCO</th>
      <td>0.510131</td>
      <td>3.507463e-12</td>
    </tr>
    <tr>
      <th>POPGRO_TUBINC</th>
      <td>0.442339</td>
      <td>3.384403e-09</td>
    </tr>
    <tr>
      <th>DTHCMD_PM2EXP</th>
      <td>0.283199</td>
      <td>2.491837e-04</td>
    </tr>
    <tr>
      <th>CO2EMI_EPISCO</th>
      <td>0.282734</td>
      <td>2.553620e-04</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
cancer_rate_imputed_numeric_correlation = cancer_rate_imputed_numeric.corr()
mask = np.triu(cancer_rate_imputed_numeric_correlation)
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)
plt.show()
```


    
![png](output_127_0.png)
    



```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
def correlation_significance(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
cancer_rate_imputed_numeric_correlation_p_values = correlation_significance(cancer_rate_imputed_numeric)                     
mask = np.invert(np.tril(cancer_rate_imputed_numeric_correlation_p_values<0.05)) 
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)  
```


    
![png](output_129_0.png)
    



```python
##################################
# Filtering out one among the 
# highly correlated variable pairs with
# lesser Pearson.Correlation.Coefficient
# when compared to the target variable
##################################
cancer_rate_imputed_numeric.drop(['GDPPER','METEMI'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_imputed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 13)


### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A Yeo-Johnson transformation was applied to all numeric variables to improve distributional shape.
2. Most variables achieved symmetrical distributions with minimal outliers after transformation.
3. One variable which remained skewed even after applying shape transformation was removed.
    * <span style="color: #FF0000">PM2EXP</span> 
4. The transformed dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=False)
cancer_rate_imputed_numeric_array = yeo_johnson_transformer.fit_transform(cancer_rate_imputed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
cancer_rate_transformed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array,
                                               columns=cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_transformed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_transformed_numeric, x=column)
```


    
![png](output_135_0.png)
    



    
![png](output_135_1.png)
    



    
![png](output_135_2.png)
    



    
![png](output_135_3.png)
    



    
![png](output_135_4.png)
    



    
![png](output_135_5.png)
    



    
![png](output_135_6.png)
    



    
![png](output_135_7.png)
    



    
![png](output_135_8.png)
    



    
![png](output_135_9.png)
    



    
![png](output_135_10.png)
    



    
![png](output_135_11.png)
    



    
![png](output_135_12.png)
    



```python
##################################
# Filtering out the column
# which remained skewed even
# after applying shape transformation
##################################
cancer_rate_transformed_numeric.drop(['PM2EXP'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_transformed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 12)


### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>

1. All numeric variables were transformed using the standardization method to achieve a comparable scale between values.
4. The scaled dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting standardization
# to transform the values of the 
# variables into comparable scale
##################################
standardization_scaler = StandardScaler()
cancer_rate_transformed_numeric_array = standardization_scaler.fit_transform(cancer_rate_transformed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the scaled data
##################################
cancer_rate_scaled_numeric = pd.DataFrame(cancer_rate_transformed_numeric_array,
                                          columns=cancer_rate_transformed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_scaled_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_scaled_numeric, x=column)
```


    
![png](output_141_0.png)
    



    
![png](output_141_1.png)
    



    
![png](output_141_2.png)
    



    
![png](output_141_3.png)
    



    
![png](output_141_4.png)
    



    
![png](output_141_5.png)
    



    
![png](output_141_6.png)
    



    
![png](output_141_7.png)
    



    
![png](output_141_8.png)
    



    
![png](output_141_9.png)
    



    
![png](output_141_10.png)
    



    
![png](output_141_11.png)
    


### 1.4.7 Data Encoding <a class="anchor" id="1.4.7"></a>

1. One-hot encoding was applied to the <span style="color: #FF0000">HDICAP_VH</span> variable resulting to 4 additional columns in the dataset:
    * <span style="color: #FF0000">HDICAP_L</span>
    * <span style="color: #FF0000">HDICAP_M</span>
    * <span style="color: #FF0000">HDICAP_H</span>
    * <span style="color: #FF0000">HDICAP_VH</span>


```python
##################################
# Formulating the categorical column
# for encoding transformation
##################################
cancer_rate_categorical_encoded = pd.DataFrame(cancer_rate_cleaned_categorical.loc[:, 'HDICAT'].to_list(),
                                               columns=['HDICAT'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the categorical column
##################################
cancer_rate_categorical_encoded = pd.get_dummies(cancer_rate_categorical_encoded, columns=['HDICAT'])
```

### 1.4.8 Preprocessed Data Description <a class="anchor" id="1.4.8"></a>

1. The preprocessed dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (categorical)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/18 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **4/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT_L</span>
             * <span style="color: #FF0000">HDICAT_M</span>
             * <span style="color: #FF0000">HDICAT_H</span>
             * <span style="color: #FF0000">HDICAT_VH</span>


```python
##################################
# Consolidating both numeric columns
# and encoded categorical columns
##################################
cancer_rate_preprocessed = pd.concat([cancer_rate_scaled_numeric,cancer_rate_categorical_encoded], axis=1, join='inner')  
```


```python
##################################
# Performing a general exploration of the consolidated dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_preprocessed.shape)
```

    Dataset Dimensions: 
    


    (163, 16)


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. Bivariate analysis identified individual predictors with generally positive association to the target variable based on visual inspection.
2. Higher values or higher proportions for the following predictors are associated with the <span style="color: #FF0000">CANRAT</span> HIGH category: 
    * <span style="color: #FF0000">URBPOP</span>
    * <span style="color: #FF0000">LIFEXP</span>    
    * <span style="color: #FF0000">CO2EMI</span>    
    * <span style="color: #FF0000">GDPCAP</span>    
    * <span style="color: #FF0000">EPISCO</span>    
    * <span style="color: #FF0000">HDICAP_VH=1</span>
3. Decreasing values or smaller proportions for the following predictors are associated with the <span style="color: #FF0000">CANRAT</span> LOW category: 
    * <span style="color: #FF0000">POPGRO</span>
    * <span style="color: #FF0000">TUBINC</span>    
    * <span style="color: #FF0000">DTHCMD</span> 
    * <span style="color: #FF0000">HDICAP_L=0</span>
    * <span style="color: #FF0000">HDICAP_M=0</span>
    * <span style="color: #FF0000">HDICAP_H=0</span>
4. Values for the following predictors are not associated with the <span style="color: #FF0000">CANRAT</span>  HIGH or LOW categories: 
    * <span style="color: #FF0000">AGRLND</span>
    * <span style="color: #FF0000">GHGEMI</span>    
    * <span style="color: #FF0000">FORARE</span> 
    * <span style="color: #FF0000">POPDEN</span> 


```python
##################################
# Segregating the target
# and predictor variable lists
##################################
cancer_rate_preprocessed_target = cancer_rate_filtered_row['CANRAT'].to_frame()
cancer_rate_preprocessed_target.reset_index(inplace=True, drop=True)
cancer_rate_preprocessed_categorical = cancer_rate_preprocessed[cancer_rate_categorical_encoded.columns]
cancer_rate_preprocessed_categorical_combined = cancer_rate_preprocessed_categorical.join(cancer_rate_preprocessed_target)
cancer_rate_preprocessed = cancer_rate_preprocessed.drop(cancer_rate_categorical_encoded.columns, axis=1) 
cancer_rate_preprocessed_predictors = cancer_rate_preprocessed.columns
cancer_rate_preprocessed_combined = cancer_rate_preprocessed.join(cancer_rate_preprocessed_target)
```


```python
##################################
# Segregating the target
# and predictor variable names
##################################
y_variable = 'CANRAT'
x_variables = cancer_rate_preprocessed_predictors
```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 6
num_cols = 2
```


```python
##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 30))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual boxplots
# for all scaled numeric columns
##################################
for i, x_variable in enumerate(x_variables):
    ax = axes[i]
    ax.boxplot([group[x_variable] for name, group in cancer_rate_preprocessed_combined.groupby(y_variable, observed=True)])
    ax.set_title(f'{y_variable} Versus {x_variable}')
    ax.set_xlabel(y_variable)
    ax.set_ylabel(x_variable)
    ax.set_xticks(range(1, len(cancer_rate_preprocessed_combined[y_variable].unique()) + 1), ['Low', 'High'])

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()
```


    
![png](output_153_0.png)
    



```python
##################################
# Segregating the target
# and predictor variable names
##################################
y_variables = cancer_rate_preprocessed_categorical.columns
x_variable = 'CANRAT'

##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 2
num_cols = 2

##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual stacked column plots
# for all categorical columns
##################################
for i, y_variable in enumerate(y_variables):
    ax = axes[i]
    category_counts = cancer_rate_preprocessed_categorical_combined.groupby([x_variable, y_variable], observed=True).size().unstack(fill_value=0)
    category_proportions = category_counts.div(category_counts.sum(axis=1), axis=0)
    category_proportions.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{x_variable} Versus {y_variable}')
    ax.set_xlabel(x_variable)
    ax.set_ylabel('Proportions')

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()
```


    
![png](output_154_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups LOW and HIGH is equal to zero  
    * **Alternative**: Difference in the means between groups LOW and HIGH is not equal to zero   
2. There is sufficient evidence to conclude of a statistically significant difference between the means of the numeric measurements obtained from LOW and HIGH groups of the <span style="color: #FF0000">CANRAT</span> target variable in 9 of the 12 numeric predictors given their high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">GDPCAP</span>: T.Test.Statistic=-11.937, Correlation.PValue=0.000
    * <span style="color: #FF0000">EPISCO</span>: T.Test.Statistic=-11.789, Correlation.PValue=0.000 
    * <span style="color: #FF0000">LIFEXP</span>: T.Test.Statistic=-10.979, Correlation.PValue=0.000  
    * <span style="color: #FF0000">TUBINC</span>: T.Test.Statistic=+9.609, Correlation.PValue=0.000 
    * <span style="color: #FF0000">DTHCMD</span>: T.Test.Statistic=+8.376, Correlation.PValue=0.000 
    * <span style="color: #FF0000">CO2EMI</span>: T.Test.Statistic=-7.031, Correlation.PValue=0.000  
    * <span style="color: #FF0000">URBPOP</span>: T.Test.Statistic=-6.541, Correlation.PValue=0.000   
    * <span style="color: #FF0000">POPGRO</span>: T.Test.Statistic=+4.905, Correlation.PValue=0.000
    * <span style="color: #FF0000">GHGEMI</span>: T.Test.Statistic=-2.243, Correlation.PValue=0.026
3. The relationship between the categorical predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: The categorical predictor is independent of the categorical target variable 
    * **Alternative**: The categorical predictor is dependent of the categorical target variable    
2. There is sufficient evidence to conclude of a statistically significant relationship difference between the categories of the categorical predictors and the LOW and HIGH groups of the <span style="color: #FF0000">CANRAT</span> target variable in all 4 categorical predictors given their high chisquare statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">HDICAT_VH</span>: ChiSquare.Test.Statistic=76.764, ChiSquare.Test.PValue=0.000
    * <span style="color: #FF0000">HDICAT_H</span>: ChiSquare.Test.Statistic=13.860, ChiSquare.Test.PValue=0.000   
    * <span style="color: #FF0000">HDICAT_M</span>: ChiSquare.Test.Statistic=10.286, ChiSquare.Test.PValue=0.001 
    * <span style="color: #FF0000">HDICAT_L</span>: ChiSquare.Test.Statistic=9.081, ChiSquare.Test.PValue=0.002


```python
##################################
# Computing the t-test 
# statistic and p-values
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_ttest_target = {}
cancer_rate_preprocessed_numeric = cancer_rate_preprocessed_combined
cancer_rate_preprocessed_numeric_columns = cancer_rate_preprocessed_predictors
for numeric_column in cancer_rate_preprocessed_numeric_columns:
    group_0 = cancer_rate_preprocessed_numeric[cancer_rate_preprocessed_numeric.loc[:,'CANRAT']=='Low']
    group_1 = cancer_rate_preprocessed_numeric[cancer_rate_preprocessed_numeric.loc[:,'CANRAT']=='High']
    cancer_rate_preprocessed_numeric_ttest_target['CANRAT_' + numeric_column] = stats.ttest_ind(
        group_0[numeric_column], 
        group_1[numeric_column], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_summary = cancer_rate_preprocessed_numeric.from_dict(cancer_rate_preprocessed_numeric_ttest_target, orient='index')
cancer_rate_preprocessed_numeric_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(cancer_rate_preprocessed_numeric_summary.sort_values(by=['T.Test.PValue'], ascending=True).head(12))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_GDPCAP</th>
      <td>-11.936988</td>
      <td>6.247937e-24</td>
    </tr>
    <tr>
      <th>CANRAT_EPISCO</th>
      <td>-11.788870</td>
      <td>1.605980e-23</td>
    </tr>
    <tr>
      <th>CANRAT_LIFEXP</th>
      <td>-10.979098</td>
      <td>2.754214e-21</td>
    </tr>
    <tr>
      <th>CANRAT_TUBINC</th>
      <td>9.608760</td>
      <td>1.463678e-17</td>
    </tr>
    <tr>
      <th>CANRAT_DTHCMD</th>
      <td>8.375558</td>
      <td>2.552108e-14</td>
    </tr>
    <tr>
      <th>CANRAT_CO2EMI</th>
      <td>-7.030702</td>
      <td>5.537463e-11</td>
    </tr>
    <tr>
      <th>CANRAT_URBPOP</th>
      <td>-6.541001</td>
      <td>7.734940e-10</td>
    </tr>
    <tr>
      <th>CANRAT_POPGRO</th>
      <td>4.904817</td>
      <td>2.269446e-06</td>
    </tr>
    <tr>
      <th>CANRAT_GHGEMI</th>
      <td>-2.243089</td>
      <td>2.625563e-02</td>
    </tr>
    <tr>
      <th>CANRAT_FORARE</th>
      <td>-1.174143</td>
      <td>2.420717e-01</td>
    </tr>
    <tr>
      <th>CANRAT_POPDEN</th>
      <td>-0.495221</td>
      <td>6.211191e-01</td>
    </tr>
    <tr>
      <th>CANRAT_AGRLND</th>
      <td>-0.047628</td>
      <td>9.620720e-01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the chisquare
# statistic and p-values
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_chisquare_target = {}
cancer_rate_preprocessed_categorical = cancer_rate_preprocessed_categorical_combined
cancer_rate_preprocessed_categorical_columns = ['HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH']
for categorical_column in cancer_rate_preprocessed_categorical_columns:
    contingency_table = pd.crosstab(cancer_rate_preprocessed_categorical[categorical_column], 
                                    cancer_rate_preprocessed_categorical['CANRAT'])
    cancer_rate_preprocessed_categorical_chisquare_target['CANRAT_' + categorical_column] = stats.chi2_contingency(
        contingency_table)[0:2]
```


```python
##################################
# Formulating the pairwise chisquare summary
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_summary = cancer_rate_preprocessed_categorical.from_dict(cancer_rate_preprocessed_categorical_chisquare_target, orient='index')
cancer_rate_preprocessed_categorical_summary.columns = ['ChiSquare.Test.Statistic', 'ChiSquare.Test.PValue']
display(cancer_rate_preprocessed_categorical_summary.sort_values(by=['ChiSquare.Test.PValue'], ascending=True).head(4))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ChiSquare.Test.Statistic</th>
      <th>ChiSquare.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_HDICAT_VH</th>
      <td>76.764134</td>
      <td>1.926446e-18</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_M</th>
      <td>13.860367</td>
      <td>1.969074e-04</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_L</th>
      <td>10.285575</td>
      <td>1.340742e-03</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_H</th>
      <td>9.080788</td>
      <td>2.583087e-03</td>
    </tr>
  </tbody>
</table>
</div>


## 1.6. Neural Network Classification Gradient and Weight Updates <a class="anchor" id="1.6"></a>

### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>
1. Among the predictor variables determined to have a statistically significant difference between the means of the numeric measurements obtained from LOW and HIGH groups of the <span style="color: #FF0000">CANRAT</span> target variable, only 2 were retained with the highest absolute t-test statistic values with reported low p-values less than the significance level of 0.05.. 
    * <span style="color: #FF0000">GDPCAP</span>: T.Test.Statistic=-11.937, Correlation.PValue=0.000
    * <span style="color: #FF0000">EPISCO</span>: T.Test.Statistic=-11.789, Correlation.PValue=0.000 


```python
##################################
# Filtering certain numeric columns
# and encoded categorical columns
# after hypothesis testing
##################################
cancer_rate_premodelling = cancer_rate_preprocessed_combined.drop(['URBPOP', 'POPGRO', 'LIFEXP', 'TUBINC', 'DTHCMD', 'AGRLND', 'GHGEMI','FORARE', 'CO2EMI', 'POPDEN'], axis=1)
cancer_rate_premodelling.columns
```




    Index(['GDPCAP', 'EPISCO', 'CANRAT'], dtype='object')




```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_premodelling.shape)
```

    Dataset Dimensions: 
    


    (163, 3)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate_premodelling.dtypes)
```

    Column Names and Data Types:
    


    GDPCAP     float64
    EPISCO     float64
    CANRAT    category
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate_premodelling.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
      <th>CANRAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.549766</td>
      <td>1.306738</td>
      <td>High</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.407752</td>
      <td>1.102912</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.879374</td>
      <td>1.145832</td>
      <td>High</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.685426</td>
      <td>0.739753</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.657777</td>
      <td>2.218327</td>
      <td>High</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Converting the dataframe to
# a numpy array
##################################
cancer_rate_premodelling_matrix = cancer_rate_premodelling.to_numpy()
```


```python
##################################
# Formulating the scatterplot
# of the selected numeric predictors
# by categorical response classes
##################################
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='High', 0],
        cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='High', 1], 
        'o', 
        label='High', 
        color='darkslateblue')
ax.plot(cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='Low', 0],
        cancer_rate_premodelling_matrix[cancer_rate_premodelling_matrix[:,2]=='Low', 1], 
        'x', 
        label='Low', 
        color='chocolate')
ax.axes.set_ylabel('EPISCO')
ax.axes.set_xlabel('GDPCAP')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set(title='CANRAT Class Distribution')
ax.legend(loc='upper left',title='CANRAT');
```


    
![png](output_167_0.png)
    



```python
##################################
# Preparing the data and
# and converting to a suitable format
# as a neural network model input
##################################
matrix_x_values = cancer_rate_premodelling.iloc[:,0:2].to_numpy()
y_values = np.where(cancer_rate_premodelling['CANRAT'] == 'High', 1, 0)
```

### 1.6.2 Sigmoid Activation Function <a class="anchor" id="1.6.2"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Activation Functions](https://link.springer.com/book/10.1007/978-0-387-84858-7) play a crucial role in neural networks by introducing non-linearity into the network, enabling the model to learn complex patterns and relationships within the data. In the context of a neural network classification model, activation functions are applied to the output of each neuron in the hidden layers to introduce non-linear mappings between the input and output, allowing the network to approximate complex functions and make non-linear decisions. Activation functions are significant during model development by introducing non-linearity (without activation functions, the neural network would simply be a series of linear transformations, no matter how many layers it has. Activation functions introduce non-linearities to the model, enabling it to learn and represent complex patterns and relationships in the data); propagating back gradients (activation functions help in the backpropagation algorithm by providing gradients that indicate the direction and magnitude of adjustments to the weights during training. These gradients are necessary for optimizing the network's parameters through techniques like gradient descent}; and normalizing outputs (activation functions also help in normalizing the output of each neuron, ensuring that it falls within a specific range. This normalization prevents the activation values from becoming too large or too small, which can lead to numerical instability or saturation of gradients during training). The choice of activation function can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.

[Sigmoid Activation Function](https://link.springer.com/book/10.1007/978-0-387-84858-7) transforms a variable by determining the quotient between one and the sum of one and the Euler's constant raised to the given value of the variable. The resulting output resembles a smooth S-shaped curve which ranges from zero to one. Since, it squashes the input values between zero and one, it is useful for binary classification. The curve profile is not zero-centered which can lead to vanishing gradients problem during backpropagation. It is also prone to saturation, causing gradients to vanish when the input is too large or too small.

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 4
2. The backpropagation and gradient descent algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Activation Function</span> = Sigmoid Activation Function
3. The final loss estimate determined as 0.27805 at the 1000th epoch was not optimally low as compared to those obtained using the other parameter settings.
4. Applying the backpropagation and gradient descent algorithms with a Sigmoid activation function, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 74.84662
5. The estimated classification accuracy using the backpropagation and gradient descent algorithms with a Sigmoid activation function was not optimal as compared to those obtained using the other parameter settings, also demonstrating a consistently smooth profile during the epoch training


```python
##################################
# Creating a class object
# for the neural network algorithm
# using a sigmoid activation function
##################################
class NeuralNetwork_Sigmoid:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.bias1 = np.zeros((1, hidden_size1))
        
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias2 = np.zeros((1, hidden_size2))
        
        self.weights3 = np.random.randn(hidden_size2, hidden_size3)
        self.bias3 = np.zeros((1, hidden_size3))
        
        self.weights4 = np.random.randn(hidden_size3, output_size)
        self.bias4 = np.zeros((1, output_size))
        
        self.gradients = {'dw1': [], 'db1': [], 'dw2': [], 'db2': [], 'dw3': [], 'db3': [], 'dw4': [], 'db4': []}
        self.losses = []
        self.accuracies = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, x):
        self.hidden1 = self.sigmoid(np.dot(x, self.weights1) + self.bias1)
        self.hidden2 = self.sigmoid(np.dot(self.hidden1, self.weights2) + self.bias2)
        self.hidden3 = self.sigmoid(np.dot(self.hidden2, self.weights3) + self.bias3)
        self.output = self.softmax(np.dot(self.hidden3, self.weights4) + self.bias4)
        
        return self.output
    
    def backward(self, x, y, lr):
        m = x.shape[0]
        
       # Computing the output layer gradients
        output_error = self.output - y
        output_delta = output_error / m
        
        # Computing the hidden layer 3 gradients
        hidden3_error = np.dot(output_delta, self.weights4.T)
        hidden3_delta = hidden3_error * self.sigmoid_derivative(self.hidden3)
        
        # Computing the hidden layer 2 gradients
        hidden2_error = np.dot(hidden3_delta, self.weights3.T)
        hidden2_delta = hidden2_error * self.sigmoid_derivative(self.hidden2)
        
        # Computing the hidden layer 1 gradients
        hidden1_error = np.dot(hidden2_delta, self.weights2.T)
        hidden1_delta = hidden1_error * self.sigmoid_derivative(self.hidden1)
        
        # Updating weights and biases based on computed gradients
        self.weights4 -= lr * np.dot(self.hidden3.T, output_delta)
        self.bias4 -= lr * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights3 -= lr * np.dot(self.hidden2.T, hidden3_delta)
        self.bias3 -= lr * np.sum(hidden3_delta, axis=0, keepdims=True)
        
        self.weights2 -= lr * np.dot(self.hidden1.T, hidden2_delta)
        self.bias2 -= lr * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.weights1 -= lr * np.dot(x.T, hidden1_delta)
        self.bias1 -= lr * np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Storing the updated gradients
        self.gradients['dw1'].append(np.mean(np.abs(hidden1_delta)))
        self.gradients['db1'].append(np.mean(np.abs(self.bias1)))
        self.gradients['dw2'].append(np.mean(np.abs(hidden2_delta)))
        self.gradients['db2'].append(np.mean(np.abs(self.bias2)))
        self.gradients['dw3'].append(np.mean(np.abs(hidden3_delta)))
        self.gradients['db3'].append(np.mean(np.abs(self.bias3)))
        self.gradients['dw4'].append(np.mean(np.abs(output_delta)))
        self.gradients['db4'].append(np.mean(np.abs(self.bias4)))
                       
    def train(self, x, y, epochs, lr):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            loss = -np.mean(y * np.log(output))
            self.losses.append(loss)
            accuracy = self.accuracy(x, np.argmax(y, axis=1))
            self.accuracies.append(accuracy)
            if i % 100 == 0:
                print(f'Epoch {i}: Loss {loss}, Accuracy {accuracy}')
                
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def accuracy(self, x, y):
        pred = self.predict(x)
        return np.mean(pred == y)
```


```python
##################################
# Preparing the training data
##################################
X = matrix_x_values
y = y_values

##################################
# Performing a one-hot encoding
# of the target response labels
##################################
num_classes = 2
y_one_hot = np.eye(num_classes)[y]
```


```python
##################################
# Defining the neural network components
##################################
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
hidden_size3 = 4
output_size = 2
```


```python
##################################
# Initializing a neural network model object
# with sigmoid activation function
##################################
np.random.seed(88888)
nn_sigmoid = NeuralNetwork_Sigmoid(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
```


```python
##################################
# Training a neural network model
# with sigmoid activation function
##################################
nn_sigmoid.train(X, y_one_hot, epochs=1001, lr=0.01)
```

    Epoch 0: Loss 0.316009360251368, Accuracy 0.7484662576687117
    Epoch 100: Loss 0.29042169855505795, Accuracy 0.7484662576687117
    Epoch 200: Loss 0.28610782259171613, Accuracy 0.7484662576687117
    Epoch 300: Loss 0.2846113282316412, Accuracy 0.7484662576687117
    Epoch 400: Loss 0.2835652925120681, Accuracy 0.7484662576687117
    Epoch 500: Loss 0.2826104420839403, Accuracy 0.7484662576687117
    Epoch 600: Loss 0.28168417134187307, Accuracy 0.7484662576687117
    Epoch 700: Loss 0.28077152406724765, Accuracy 0.7484662576687117
    Epoch 800: Loss 0.27986564301503736, Accuracy 0.7484662576687117
    Epoch 900: Loss 0.2789612062764938, Accuracy 0.7484662576687117
    Epoch 1000: Loss 0.27805328141665736, Accuracy 0.7484662576687117
    


```python
##################################
# Plotting the computed gradients
# of the neural network model
# with sigmoid activation function
##################################
plt.figure(figsize=(10, 6))
for key, value in nn_sigmoid.gradients.items():
    plt.plot(value, label=key)
plt.title('Sigmoid Activation: Gradients by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Gradients')
plt.ylim(-0.05, 0.50)
plt.xlim(-50,1000)
plt.legend(loc="upper left")
plt.show()
```


    
![png](output_175_0.png)
    



```python
##################################
# Plotting the computed cost
# of the neural network model
# with sigmoid activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_sigmoid.losses)
plt.title('Sigmoid Activation: Cost Function by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim(0.05, 0.70)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_176_0.png)
    



```python
##################################
# Plotting the computed accuracy
# of the neural network model
# with sigmoid activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_sigmoid.accuracies)
plt.title('Sigmoid Activation: Classification by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_177_0.png)
    



```python
##################################
# Gathering the final values for 
# accuracy and loss error
##################################
sigmoid_metrics = pd.DataFrame(["ACCURACY","LOSS"])
sigmoid_values = pd.DataFrame([nn_sigmoid.accuracies[-1],nn_sigmoid.losses[-1]])
sigmoid_method = pd.DataFrame(["Sigmoid"]*2)
sigmoid_summary = pd.concat([sigmoid_metrics, 
                             sigmoid_values,
                             sigmoid_method], axis=1)
sigmoid_summary.columns = ['Metric', 'Value', 'Method']
sigmoid_summary.reset_index(inplace=True, drop=True)
display(sigmoid_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.748466</td>
      <td>Sigmoid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.278053</td>
      <td>Sigmoid</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.3 Rectified Linear Unit Activation Function <a class="anchor" id="1.6.3"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data. This activation function is commonly used in the output layer for binary classification problems.

[Activation Functions](https://link.springer.com/book/10.1007/978-0-387-84858-7) play a crucial role in neural networks by introducing non-linearity into the network, enabling the model to learn complex patterns and relationships within the data. In the context of a neural network classification model, activation functions are applied to the output of each neuron in the hidden layers to introduce non-linear mappings between the input and output, allowing the network to approximate complex functions and make non-linear decisions. Activation functions are significant during model development by introducing non-linearity (without activation functions, the neural network would simply be a series of linear transformations, no matter how many layers it has. Activation functions introduce non-linearities to the model, enabling it to learn and represent complex patterns and relationships in the data); propagating back gradients (activation functions help in the backpropagation algorithm by providing gradients that indicate the direction and magnitude of adjustments to the weights during training. These gradients are necessary for optimizing the network's parameters through techniques like gradient descent}; and normalizing outputs (activation functions also help in normalizing the output of each neuron, ensuring that it falls within a specific range. This normalization prevents the activation values from becoming too large or too small, which can lead to numerical instability or saturation of gradients during training). The choice of activation function can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.

[Rectified Linear Unit Activation Function](https://link.springer.com/book/10.1007/978-0-387-84858-7) transforms a variable by determining the maximum between zero and the given value of the variable. The resulting output which ranges from zero to positive infinity is a piecewise linear function that returns zero for negative inputs and the input value for positive inputs. It is a simple and computationally efficient method which solves the vanishing gradients problem and accelerates convergence by avoiding saturation to positive values. However, it can suffer from dying RELU problem where neurons become inactive (outputs zero) for all inputs during training if large gradients consistently flow through them.  The curve profile is not zero-centered which can lead to vanishing gradients problem during backpropagation. It is also prone to saturation, causing gradients to vanish when the input is too large or too small. This activation function is widely used in hidden layers of deep neural networks.

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 4
2. The backpropagation and gradient descent algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Activation Function</span> = Rectified Linear Unit Activation Function (RELU)
3. The final loss estimate determined as 0.09623 at the 1000th epoch was not optimally low as compared to those obtained using the other parameter settings.
4. Applying the backpropagation and gradient descent algorithms with a RELU activation function, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 93.25153
5. The estimated classification accuracy using the backpropagation and gradient descent algorithms with a RELU activation function was optimal as compared to those obtained using the other parameter settings, also demonstrating a consistently smooth profile during the epoch training


```python
##################################
# Creating a class object
# for the neural network algorithm
# using a RELU activation function
##################################
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork_RELU:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.bias1 = np.zeros((1, hidden_size1))
        
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias2 = np.zeros((1, hidden_size2))
        
        self.weights3 = np.random.randn(hidden_size2, hidden_size3)
        self.bias3 = np.zeros((1, hidden_size3))
        
        self.weights4 = np.random.randn(hidden_size3, output_size)
        self.bias4 = np.zeros((1, output_size))
        
        self.gradients = {'dw1': [], 'db1': [], 'dw2': [], 'db2': [], 'dw3': [], 'db3': [], 'dw4': [], 'db4': []}
        self.losses = []
        self.accuracies = []
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, x):
        self.hidden1 = self.relu(np.dot(x, self.weights1) + self.bias1)
        self.hidden2 = self.relu(np.dot(self.hidden1, self.weights2) + self.bias2)
        self.hidden3 = self.relu(np.dot(self.hidden2, self.weights3) + self.bias3)
        self.output = self.softmax(np.dot(self.hidden3, self.weights4) + self.bias4)
        
        return self.output
    
    def backward(self, x, y, lr):
        m = x.shape[0]
        
        # Computing the output layer gradients
        output_error = self.output - y
        output_delta = output_error / m
        
        # Computing the hidden layer 3 gradients
        hidden3_error = np.dot(output_delta, self.weights4.T)
        hidden3_delta = hidden3_error * (self.hidden3 > 0)
        
        # Computing the hidden layer 2 gradients
        hidden2_error = np.dot(hidden3_delta, self.weights3.T)
        hidden2_delta = hidden2_error * (self.hidden2 > 0)
        
        # Computing the hidden layer 1 gradients
        hidden1_error = np.dot(hidden2_delta, self.weights2.T)
        hidden1_delta = hidden1_error * (self.hidden1 > 0)
        
        # Updating weights and biases based on computed gradients
        self.weights4 -= lr * np.dot(self.hidden3.T, output_delta)
        self.bias4 -= lr * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights3 -= lr * np.dot(self.hidden2.T, hidden3_delta)
        self.bias3 -= lr * np.sum(hidden3_delta, axis=0, keepdims=True)
        
        self.weights2 -= lr * np.dot(self.hidden1.T, hidden2_delta)
        self.bias2 -= lr * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.weights1 -= lr * np.dot(x.T, hidden1_delta)
        self.bias1 -= lr * np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Storing computed gradients
        self.gradients['dw1'].append(np.mean(np.abs(hidden1_delta)))
        self.gradients['db1'].append(np.mean(np.abs(self.bias1)))
        self.gradients['dw2'].append(np.mean(np.abs(hidden2_delta)))
        self.gradients['db2'].append(np.mean(np.abs(self.bias2)))
        self.gradients['dw3'].append(np.mean(np.abs(hidden3_delta)))
        self.gradients['db3'].append(np.mean(np.abs(self.bias3)))
        self.gradients['dw4'].append(np.mean(np.abs(output_delta)))
        self.gradients['db4'].append(np.mean(np.abs(self.bias4)))
        
    def train(self, x, y, epochs, lr):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            loss = -np.mean(y * np.log(output))
            self.losses.append(loss)
            accuracy = self.accuracy(x, np.argmax(y, axis=1))
            self.accuracies.append(accuracy)
            if i % 100 == 0:
                print(f'Epoch {i}: Loss {loss}, Accuracy {accuracy}')
                
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def accuracy(self, x, y):
        pred = self.predict(x)
        return np.mean(pred == y)

```


```python
##################################
# Preparing the training data
##################################
X = matrix_x_values
y = y_values

##################################
# Performing a one-hot encoding
# of the target response labels
##################################
num_classes = 2
y_one_hot = np.eye(num_classes)[y]
```


```python
##################################
# Defining the neural network components
##################################
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
hidden_size3 = 4
output_size = 2
```


```python
##################################
# Initializing a neural network model object
# with RELU activation function
##################################
np.random.seed(88888)
nn_relu = NeuralNetwork_RELU(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
```


```python
##################################
# Training a neural network model
# with RELU activation function
##################################
nn_relu.train(X, y_one_hot, epochs=1001, lr=0.01)
```

    Epoch 0: Loss 0.3700211712894534, Accuracy 0.2822085889570552
    Epoch 100: Loss 0.1794249587592613, Accuracy 0.901840490797546
    Epoch 200: Loss 0.13224275039852623, Accuracy 0.9079754601226994
    Epoch 300: Loss 0.1167678880588832, Accuracy 0.9141104294478528
    Epoch 400: Loss 0.10890482140743643, Accuracy 0.9202453987730062
    Epoch 500: Loss 0.10455617528678188, Accuracy 0.9202453987730062
    Epoch 600: Loss 0.10189142339749091, Accuracy 0.9202453987730062
    Epoch 700: Loss 0.09987241742776284, Accuracy 0.9263803680981595
    Epoch 800: Loss 0.09805768178547344, Accuracy 0.9325153374233128
    Epoch 900: Loss 0.096972829866501, Accuracy 0.9325153374233128
    Epoch 1000: Loss 0.09623684890656739, Accuracy 0.9325153374233128
    


```python
##################################
# Plotting the computed gradients
# of the neural network model
# with RELU activation function
##################################
plt.figure(figsize=(10, 6))
for key, value in nn_relu.gradients.items():
    plt.plot(value, label=key)
plt.title('RELU Activation: Gradients by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Gradients')
plt.ylim(-0.05, 0.50)
plt.xlim(-50,1000)
plt.legend(loc="upper left")
plt.show()
```


    
![png](output_185_0.png)
    



```python
##################################
# Plotting the computed cost
# of the neural network model
# with RELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_relu.losses)
plt.title('RELU Activation: Cost Function by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim(0.05, 0.70)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_186_0.png)
    



```python
##################################
# Plotting the computed accuracy
# of the neural network model
# with RELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_relu.accuracies)
plt.title('RELU Activation: Classification by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_187_0.png)
    



```python
##################################
# Gathering the final values for 
# accuracy and loss error
# using a RELU activation function
##################################
relu_metrics = pd.DataFrame(["ACCURACY","LOSS"])
relu_values = pd.DataFrame([nn_relu.accuracies[-1],nn_relu.losses[-1]])
relu_method = pd.DataFrame(["RELU"]*2)
relu_summary = pd.concat([relu_metrics, 
                          relu_values,
                          relu_method], axis=1)
relu_summary.columns = ['Metric', 'Value', 'Method']
relu_summary.reset_index(inplace=True, drop=True)
display(relu_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.932515</td>
      <td>RELU</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.096237</td>
      <td>RELU</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.4 Leaky Rectified Linear Unit Activation Function <a class="anchor" id="1.6.4"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Activation Functions](https://link.springer.com/book/10.1007/978-0-387-84858-7) play a crucial role in neural networks by introducing non-linearity into the network, enabling the model to learn complex patterns and relationships within the data. In the context of a neural network classification model, activation functions are applied to the output of each neuron in the hidden layers to introduce non-linear mappings between the input and output, allowing the network to approximate complex functions and make non-linear decisions. Activation functions are significant during model development by introducing non-linearity (without activation functions, the neural network would simply be a series of linear transformations, no matter how many layers it has. Activation functions introduce non-linearities to the model, enabling it to learn and represent complex patterns and relationships in the data); propagating back gradients (activation functions help in the backpropagation algorithm by providing gradients that indicate the direction and magnitude of adjustments to the weights during training. These gradients are necessary for optimizing the network's parameters through techniques like gradient descent}; and normalizing outputs (activation functions also help in normalizing the output of each neuron, ensuring that it falls within a specific range. This normalization prevents the activation values from becoming too large or too small, which can lead to numerical instability or saturation of gradients during training). The choice of activation function can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.

[Leaky Rectified Linear Unit Activation Function](https://link.springer.com/book/10.1007/978-0-387-84858-7) transforms a variable by using the given value of the variable when it is greater than zero, but multiplies this value with a specific alpha otherwise. The resulting output which ranges from negative infinity to positive infinity is a piecewise linear function that returns the input value for positive inputs but allows a small non-zero value when the input is negative, preventing the dying RELU problem. It introduces a small slope for negative inputs, keeping the gradient flowing during backpropagation. This activation function provides a good alternative to RELU especially in networks where dying RELU is a concern.

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 4
2. The backpropagation and gradient descent algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Activation Function</span> = Leaky Rectified Linear Unit Activation Function (Leaky RELU)
3. The final loss estimate determined as 0.09377 at the 1000th epoch was not optimally low as compared to those obtained using the other parameter settings.
4. Applying the backpropagation and gradient descent algorithms with a Leaky RELU activation function, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 92.63803
5. The estimated classification accuracy using the backpropagation and gradient descent algorithms with a Leaky RELU activation function was optimal as compared to those obtained using the other parameter settings, also demonstrating a consistently smooth profile during the epoch training.


```python
##################################
# Creating a class object
# for the neural network algorithm
# using a Leaky RELU activation function
##################################
class NeuralNetwork_LeakyRELU:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.bias1 = np.zeros((1, hidden_size1))
        
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias2 = np.zeros((1, hidden_size2))
        
        self.weights3 = np.random.randn(hidden_size2, hidden_size3)
        self.bias3 = np.zeros((1, hidden_size3))
        
        self.weights4 = np.random.randn(hidden_size3, output_size)
        self.bias4 = np.zeros((1, output_size))
        
        self.gradients = {'dw1': [], 'db1': [], 'dw2': [], 'db2': [], 'dw3': [], 'db3': [], 'dw4': [], 'db4': []}
        self.losses = []
        self.accuracies = []
        
    def leaky_relu(self, x, alpha=0.10):
        return np.where(x > 0, x, alpha * x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, x):
        self.hidden1 = self.leaky_relu(np.dot(x, self.weights1) + self.bias1)
        self.hidden2 = self.leaky_relu(np.dot(self.hidden1, self.weights2) + self.bias2)
        self.hidden3 = self.leaky_relu(np.dot(self.hidden2, self.weights3) + self.bias3)
        self.output = self.softmax(np.dot(self.hidden3, self.weights4) + self.bias4)
        
        return self.output
    
    def backward(self, x, y, lr):
        m = x.shape[0]
        
        # Computing the output layer gradients
        output_error = self.output - y
        output_delta = output_error / m
        
        # Computing the hidden layer 3 gradients
        hidden3_error = np.dot(output_delta, self.weights4.T)
        hidden3_delta = hidden3_error * (self.hidden3 > 0) + 0.01 * hidden3_error * (self.hidden3 <= 0)
        
        # Computing the hidden layer 2 gradients
        hidden2_error = np.dot(hidden3_delta, self.weights3.T)
        hidden2_delta = hidden2_error * (self.hidden2 > 0) + 0.01 * hidden2_error * (self.hidden2 <= 0)
        
        # Computing the hidden layer 1 gradients
        hidden1_error = np.dot(hidden2_delta, self.weights2.T)
        hidden1_delta = hidden1_error * (self.hidden1 > 0) + 0.01 * hidden1_error * (self.hidden1 <= 0)
        
        # Updating weights and biases based on computed gradients
        self.weights4 -= lr * np.dot(self.hidden3.T, output_delta)
        self.bias4 -= lr * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights3 -= lr * np.dot(self.hidden2.T, hidden3_delta)
        self.bias3 -= lr * np.sum(hidden3_delta, axis=0, keepdims=True)
        
        self.weights2 -= lr * np.dot(self.hidden1.T, hidden2_delta)
        self.bias2 -= lr * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.weights1 -= lr * np.dot(x.T, hidden1_delta)
        self.bias1 -= lr * np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Storing updated gradients
        self.gradients['dw1'].append(np.mean(np.abs(hidden1_delta)))
        self.gradients['db1'].append(np.mean(np.abs(self.bias1)))
        self.gradients['dw2'].append(np.mean(np.abs(hidden2_delta)))
        self.gradients['db2'].append(np.mean(np.abs(self.bias2)))
        self.gradients['dw3'].append(np.mean(np.abs(hidden3_delta)))
        self.gradients['db3'].append(np.mean(np.abs(self.bias3)))
        self.gradients['dw4'].append(np.mean(np.abs(output_delta)))
        self.gradients['db4'].append(np.mean(np.abs(self.bias4)))
        
    def train(self, x, y, epochs, lr):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            loss = -np.mean(y * np.log(output))
            self.losses.append(loss)
            accuracy = self.accuracy(x, np.argmax(y, axis=1))
            self.accuracies.append(accuracy)
            if i % 100 == 0:
                print(f'Epoch {i}: Loss {loss}, Accuracy {accuracy}')
                
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def accuracy(self, x, y):
        pred = self.predict(x)
        return np.mean(pred == y)
    
```


```python
##################################
# Preparing the training data
##################################
X = matrix_x_values
y = y_values

##################################
# Performing a one-hot encoding
# of the target response labels
##################################
num_classes = 2
y_one_hot = np.eye(num_classes)[y]
```


```python
##################################
# Defining the neural network components
##################################
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
hidden_size3 = 4
output_size = 2
```


```python
##################################
# Initializing a neural network model object
# with RELU activation function
##################################
np.random.seed(88888)
nn_leakyrelu = NeuralNetwork_LeakyRELU(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
```


```python
##################################
# Training a neural network model
# with Leaky RELU activation function
##################################
nn_leakyrelu.train(X, y_one_hot, epochs=1001, lr=0.01)
```

    Epoch 0: Loss 0.5521727210322692, Accuracy 0.3803680981595092
    Epoch 100: Loss 0.1584376716587371, Accuracy 0.9079754601226994
    Epoch 200: Loss 0.12183916598907361, Accuracy 0.901840490797546
    Epoch 300: Loss 0.11001101254184688, Accuracy 0.9263803680981595
    Epoch 400: Loss 0.10402068587112523, Accuracy 0.9263803680981595
    Epoch 500: Loss 0.10064359626697053, Accuracy 0.9263803680981595
    Epoch 600: Loss 0.09839877372876189, Accuracy 0.9325153374233128
    Epoch 700: Loss 0.09675845212053208, Accuracy 0.9325153374233128
    Epoch 800: Loss 0.09550172181604426, Accuracy 0.9263803680981595
    Epoch 900: Loss 0.09454243365295582, Accuracy 0.9263803680981595
    Epoch 1000: Loss 0.093777159984923, Accuracy 0.9263803680981595
    


```python
##################################
# Plotting the computed gradients
# of the neural network model
# with RELU activation function
##################################
plt.figure(figsize=(10, 6))
for key, value in nn_leakyrelu.gradients.items():
    plt.plot(value, label=key)
plt.title('Leaky RELU Activation: Gradients by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Gradients')
plt.ylim(-0.05, 0.50)
plt.xlim(-50,1000)
plt.legend(loc="upper left")
plt.show()
```


    
![png](output_195_0.png)
    



```python
##################################
# Plotting the computed cost
# of the neural network model
# with Leaky RELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_leakyrelu.losses)
plt.title('Leaky RELU Activation: Cost Function by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim(0.05, 0.70)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_196_0.png)
    



```python
##################################
# Plotting the computed accuracy
# of the neural network model
# with Leaky RELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_leakyrelu.accuracies)
plt.title('Leaky RELU Activation: Classification by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_197_0.png)
    



```python
##################################
# Gathering the final values for 
# accuracy and loss error
# using a Leaky RELU activation function
##################################
leakyrelu_metrics = pd.DataFrame(["ACCURACY","LOSS"])
leakyrelu_values = pd.DataFrame([nn_leakyrelu.accuracies[-1],nn_leakyrelu.losses[-1]])
leakyrelu_method = pd.DataFrame(["Leaky_RELU"]*2)
leakyrelu_summary = pd.concat([leakyrelu_metrics,
                               leakyrelu_values,
                               leakyrelu_method], axis=1)
leakyrelu_summary.columns = ['Metric', 'Value', 'Method']
leakyrelu_summary.reset_index(inplace=True, drop=True)
display(leakyrelu_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>Leaky_RELU</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.093777</td>
      <td>Leaky_RELU</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.5 Exponential Linear Unit Activation Function <a class="anchor" id="1.6.5"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Activation Functions](https://link.springer.com/book/10.1007/978-0-387-84858-7) play a crucial role in neural networks by introducing non-linearity into the network, enabling the model to learn complex patterns and relationships within the data. In the context of a neural network classification model, activation functions are applied to the output of each neuron in the hidden layers to introduce non-linear mappings between the input and output, allowing the network to approximate complex functions and make non-linear decisions. Activation functions are significant during model development by introducing non-linearity (without activation functions, the neural network would simply be a series of linear transformations, no matter how many layers it has. Activation functions introduce non-linearities to the model, enabling it to learn and represent complex patterns and relationships in the data); propagating back gradients (activation functions help in the backpropagation algorithm by providing gradients that indicate the direction and magnitude of adjustments to the weights during training. These gradients are necessary for optimizing the network's parameters through techniques like gradient descent}; and normalizing outputs (activation functions also help in normalizing the output of each neuron, ensuring that it falls within a specific range. This normalization prevents the activation values from becoming too large or too small, which can lead to numerical instability or saturation of gradients during training). The choice of activation function can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.

[Exponential Linear Unit Activation Function](https://link.springer.com/book/10.1007/978-0-387-84858-7) transforms a variable by using the given value of the variable when it is greater than zero, but uses this value as power for the Euler's constant then subtracted by one prior to multiplying with a specific alpha otherwise. The resulting output which ranges from negative infinity to positive infinity is similar to a Leaky RELU profile but with an exponential function for negative inputs, allowing negative values to have negative outputs. Its profile is smooth and differentiable helping speed up convergence which can lead to faster learning. This activation function can be a good choice when smoother and faster convergence are desired.

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 4
2. The backpropagation and gradient descent algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Activation Function</span> = Exponential Linear Unit (ELU)
3. The final loss estimate determined as 0.09443 at the 1000th epoch was not optimally low as compared to those obtained using the other parameter settings.
4. Applying the backpropagation and gradient descent algorithms with a ELU activation function, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 92.63803
5. The estimated classification accuracy using the backpropagation and gradient descent algorithms with a ELU activation function was optimal as compared to those obtained using the other parameter settings, also demonstrating a consistently smooth profile during the epoch training.


```python
##################################
# Creating a class object
# for the neural network algorithm
# using an ELU activation function
##################################
class NeuralNetwork_ELU:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.bias1 = np.zeros((1, hidden_size1))
        
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias2 = np.zeros((1, hidden_size2))
        
        self.weights3 = np.random.randn(hidden_size2, hidden_size3)
        self.bias3 = np.zeros((1, hidden_size3))
        
        self.weights4 = np.random.randn(hidden_size3, output_size)
        self.bias4 = np.zeros((1, output_size))
        
        self.gradients = {'dw1': [], 'db1': [], 'dw2': [], 'db2': [], 'dw3': [], 'db3': [], 'dw4': [], 'db4': []}
        self.losses = []
        self.accuracies = []
        
    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, x):
        self.hidden1 = self.elu(np.dot(x, self.weights1) + self.bias1)
        self.hidden2 = self.elu(np.dot(self.hidden1, self.weights2) + self.bias2)
        self.hidden3 = self.elu(np.dot(self.hidden2, self.weights3) + self.bias3)
        self.output = self.softmax(np.dot(self.hidden3, self.weights4) + self.bias4)
        
        return self.output
    
    def backward(self, x, y, lr):
        m = x.shape[0]
        
        # Computing output layer gradients
        output_error = self.output - y
        output_delta = output_error / m
        
        # Computing hidden layer 3 gradients
        hidden3_error = np.dot(output_delta, self.weights4.T)
        hidden3_delta = hidden3_error * (self.hidden3 > 0) + (np.exp(self.hidden3) - 1) * (self.hidden3 <= 0)
        
        # Computing hidden layer 2 gradients
        hidden2_error = np.dot(hidden3_delta, self.weights3.T)
        hidden2_delta = hidden2_error * (self.hidden2 > 0) + (np.exp(self.hidden2) - 1) * (self.hidden2 <= 0)
        
        # Computing hidden layer 1 gradients
        hidden1_error = np.dot(hidden2_delta, self.weights2.T)
        hidden1_delta = hidden1_error * (self.hidden1 > 0) + (np.exp(self.hidden1) - 1) * (self.hidden1 <= 0)
        
        # Updating weights and biases based on computed gradients
        self.weights4 -= lr * np.dot(self.hidden3.T, output_delta)
        self.bias4 -= lr * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights3 -= lr * np.dot(self.hidden2.T, hidden3_delta)
        self.bias3 -= lr * np.sum(hidden3_delta, axis=0, keepdims=True)
        
        self.weights2 -= lr * np.dot(self.hidden1.T, hidden2_delta)
        self.bias2 -= lr * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.weights1 -= lr * np.dot(x.T, hidden1_delta)
        self.bias1 -= lr * np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Storing updated gradients
        self.gradients['dw1'].append(np.mean(np.abs(hidden1_delta)))
        self.gradients['db1'].append(np.mean(np.abs(self.bias1)))
        self.gradients['dw2'].append(np.mean(np.abs(hidden2_delta)))
        self.gradients['db2'].append(np.mean(np.abs(self.bias2)))
        self.gradients['dw3'].append(np.mean(np.abs(hidden3_delta)))
        self.gradients['db3'].append(np.mean(np.abs(self.bias3)))
        self.gradients['dw4'].append(np.mean(np.abs(output_delta)))
        self.gradients['db4'].append(np.mean(np.abs(self.bias4)))
        
    def train(self, x, y, epochs, lr):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            loss = -np.mean(y * np.log(output))
            self.losses.append(loss)
            accuracy = self.accuracy(x, np.argmax(y, axis=1))
            self.accuracies.append(accuracy)
            if i % 100 == 0:
                print(f'Epoch {i}: Loss {loss}, Accuracy {accuracy}')
                
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def accuracy(self, x, y):
        pred = self.predict(x)
        return np.mean(pred == y)
    
```


```python
##################################
# Preparing the training data
##################################
X = matrix_x_values
y = y_values

##################################
# Performing a one-hot encoding
# of the target response labels
##################################
num_classes = 2
y_one_hot = np.eye(num_classes)[y]
```


```python
##################################
# Defining the neural network components
##################################
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
hidden_size3 = 4
output_size = 2
```


```python
##################################
# Initializing a neural network model object
# with ELU activation function
##################################
np.random.seed(88888)
nn_elu = NeuralNetwork_ELU(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
```


```python
##################################
# Training a neural network model
# with ELU activation function
##################################
nn_elu.train(X, y_one_hot, epochs=1001, lr=0.01)
```

    Epoch 0: Loss 0.6239532743548983, Accuracy 0.6748466257668712
    Epoch 100: Loss 0.18559850810160425, Accuracy 0.7852760736196319
    Epoch 200: Loss 0.1559148417573589, Accuracy 0.8711656441717791
    Epoch 300: Loss 0.1318774124814263, Accuracy 0.901840490797546
    Epoch 400: Loss 0.11492254996431317, Accuracy 0.9141104294478528
    Epoch 500: Loss 0.10509129045149036, Accuracy 0.9263803680981595
    Epoch 600: Loss 0.10021355309897699, Accuracy 0.9263803680981595
    Epoch 700: Loss 0.09771434144678737, Accuracy 0.9263803680981595
    Epoch 800: Loss 0.09611887321358292, Accuracy 0.9263803680981595
    Epoch 900: Loss 0.09509628626392996, Accuracy 0.9263803680981595
    Epoch 1000: Loss 0.09443192186232484, Accuracy 0.9263803680981595
    


```python
##################################
# Plotting the computed gradients
# of the neural network model
# with ELU activation function
##################################
plt.figure(figsize=(10, 6))
for key, value in nn_elu.gradients.items():
    plt.plot(value, label=key)
plt.title('ELU Activation: Gradients by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Gradients')
plt.ylim(-0.05, 0.50)
plt.xlim(-50,1000)
plt.legend(loc="upper left")
plt.show()
```


    
![png](output_205_0.png)
    



```python
##################################
# Plotting the computed cost
# of the neural network model
# with ELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_elu.losses)
plt.title('ELU Activation: Cost Function by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim(0.05, 0.70)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_206_0.png)
    



```python
##################################
# Plotting the computed accuracy
# of the neural network model
# with ELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_elu.accuracies)
plt.title('ELU Activation: Classification by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_207_0.png)
    



```python
##################################
# Gathering the final values for 
# accuracy and loss error
# using a ELU activation function
##################################
elu_metrics = pd.DataFrame(["ACCURACY","LOSS"])
elu_values = pd.DataFrame([nn_elu.accuracies[-1],nn_elu.losses[-1]])
elu_method = pd.DataFrame(["ELU"]*2)
elu_summary = pd.concat([elu_metrics,
                         elu_values,
                         elu_method], axis=1)
elu_summary.columns = ['Metric', 'Value', 'Method']
elu_summary.reset_index(inplace=True, drop=True)
display(elu_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ELU</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.094432</td>
      <td>ELU</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.6 Scaled Exponential Linear Unit Activation Function <a class="anchor" id="1.6.6"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Activation Functions](https://link.springer.com/book/10.1007/978-0-387-84858-7) play a crucial role in neural networks by introducing non-linearity into the network, enabling the model to learn complex patterns and relationships within the data. In the context of a neural network classification model, activation functions are applied to the output of each neuron in the hidden layers to introduce non-linear mappings between the input and output, allowing the network to approximate complex functions and make non-linear decisions. Activation functions are significant during model development by introducing non-linearity (without activation functions, the neural network would simply be a series of linear transformations, no matter how many layers it has. Activation functions introduce non-linearities to the model, enabling it to learn and represent complex patterns and relationships in the data); propagating back gradients (activation functions help in the backpropagation algorithm by providing gradients that indicate the direction and magnitude of adjustments to the weights during training. These gradients are necessary for optimizing the network's parameters through techniques like gradient descent}; and normalizing outputs (activation functions also help in normalizing the output of each neuron, ensuring that it falls within a specific range. This normalization prevents the activation values from becoming too large or too small, which can lead to numerical instability or saturation of gradients during training). The choice of activation function can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.

[Scaled Exponential Linear Unit Activation Function](https://link.springer.com/book/10.1007/978-0-387-84858-7) transforms a variable by using the given value of the variable when it is greater than zero, but uses this value as power for the Euler's constant then subtracted by one prior to multiplying with a specific alpha otherwise - but multiplying either value with a separately defined scaling factor. The resulting output which ranges from negative infinity to positive infinity is similar to an ELU profile but was designed to maintain a stable mean and variance of activations across layers, promoting self-normalization. This method provides a solution to vanishing and exploding gradient problems in deep networks. This activation function is typically used in deep neural networks where normalizing the activations is crucial for stability.

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 4
2. The backpropagation and gradient descent algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Activation Function</span> = Scaled Exponential Linear Unit (SELU)
3. The final loss estimate determined as 0.09674 at the 1000th epoch was not optimally low as compared to those obtained using the other parameter settings.
4. Applying the backpropagation and gradient descent algorithms with a SELU activation function, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 90.18404
5. The estimated classification accuracy using the backpropagation and gradient descent algorithms with a SELU activation function was optimal as compared to those obtained using the other parameter settings, also demonstrating a consistently smooth profile during the epoch training.


```python
##################################
# Creating a class object
# for the neural network algorithm
# using a SELU activation function
##################################
class NeuralNetwork_SELU:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.bias1 = np.zeros((1, hidden_size1))
        
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias2 = np.zeros((1, hidden_size2))
        
        self.weights3 = np.random.randn(hidden_size2, hidden_size3)
        self.bias3 = np.zeros((1, hidden_size3))
        
        self.weights4 = np.random.randn(hidden_size3, output_size)
        self.bias4 = np.zeros((1, output_size))
        
        self.gradients = {'dw1': [], 'db1': [], 'dw2': [], 'db2': [], 'dw3': [], 'db3': [], 'dw4': [], 'db4': []}
        self.losses = []
        self.accuracies = []
        
    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, x):
        self.hidden1 = self.selu(np.dot(x, self.weights1) + self.bias1)
        self.hidden2 = self.selu(np.dot(self.hidden1, self.weights2) + self.bias2)
        self.hidden3 = self.selu(np.dot(self.hidden2, self.weights3) + self.bias3)
        self.output = self.softmax(np.dot(self.hidden3, self.weights4) + self.bias4)
        
        return self.output
    
    def backward(self, x, y, lr):
        m = x.shape[0]
        
        # Computing output layer gradients
        output_error = self.output - y
        output_delta = output_error / m
        
        # Computing hidden layer 3 gradients
        hidden3_error = np.dot(output_delta, self.weights4.T)
        hidden3_delta = hidden3_error * (self.hidden3 > 0)
        
        # Computing hidden layer 2 gradients
        hidden2_error = np.dot(hidden3_delta, self.weights3.T)
        hidden2_delta = hidden2_error * (self.hidden2 > 0)
        
        # Computing hidden layer 1 gradients
        hidden1_error = np.dot(hidden2_delta, self.weights2.T)
        hidden1_delta = hidden1_error * (self.hidden1 > 0)
        
        # Updating weights and biases based on computed gradients
        self.weights4 -= lr * np.dot(self.hidden3.T, output_delta)
        self.bias4 -= lr * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights3 -= lr * np.dot(self.hidden2.T, hidden3_delta)
        self.bias3 -= lr * np.sum(hidden3_delta, axis=0, keepdims=True)
        
        self.weights2 -= lr * np.dot(self.hidden1.T, hidden2_delta)
        self.bias2 -= lr * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.weights1 -= lr * np.dot(x.T, hidden1_delta)
        self.bias1 -= lr * np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Storing computed gradients
        self.gradients['dw1'].append(np.mean(np.abs(hidden1_delta)))
        self.gradients['db1'].append(np.mean(np.abs(self.bias1)))
        self.gradients['dw2'].append(np.mean(np.abs(hidden2_delta)))
        self.gradients['db2'].append(np.mean(np.abs(self.bias2)))
        self.gradients['dw3'].append(np.mean(np.abs(hidden3_delta)))
        self.gradients['db3'].append(np.mean(np.abs(self.bias3)))
        self.gradients['dw4'].append(np.mean(np.abs(output_delta)))
        self.gradients['db4'].append(np.mean(np.abs(self.bias4)))
        
    def train(self, x, y, epochs, lr):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            loss = -np.mean(y * np.log(output))
            self.losses.append(loss)
            accuracy = self.accuracy(x, np.argmax(y, axis=1))
            self.accuracies.append(accuracy)
            if i % 100 == 0:
                print(f'Epoch {i}: Loss {loss}, Accuracy {accuracy}')
                
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def accuracy(self, x, y):
        pred = self.predict(x)
        return np.mean(pred == y)
```


```python
##################################
# Preparing the training data
##################################
X = matrix_x_values
y = y_values

##################################
# Performing a one-hot encoding
# of the target response labels
##################################
num_classes = 2
y_one_hot = np.eye(num_classes)[y]
```


```python
##################################
# Defining the neural network components
##################################
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
hidden_size3 = 4
output_size = 2
```


```python
##################################
# Initializing a neural network model object
# with SELU activation function
##################################
np.random.seed(88888)
nn_selu = NeuralNetwork_SELU(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
```


```python
##################################
# Training a neural network model
# with SELU activation function
##################################
nn_selu.train(X, y_one_hot, epochs=1001, lr=0.01)
```

    Epoch 0: Loss 1.0981323160350691, Accuracy 0.3006134969325153
    Epoch 100: Loss 0.1944859567995403, Accuracy 0.8343558282208589
    Epoch 200: Loss 0.15078198879619675, Accuracy 0.8773006134969326
    Epoch 300: Loss 0.13087303505062364, Accuracy 0.8773006134969326
    Epoch 400: Loss 0.11912606258714759, Accuracy 0.8773006134969326
    Epoch 500: Loss 0.111495190606054, Accuracy 0.8895705521472392
    Epoch 600: Loss 0.10641661771976144, Accuracy 0.901840490797546
    Epoch 700: Loss 0.1028003852939949, Accuracy 0.901840490797546
    Epoch 800: Loss 0.10017443966015926, Accuracy 0.901840490797546
    Epoch 900: Loss 0.09822970379617377, Accuracy 0.8957055214723927
    Epoch 1000: Loss 0.09674886443517193, Accuracy 0.901840490797546
    


```python
##################################
# Plotting the computed gradients
# of the neural network model
# with SELU activation function
##################################
plt.figure(figsize=(10, 6))
for key, value in nn_selu.gradients.items():
    plt.plot(value, label=key)
plt.title('SELU Activation: Gradients by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Gradients')
plt.ylim(-0.05, 0.50)
plt.xlim(-50,1000)
plt.legend(loc="upper left")
plt.show()
```


    
![png](output_215_0.png)
    



```python
##################################
# Plotting the computed cost
# of the neural network model
# with SELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_selu.losses)
plt.title('SELU Activation: Cost Function by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim(0.05, 0.70)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_216_0.png)
    



```python
##################################
# Plotting the computed accuracy
# of the neural network model
# with SELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_selu.accuracies)
plt.title('SELU Activation: Classification by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_217_0.png)
    



```python
##################################
# Gathering the final values for 
# accuracy and loss error
# using a SELU activation function
##################################
selu_metrics = pd.DataFrame(["ACCURACY","LOSS"])
selu_values = pd.DataFrame([nn_selu.accuracies[-1],nn_selu.losses[-1]])
selu_method = pd.DataFrame(["SELU"]*2)
selu_summary = pd.concat([selu_metrics,
                          selu_values,
                          selu_method], axis=1)
selu_summary.columns = ['Metric', 'Value', 'Method']
selu_summary.reset_index(inplace=True, drop=True)
display(selu_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.901840</td>
      <td>SELU</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.096749</td>
      <td>SELU</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.7 Randomized Leaky Rectified Linear Unit Activation Function <a class="anchor" id="1.6.7"></a>

[Backpropagation](https://link.springer.com/book/10.1007/978-0-387-84858-7) and [Weight Update](https://link.springer.com/book/10.1007/978-0-387-84858-7), in the context of an artificial neural network, involve the process of iteratively adjusting the weights of the connections between neurons in the network to minimize the difference between the predicted and the actual target responses. Input data is fed into the neural network, and it propagates through the network layer by layer, starting from the input layer, through hidden layers, and ending at the output layer. At each neuron, the weighted sum of inputs is calculated, followed by the application of an activation function to produce the neuron's output. Once the forward pass is complete, the network's output is compared to the actual target output. The difference between the predicted output and the actual output is quantified using a loss function, which measures the discrepancy between the predicted and actual values. Common loss functions for classification tasks include cross-entropy loss. During the backward pass, the error is propagated backward through the network to compute the gradients of the loss function with respect to each weight in the network. This is achieved using the chain rule of calculus, which allows the error to be decomposed and distributed backward through the network. The gradients quantify how much a change in each weight would affect the overall error of the network. Once the gradients are computed, the weights are updated in the opposite direction of the gradient to minimize the error. This update is typically performed using an optimization algorithm such as gradient descent, which adjusts the weights in proportion to their gradients and a learning rate hyperparameter. The learning rate determines the size of the step taken in the direction opposite to the gradient. These steps are repeated for multiple iterations (epochs) over the training data. As the training progresses, the weights are adjusted iteratively to minimize the error, leading to a neural network model that accurately classifies input data.

[Activation Functions](https://link.springer.com/book/10.1007/978-0-387-84858-7) play a crucial role in neural networks by introducing non-linearity into the network, enabling the model to learn complex patterns and relationships within the data. In the context of a neural network classification model, activation functions are applied to the output of each neuron in the hidden layers to introduce non-linear mappings between the input and output, allowing the network to approximate complex functions and make non-linear decisions. Activation functions are significant during model development by introducing non-linearity (without activation functions, the neural network would simply be a series of linear transformations, no matter how many layers it has. Activation functions introduce non-linearities to the model, enabling it to learn and represent complex patterns and relationships in the data); propagating back gradients (activation functions help in the backpropagation algorithm by providing gradients that indicate the direction and magnitude of adjustments to the weights during training. These gradients are necessary for optimizing the network's parameters through techniques like gradient descent}; and normalizing outputs (activation functions also help in normalizing the output of each neuron, ensuring that it falls within a specific range. This normalization prevents the activation values from becoming too large or too small, which can lead to numerical instability or saturation of gradients during training). The choice of activation function can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.

[Randomized Leaky Rectified Linear Unit Activation Function](https://link.springer.com/book/10.1007/978-0-387-84858-7) transforms a variable by using the given value of the variable when it is greater than zero, but multiplies this value with a random number obtained from a uniform distribution. The resulting output which ranges from negative infinity to positive infinity is similar to a Leaky RELU profile but the slope parameter is randomly initialized and updated during training. This process introduces randomness into the activations, acting as a form of regularization and reducing overfitting. This provides a trade-off between the benefits of RELU and Leaky RELU along with the regularization effect of randomness. This activation function is typically useful when dealing with overfitting or training large neural networks where regularization is necessary.

1. A neural network with the following structure was formulated:
    * <span style="color: #FF0000">Hidden Layer</span> = 3
    * <span style="color: #FF0000">Number of Nodes per Hidden Layer</span> = 4
2. The backpropagation and gradient descent algorithms were implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01
    * <span style="color: #FF0000">Epochs</span> = 1000
    * <span style="color: #FF0000">Activation Function</span> = Randomized Leaky Rectified Linear Unit (RReLU)
3. The final loss estimate determined as 0.09765 at the 1000th epoch was not optimally low as compared to those obtained using the other parameter settings.
4. Applying the backpropagation and gradient descent algorithms with a RReLU activation function, the neural network model performance is estimated as follows:
    * <span style="color: #FF0000">Accuracy</span> = 92.63803
5. The estimated classification accuracy using the backpropagation and gradient descent algorithms with a RReLU activation function was optimal as compared to those obtained using the other parameter settings, also demonstrating a consistently smooth profile during the epoch training.


```python
##################################
# Creating a class object
# for the neural network algorithm
# using a RRELU activation function
##################################
class NeuralNetwork_RRELU:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.bias1 = np.zeros((1, hidden_size1))
        
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias2 = np.zeros((1, hidden_size2))
        
        self.weights3 = np.random.randn(hidden_size2, hidden_size3)
        self.bias3 = np.zeros((1, hidden_size3))
        
        self.weights4 = np.random.randn(hidden_size3, output_size)
        self.bias4 = np.zeros((1, output_size))
        
        self.gradients = {'dw1': [], 'db1': [], 'dw2': [], 'db2': [], 'dw3': [], 'db3': [], 'dw4': [], 'db4': []}
        self.losses = []
        self.accuracies = []
        
    def randomized_leaky_relu(self, x):
        alpha = np.random.uniform(0.01, 0.1)  # Randomly generate alpha in the range [0.01, 0.1]
        return np.where(x > 0, x, alpha * x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, x):
        self.hidden1 = self.randomized_leaky_relu(np.dot(x, self.weights1) + self.bias1)
        self.hidden2 = self.randomized_leaky_relu(np.dot(self.hidden1, self.weights2) + self.bias2)
        self.hidden3 = self.randomized_leaky_relu(np.dot(self.hidden2, self.weights3) + self.bias3)
        self.output = self.softmax(np.dot(self.hidden3, self.weights4) + self.bias4)
        
        return self.output
    
    def backward(self, x, y, lr):
        m = x.shape[0]
        
        # Computing output layer gradients
        output_error = self.output - y
        output_delta = output_error / m
        
        # Computing hidden layer 3 gradients
        hidden3_error = np.dot(output_delta, self.weights4.T)
        hidden3_delta = hidden3_error * (self.hidden3 > 0) + 0.01 * hidden3_error * (self.hidden3 <= 0)
        
        # Computing hidden layer 2 gradients
        hidden2_error = np.dot(hidden3_delta, self.weights3.T)
        hidden2_delta = hidden2_error * (self.hidden2 > 0) + 0.01 * hidden2_error * (self.hidden2 <= 0)
        
        # Computing hidden layer 1 gradients
        hidden1_error = np.dot(hidden2_delta, self.weights2.T)
        hidden1_delta = hidden1_error * (self.hidden1 > 0) + 0.01 * hidden1_error * (self.hidden1 <= 0)
        
        # Updating weights and biases based on computed gradients
        self.weights4 -= lr * np.dot(self.hidden3.T, output_delta)
        self.bias4 -= lr * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights3 -= lr * np.dot(self.hidden2.T, hidden3_delta)
        self.bias3 -= lr * np.sum(hidden3_delta, axis=0, keepdims=True)
        
        self.weights2 -= lr * np.dot(self.hidden1.T, hidden2_delta)
        self.bias2 -= lr * np.sum(hidden2_delta, axis=0, keepdims=True)
        
        self.weights1 -= lr * np.dot(x.T, hidden1_delta)
        self.bias1 -= lr * np.sum(hidden1_delta, axis=0, keepdims=True)
        
        # Storing computed gradients
        self.gradients['dw1'].append(np.mean(np.abs(hidden1_delta)))
        self.gradients['db1'].append(np.mean(np.abs(self.bias1)))
        self.gradients['dw2'].append(np.mean(np.abs(hidden2_delta)))
        self.gradients['db2'].append(np.mean(np.abs(self.bias2)))
        self.gradients['dw3'].append(np.mean(np.abs(hidden3_delta)))
        self.gradients['db3'].append(np.mean(np.abs(self.bias3)))
        self.gradients['dw4'].append(np.mean(np.abs(output_delta)))
        self.gradients['db4'].append(np.mean(np.abs(self.bias4)))
        
    def train(self, x, y, epochs, lr):
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            loss = -np.mean(y * np.log(output))
            self.losses.append(loss)
            accuracy = self.accuracy(x, np.argmax(y, axis=1))
            self.accuracies.append(accuracy)
            if i % 100 == 0:
                print(f'Epoch {i}: Loss {loss}, Accuracy {accuracy}')
                
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
    
    def accuracy(self, x, y):
        pred = self.predict(x)
        return np.mean(pred == y)
    
```


```python
##################################
# Preparing the training data
##################################
X = matrix_x_values
y = y_values

##################################
# Performing a one-hot encoding
# of the target response labels
##################################
num_classes = 2
y_one_hot = np.eye(num_classes)[y]
```


```python
##################################
# Defining the neural network components
##################################
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
hidden_size3 = 4
output_size = 2
```


```python
##################################
# Initializing a neural network model object
# with RRELU activation function
##################################
np.random.seed(88888)
nn_rrelu = NeuralNetwork_RRELU(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
```


```python
##################################
# Training a neural network model
# with RRELU activation function
##################################
nn_rrelu.train(X, y_one_hot, epochs=1001, lr=0.01)
```

    Epoch 0: Loss 0.42536277001068584, Accuracy 0.3496932515337423
    Epoch 100: Loss 0.17225380216691843, Accuracy 0.901840490797546
    Epoch 200: Loss 0.13053451403709007, Accuracy 0.9079754601226994
    Epoch 300: Loss 0.11370320839132757, Accuracy 0.8957055214723927
    Epoch 400: Loss 0.10982347564770703, Accuracy 0.901840490797546
    Epoch 500: Loss 0.10483990809689303, Accuracy 0.9263803680981595
    Epoch 600: Loss 0.1015575263421128, Accuracy 0.9325153374233128
    Epoch 700: Loss 0.09951830751495384, Accuracy 0.9202453987730062
    Epoch 800: Loss 0.09799491669757471, Accuracy 0.9325153374233128
    Epoch 900: Loss 0.09839336828285061, Accuracy 0.9263803680981595
    Epoch 1000: Loss 0.09765980921693111, Accuracy 0.9263803680981595
    


```python
##################################
# Plotting the computed gradients
# of the neural network model
# with RRELU activation function
##################################
plt.figure(figsize=(10, 6))
for key, value in nn_rrelu.gradients.items():
    plt.plot(value, label=key)
plt.title('RRELU Activation: Gradients by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Gradients')
plt.ylim(-0.05, 0.50)
plt.xlim(-50,1000)
plt.legend(loc="upper left")
plt.show()
```


    
![png](output_225_0.png)
    



```python
##################################
# Plotting the computed cost
# of the neural network model
# with RRELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_rrelu.losses)
plt.title('RRELU Activation: Cost Function by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.ylim(0.05, 0.70)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_226_0.png)
    



```python
##################################
# Plotting the computed accuracy
# of the neural network model
# with RRELU activation function
##################################
plt.figure(figsize=(10, 6))
plt.plot(nn_rrelu.accuracies)
plt.title('RRELU Activation: Classification by Iteration')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim(0.00, 1.00)
plt.xlim(-50,1000)
plt.show()
```


    
![png](output_227_0.png)
    



```python
##################################
# Gathering the final values for 
# accuracy and loss error
# using a RRELU activation function
##################################
rrelu_metrics = pd.DataFrame(["ACCURACY","LOSS"])
rrelu_values = pd.DataFrame([nn_rrelu.accuracies[-1],nn_rrelu.losses[-1]])
rrelu_method = pd.DataFrame(["RRELU"]*2)
rrelu_summary = pd.concat([rrelu_metrics,
                           rrelu_values,
                           rrelu_method], axis=1)
rrelu_summary.columns = ['Metric', 'Value', 'Method']
rrelu_summary.reset_index(inplace=True, drop=True)
display(rrelu_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.92638</td>
      <td>RRELU</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.09766</td>
      <td>RRELU</td>
    </tr>
  </tbody>
</table>
</div>


## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>

1. This activation function showed the presence of the vanishing gradients problem resulting to a relatively higher loss and lower classification accuracy:
    * <span style="color: #FF0000">Sigmoid</span> = Sigmoid Activation Function
2. These activation functions demonstrated the absence of the vanishing gradients problem resulting to a sufficiently low loss and higher classification accuracy:
    * <span style="color: #FF0000">RELU</span> = Rectified Linear Unit Activation Function
    * <span style="color: #FF0000">Leaky_RELU</span> = Leaky Rectified Linear Unit Activation Function
    * <span style="color: #FF0000">ELU</span> = Exponential Linear Unit Activation Function
    * <span style="color: #FF0000">SELU</span> = Scaled Exponential Linear Unit Activation Function
    * <span style="color: #FF0000">RRELU</span> = Randomized Leaky Rectified Linear Unit Activation Function                 
3. The choice of **Activation Function** can significantly impact the performance and training dynamics of a neural network classification model, making it an important consideration during model development. Different activation functions have different properties, and selecting the appropriate one depends on factors such as the nature of the problem, the characteristics of the data, and the desired behavior of the network.



```python
##################################
# Consolidating all the
# model performance metrics
##################################
model_performance_comparison = pd.concat([sigmoid_summary, 
                                          relu_summary,
                                          leakyrelu_summary, 
                                          elu_summary,
                                          selu_summary, 
                                          rrelu_summary], 
                                         ignore_index=True)
print('Neural Network Model Comparison: ')
display(model_performance_comparison)
```

    Neural Network Model Comparison: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.748466</td>
      <td>Sigmoid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.278053</td>
      <td>Sigmoid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACCURACY</td>
      <td>0.932515</td>
      <td>RELU</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LOSS</td>
      <td>0.096237</td>
      <td>RELU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>Leaky_RELU</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LOSS</td>
      <td>0.093777</td>
      <td>Leaky_RELU</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ELU</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LOSS</td>
      <td>0.094432</td>
      <td>ELU</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ACCURACY</td>
      <td>0.901840</td>
      <td>SELU</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LOSS</td>
      <td>0.096749</td>
      <td>SELU</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>RRELU</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LOSS</td>
      <td>0.097660</td>
      <td>RRELU</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating the values for the
# accuracy metrics
# for all models
##################################
model_performance_comparison_accuracy = model_performance_comparison[model_performance_comparison['Metric']=='ACCURACY']
model_performance_comparison_accuracy.reset_index(inplace=True, drop=True)
model_performance_comparison_accuracy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACCURACY</td>
      <td>0.748466</td>
      <td>Sigmoid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACCURACY</td>
      <td>0.932515</td>
      <td>RELU</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>Leaky_RELU</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>ELU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACCURACY</td>
      <td>0.901840</td>
      <td>SELU</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ACCURACY</td>
      <td>0.926380</td>
      <td>RRELU</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the values for the
# accuracy metrics
# for all models
##################################
fig, ax = plt.subplots(figsize=(7, 7))
accuracy_hbar = ax.barh(model_performance_comparison_accuracy['Method'], model_performance_comparison_accuracy['Value'])
ax.set_xlabel("Accuracy")
ax.set_ylabel("Neural Network Classification Models")
ax.bar_label(accuracy_hbar, fmt='%.5f', padding=-50, color='white', fontweight='bold')
ax.set_xlim(0,1)
plt.show()
```


    
![png](output_232_0.png)
    



```python
##################################
# Consolidating the values for the
# logarithmic loss error metrics
# for all models
##################################
model_performance_comparison_loss = model_performance_comparison[model_performance_comparison['Metric']=='LOSS']
model_performance_comparison_loss.reset_index(inplace=True, drop=True)
model_performance_comparison_loss
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metric</th>
      <th>Value</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LOSS</td>
      <td>0.278053</td>
      <td>Sigmoid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LOSS</td>
      <td>0.096237</td>
      <td>RELU</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LOSS</td>
      <td>0.093777</td>
      <td>Leaky_RELU</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LOSS</td>
      <td>0.094432</td>
      <td>ELU</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LOSS</td>
      <td>0.096749</td>
      <td>SELU</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LOSS</td>
      <td>0.097660</td>
      <td>RRELU</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Plotting the values for the
# loss error
# for all models
##################################
fig, ax = plt.subplots(figsize=(7, 7))
loss_hbar = ax.barh(model_performance_comparison_loss['Method'], model_performance_comparison_loss['Value'])
ax.set_xlabel("Loss Error")
ax.set_ylabel("Neural Network Classification Models")
ax.bar_label(loss_hbar, fmt='%.5f', padding=-50, color='white', fontweight='bold')
ax.set_xlim(0,0.40)
plt.show()
```


    
![png](output_234_0.png)
    


# 2. Summary <a class="anchor" id="Summary"></a>


![Project48_Summary.png](e7e9bf01-4de8-4281-954d-d3ed0c180dce.png)

# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Deep Learning: A Visual Approach](https://nostarch.com/deep-learning-visual-approach) by Andrew Glassner
* **[Book]** [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) by François Chollet
* **[Book]** [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7) by Trevor Hastie, Robert Tibshirani and Jerome Friedman
* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Book]** [Regression Modeling Strategies](https://link.springer.com/book/10.1007/978-1-4757-3462-1) by Frank Harrell
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Article]** [Exploratory Data Analysis in Python — A Step-by-Step Process](https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee) by Andrea D'Agostino (Towards Data Science)
* **[Article]** [Exploratory Data Analysis with Python](https://medium.com/@douglas.rochedo/exploratory-data-analysis-with-python-78b6c1d479cc) by Douglas Rocha (Medium)
* **[Article]** [4 Ways to Automate Exploratory Data Analysis (EDA) in Python](https://builtin.com/data-science/EDA-python) by Abdishakur Hassan (BuiltIn)
* **[Article]** [10 Things To Do When Conducting Your Exploratory Data Analysis (EDA)](https://www.analyticsvidhya.com) by Alifia Harmadi (Medium)
* **[Article]** [How to Handle Missing Data with Python](https://machinelearningmastery.com/handle-missing-data-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Imputing Missing Data with Simple and Advanced Techniques](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87) by Idil Ismiguzel (Towards Data Science)
* **[Article]** [Missing Data Imputation Approaches | How to handle missing values in Python](https://www.machinelearningplus.com/machine-learning/missing-data-imputation-how-to-handle-missing-values-in-python/) by Selva Prabhakaran (Machine Learning +)
* **[Article]** [Master The Skills Of Missing Data Imputation Techniques In Python(2022) And Be Successful](https://medium.com/analytics-vidhya/a-quick-guide-on-missing-data-imputation-techniques-in-python-2020-5410f3df1c1e) by Mrinal Walia (Analytics Vidhya)
* **[Article]** [How to Preprocess Data in Python](https://builtin.com/machine-learning/how-to-preprocess-data-python) by Afroz Chakure (BuiltIn)
* **[Article]** [Easy Guide To Data Preprocessing In Python](https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html) by Ahmad Anis (KDNuggets)
* **[Article]** [Data Preprocessing in Python](https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5) by Tarun Gupta (Towards Data Science)
* **[Article]** [Data Preprocessing using Python](https://medium.com/@suneet.bhopal/data-preprocessing-using-python-1bfee9268fb3) by Suneet Jain (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.com/@abonia/data-preprocessing-in-python-1f90d95d44f4) by Abonia Sojasingarayar (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.datadriveninvestor.com/data-preprocessing-3cd01eefd438) by Afroz Chakure (Medium)
* **[Article]** [Detecting and Treating Outliers | Treating the Odd One Out!](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/) by Harika Bonthu (Analytics Vidhya)
* **[Article]** [Outlier Treatment with Python](https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02) by Sangita Yemulwar (Analytics Vidhya)
* **[Article]** [A Guide to Outlier Detection in Python](https://builtin.com/data-science/outlier-detection-python) by Sadrach Pierre (BuiltIn)
* **[Article]** [How To Find Outliers in Data Using Python (and How To Handle Them)](https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/) by Eric Kleppen (Career Foundry)
* **[Article]** [Statistics in Python — Collinearity and Multicollinearity](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f) by Wei-Meng Lee (Towards Data Science)
* **[Article]** [Understanding Multicollinearity and How to Detect it in Python](https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc) by Terence Shin (Towards Data Science)
* **[Article]** [A Python Library to Remove Collinearity](https://www.yourdatateacher.com/2021/06/28/a-python-library-to-remove-collinearity/) by Gianluca Malato (Your Data Teacher)
* **[Article]** [8 Best Data Transformation in Pandas](https://ai.plainenglish.io/data-transformation-in-pandas-29b2b3c61b34) by Tirendaz AI (Medium)
* **[Article]** [Data Transformation Techniques with Python: Elevate Your Data Game!](https://medium.com/@siddharthverma.er.cse/data-transformation-techniques-with-python-elevate-your-data-game-21fcc7442cc2) by Siddharth Verma (Medium)
* **[Article]** [Data Scaling with Python](https://www.kdnuggets.com/2023/07/data-scaling-python.html) by Benjamin Obi Tayo (KDNuggets)
* **[Article]** [How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Feature Engineering: Scaling, Normalization, and Standardization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/) by Aniruddha Bhandari  (Analytics Vidhya)
* **[Article]** [How to Normalize Data Using scikit-learn in Python](https://www.digitalocean.com/community/tutorials/normalize-data-in-python) by Jayant Verma (Digital Ocean)
* **[Article]** [What are Categorical Data Encoding Methods | Binary Encoding](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/) by Shipra Saxena  (Analytics Vidhya)
* **[Article]** [Guide to Encoding Categorical Values in Python](https://pbpython.com/categorical-encoding.html) by Chris Moffitt (Practical Business Python)
* **[Article]** [Categorical Data Encoding Techniques in Python: A Complete Guide](https://soumenatta.medium.com/categorical-data-encoding-techniques-in-python-a-complete-guide-a913aae19a22) by Soumen Atta (Medium)
* **[Article]** [Categorical Feature Encoding Techniques](https://towardsdatascience.com/categorical-encoding-techniques-93ebd18e1f24) by Tara Boyle (Medium)
* **[Article]** [Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Hypothesis Testing with Python: Step by Step Hands-On Tutorial with Practical Examples](https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e) by Ece Işık Polat (Towards Data Science)
* **[Article]** [17 Statistical Hypothesis Tests in Python (Cheat Sheet)](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [A Step-by-Step Guide to Hypothesis Testing in Python using Scipy](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Gabriel Rennó (Medium)
* **[Article]** [How Does Backpropagation in a Neural Network Work?](https://builtin.com/machine-learning/backpropagation-neural-network) by Anas Al-Masri (Builtin)
* **[Article]** [A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/) by Matt Mazur (MattMazur.Com)
* **[Article]** [Understanding Backpropagation](https://towardsdatascience.com/understanding-backpropagation-abcc509ca9d0) by Brent Scarff (Towards Data Science)
* **[Article]** [Understanding Backpropagation Algorithm](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd) by Simeon Kostadinov (Towards Data Science)
* **[Article]** [A Comprehensive Guide to the Backpropagation Algorithm in Neural Networks](https://neptune.ai/blog/backpropagation-algorithm-in-neural-networks-guide) by Ahmed Gad (Neptune.AI)
* **[Article]** [Backpropagation](https://brilliant.org/wiki/backpropagation/) by John McGonagle, George Shaikouski and Christopher Williams (Brilliant)
* **[Article]** [Backpropagation in Neural Networks](https://serokell.io/blog/understanding-backpropagation) by Inna Logunova (Serokell.IO)
* **[Article]** [Backpropagation Concept Explained in 5 Levels of Difficulty](https://medium.com/coinmonks/backpropagation-concept-explained-in-5-levels-of-difficulty-8b220a939db5) by Devashish Sood (Medium)
* **[Article]** [BackProp Explainer](https://xnought.github.io/backprop-explainer/) by Donny Bertucci (GitHub)
* **[Article]** [Backpropagation Algorithm in Neural Network and Machine Learning](https://intellipaat.com/blog/tutorial/artificial-intelligence-tutorial/back-propagation-algorithm/) by Intellipaat Team
* **[Article]** [Understanding Backpropagation in Neural Networks](https://ai.plainenglish.io/understanding-backpropagation-in-neural-networks-3634aad3a3c4) by Tech-AI-Math Team
* **[Article]** [Backpropagation Neural Network using Python](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Avinash Navlani (Machine Learning Geek)
* **[Article]** [Back Propagation in Neural Network: Machine Learning Algorithm](https://www.guru99.com/backpropogation-neural-network.html) by Daniel Johnson (Guru99)
* **[Article]** [What is Backpropagation?](https://deepai.org/machine-learning-glossary-and-terms/backpropagation) by Thomas Wood (DeepAI.Org)
* **[Article]** [Activation Functions in Neural Networks [12 Types & Use Cases]](https://www.v7labs.com/blog/neural-networks-activation-functions) by Pragati Baheti (V7.Com)
* **[Article]** [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) by Sagar Sharma (Towards Data Science)
* **[Article]** [Comparison of Sigmoid, Tanh and ReLU Activation Functions](https://www.aitude.com/comparison-of-sigmoid-tanh-and-relu-activation-functions/) by Sandeep Kumar (AItude.Com)
* **[Article]** [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Choosing the Right Activation Function in Deep Learning: A Practical Overview and Comparison](https://python.plainenglish.io/choosing-the-right-activation-function-in-deep-learning-a-practical-overview-and-comparison-c5372fe167f8) by Okan Yenigün (Medium)
* **[Article]** [Activation Functions in Neural Networks](https://www.geeksforgeeks.org/activation-functions-neural-networks/) by Geeks For Geeks Team
* **[Article]** [A Practical Comparison of Activation Functions](https://becominghuman.ai/a-practical-comparison-of-activation-functions-6f348be7d274) by Danny Denenberg (Medium)
* **[Article]** [Activation Functions in Neural Networks: With 15 examples](https://encord.com/blog/activation-functions-neural-networks/) by Nikolaj Buhl (Encord.Com)
* **[Article]** [Activation functions used in Neural Networks - Which is Better?](https://anishsinghwalia.medium.com/activation-functions-used-in-neural-networks-which-is-better-d4817d91199a) by Anish Singh Walia (Medium)
* **[Article]** [6 Types of Activation Function in Neural Networks You Need to Know](https://www.upgrad.com/blog/types-of-activation-function-in-neural-networks/) by Kechit Goyal (UpGrad.Com)
* **[Article]** [Activation Functions in Neural Networks](https://www.superannotate.com/blog/activation-functions-in-neural-networks) by SuperAnnotate Team
* **[Article]** [Compare Activation Layers](https://www.mathworks.com/help/deeplearning/ug/compare-activation-layers.html) by MathWorks Team
* **[Article]** [Activation Functions In Neural Networks](https://www.comet.com/site/blog/activation-functions-in-neural-networks/) by Kurtis Pykes (Comet.Com)
* **[Article]** [ReLU vs. Sigmoid Function in Deep Neural Networks](https://wandb.ai/ayush-thakur/dl-question-bank/reports/ReLU-vs-Sigmoid-Function-in-Deep-Neural-Networks--VmlldzoyMDk0MzI) by Ayush Thakur (Wanb.AI)
* **[Article]** [Using Activation Functions in Neural Networks](https://machinelearningmastery.com/using-activation-functions-in-neural-networks/) by Jason Bronwlee (Machine Learning Mastery)
* **[Article]** [Activation Function: Top 9 Most Popular Explained & When To Use Them](https://spotintelligence.com/2023/06/16/activation-function/) by Neri Van Otten (SpotIntelligence.Com)
* **[Article]** [5 Deep Learning and Neural Network Activation Functions to Know](https://builtin.com/machine-learning/activation-functions-deep-learning) by Artem Oppermann (BuiltIn.Com)
* **[Article]** [Activation Functions in Deep Learning: Sigmoid, tanh, ReLU](https://artemoppermann.com/activation-functions-in-deep-learning-sigmoid-tanh-relu/) by Artem Oppermann
* **[Article]** [7 Types of Activation Functions in Neural Network](https://www.analyticssteps.com/blogs/7-types-activation-functions-neural-network) by Dinesh Kumawat (AnalyticsSteps.Com)
* **[Article]** [What is an Activation Function? A Complete Guide](https://blog.roboflow.com/activation-function-computer-vision/#:~:text=Petru%20Potrimba,of%20deep%20learning%20model%20outputs.) by Petru Potrimba (RoboFlow.Com)
* **[Publication]** [Data Quality for Machine Learning Tasks](https://journals.sagepub.com/doi/10.1177/0962280206074463) by Nitin Gupta, Shashank Mujumdar, Hima Patel, Satoshi Masuda, Naveen Panwar, Sambaran Bandyopadhyay, Sameep Mehta, Shanmukha Guttula, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’21: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Overview and Importance of Data Quality for Machine Learning Tasks](https://dl.acm.org/doi/10.1145/3394486.3406477) by Abhinav Jain, Hima Patel, Lokesh Nagalapatti, Nitin Gupta, Sameep Mehta, Shanmukha Guttula, Shashank Mujumdar, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’20: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Multiple Imputation of Discrete and Continuous Data by Fully Conditional Specification](https://journals.sagepub.com/doi/10.1177/0962280206074463) by Stef van Buuren (Statistical Methods in Medical Research)
* **[Publication]** [Mathematical Contributions to the Theory of Evolution: Regression, Heredity and Panmixia](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) by Karl Pearson (Royal Society)
* **[Publication]** [A New Family of Power Transformations to Improve Normality or Symmetry](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) by In-Kwon Yeo and Richard Johnson (Biometrika)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)
* **[Course]** [Machine Learning Specialization Certificate](https://www.coursera.org/specializations/machine-learning-introduction) by DeepLearning.AI Team (Coursera)

***


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

