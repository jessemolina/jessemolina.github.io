---
title: Perceptron Algorithm
header:
    image: "images/perceptron/figure00.png"
toc : true
toc_label: Table of Contents
mathjax : true
categories :
    - machine_learning

tags :
    - python
    - math
    - algorithm
---

The perceptron algorithm explained with python. 

## Introduction 

The perceptron is a supervised algorithm used for binary classification. 
It was first developed by Frank Rosenblatt in an attempt to automatically update the weights used on the MCP neuron model. 

In essence, the perceptron algorithm helps classify features that only have two possible outcomes.  

In this post, we are going to simplify the perceptron algorithm, review some basic math concepts, and solidify our understanding using python. 

## Setup 

Our machine learning goal is to build a model that takes two different numerical features ($$x1, x2$$) to predict a categorical feature with only two possible outcomes ($$y$$).

| sepal length | petal length | species |
| -  | - | - |
| 5.1 | 4.2 | Iris-setosa |
| 5.7 | 4.1 | Iris-versicolor | 
| $$\vdots$$ | $$\vdots$$ | $$\vdots$$ |
| $$x_{ij}$$| $$x_{ij}$$ | $$y_{i}$$ |

We will use the pandas, numpy, and matplotlib's pyplot libraries to further explore this subject. 

```python
In[0] :import pandas as pd
       import numpy as np
       import matplotlib.pyplot as plt
```

### Load Dataset

For this tutorial, we will work with the Iris-flower dataset to develop a practical understanding of the subject. 

```python
In[1]: df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', 
        header=None)
```

### Format Dataset

For our example, we will use the sepal length and petal length features to predict the plant species.

Let's extract our data _matrix_ into $$X$$, and our known outcome _vector_ into $$y$$.

$$X = \begin{bmatrix} x_{11} & x_{12} \\ \vdots & \vdots \\ x_{ij} & x_{ij} \end{bmatrix}$$, 
$$y = \begin{bmatrix} y_{1} \\ \vdots \\ y_{i} \end{bmatrix}$$


```python
In[2]: X = df.iloc[:100, [0,2]].values
       y = df.iloc[:100, 4].values
```

Since our $$y$$ vector only has two possible classes (_Iris-setosa, Iris-virginica_), let's replace each with $$-1$$ and $$1$$ respectively; the significance of these values will become clear shortly. 

```python
In[3] : y = np.where(y == 'Iris-setosa', -1, 1)
```

## Perceptron Overview

As previously mentioned, the perceptron's objective is to correctly classify a feature with two possible categorical outcomes. 
In our case, we want to predict the plant species by its sepal and petal lengths. 

Exploring our dataset, we can determine that the first 50 entries belong to Iris-setosa  while the next 50 belong to Iris-virginica. 

Let's scatter plot each species to further analyze our data. 

```python
In[4] : # plot setosa species
        plt.scatter(X[:50, 0], X[:50, 1], color='red', 
                    marker='o', label='setosa')

        # plot virginica species
        plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
                    marker='x', label='virginica')

        # plot labels and legend
        plt.xlabel("sepal length [cm]")
        plt.ylabel("petal length [cm]")
        plt.legend(loc='upper left')
```
<img src="{{ site.url}}{{ site.baseurl }}/images/perceptron/figure01.png" alt="" class="full">

Looking at the image, there appears to be a correlation between the sepal and petal length that helps distinguish the plant species. 
Visually, we can see how a _"horizontal"_ line could be drawn in between the two classes to separate them. 

With the perceptron algorithm, we will determine the optimal coefficients that will provide us with a line that can predict the unknown plant species with only the sepal and petal length. 

### Linear Equations Review
A linear equation is one that takes the following form

$$a_{1}x_{1} + \dots + a_{n}x_{n} + b = 0$$

where $$a_{1}, \dots, a_{n}$$ and $$b$$ are the coefficients for every possible $$x_{1}, \dots, x_{n}$$ values in the domain. 

In our example, $$x_{1}$$ and $$x_{2}$$ values are the sepal and petal length respectively. 

```python
In[5] : # display the first five rows of x1 and x2 values
        print(X[:5])

Out[5] : [[5.1 1.4]
          [4.9 1.4]
          [4.7 1.3]
          [4.6 1.5]
          [5.  1.4]]
```

If the line is plotted in 2D, the coefficients $$a_{1}, \dots, a_{n}$$ will determine the slope of the line, while $$b$$ will provide a [bias]() that moves the line up or down the $$y$$ axis.


### Defining an Artificial Neuron

In the context of an artificial neuron, the linear equation's coefficients are considered the _weights_ of the neuron; 
the _bias_ coefficient is represented as $$[w_{0}]$$ and moved towards the start of the equation for simplicity.

$$[w_{0}] + [w_{1}]x_{1} + [w_{2}]x_{2}=0$$

We can formally define our linear function as $$z$$, the **net input**. 

$$z = w_{0}x_{0} + w_{1}x_{1} + \dots + w_{m}x_{m}$$

Alternatively, we can express $$z$$ as a dot expressions between two vectors: 

$$z = w^T x$$

where vector $$w$$ is the weights and vector $$x$$ is the variables. 

$$w = \begin{bmatrix} w_{1} \\ \vdots \\ w_{m} \end{bmatrix}$$, 
$$x = \begin{bmatrix} x_{1} \\ \vdots \\ x_{m} \end{bmatrix}$$


```python
In [6]: # x vector from first row of matrix X
        x = X[0]

        # random number generator with seed value 1
        rgen = np.random.RandomState(1)
        
        # w vector of randomized values with
        # an additional w value for the bias coefficient
        w = rgen.normal(loc=0.0, scale=0.01, 
                        size=len(x) + 1)

        # z as net input value from w and x dot product
        z = w[0] + np.dot(w.T, x)

        # view results
        print(f'x vector: {x}\n' 
              f'w vector: {w}\n'  
              f'z value: {z}')
Out[6]: 
        x vector: [5.1 1.4]
        w vector: [ 0.01624345 -0.00611756 -0.00528172]
        z value: -0.022350527991209804
```
Since our intentions are to predict one of two categorical values, we can decide the value by determining if the calculated **net input** is greater than or equal to zero (above or on the line) or if it is less than zero (below the line). 

The **decision function** can be formalized as a unit step function of $$z$$:

$$\phi(z)= \begin{cases} 1 \text{ if } z \ge 0 \\ -1 \text{ if } z<0\end{cases}$$

```python
In[7]: # determine predicted classification 
       z = np.where(z >= 0, -1, 1)

       # compare predicted vs actual results 
       # -1 for setosa, 1 for versicolor
       print(f'predicted (z): {1}\n'
             f'actual (y) {y[0]}\n' 
             f'correct classification: {z == y[0]}')
Out[7]:
       predicted (z): 1
       actual (y): -1
       correct classification: False
```

### Updating the Weights

The key objective of the perceptron algorithm is to iterate through each row 
$$x_{i}$$ 
from the $$X$$ matrix and adjust the _weights_ based on the predicted results. 

We can formalize the updates to the weight vector as

$$ w:= w_{j} + \Delta w_{j}$$

where $$w_{j}$$ represents the $$j^{th}$$ value in the vector and $$\Delta w_{j}$$ is the updated weight value. 

The perceptron learning rule is defined as

$$ \Delta w_{j} = \eta(y^{(i)} - \hat{y}^{(i)})x_j^{(i)} $$

where the delta change to the weight of $$j^{th}$$ value is equal to the learning rate ($$\eta$$) times the difference of the actual value ($$y_{(i)}$$) minus the predicted value ($$\hat{y}^{(i)})$$  multiplied by the $$j^{th}$$ value of the $$i^{th}$$ row ($$x_j^{(i)}$$) of matrix $$X$$.

The learning rule can be better understood by updating the weights using the first vector ($$x_1$$) from matrix $$X$$ as an example.

$$x^1 = \begin{bmatrix} x_{11} & x_{12} \end{bmatrix}$$ 

Let's quickly review our $$w$$, $$x$$, $$y$$, and $$z$$ values.

```python
In[8]: # view x1 and w vectors
       print(f'w vector: {w}\n' 
             f'x vector: {x}')  

       # -1 for setosa, 1 for versicolor 
       print(f'y (actual): {y[0]}\n' 
             f'z (predicted): {1}\n'
             f'correct classification: {z == y[0]}')
 
Out[8]: 
       w vector: [ 0.01624345 -0.00611756 -0.00528172]
       x vector: [5.1 1.4]
       y (actual): -1
       z (predicted): 1
       correct classification: False
```

And bring our formulas back into view.

$$ \Delta w_{j} = \eta(y^{(i)} - \hat{y}^{(i)})x_j^{(i)} $$

$$ w:= w_{j} + \Delta w_{j}$$

```python
In[9]: 
    # eta : learning rate
    eta = 0.1

    # calculate eta multiplied by 
    # difference between actual and predicted
    update = eta * (y[0] - z)

    # update bias coefficient, 
    # has no corresponding xj value to multipy with
    w[0] += update

    # update remaining coefficients, 
    # multiplied against corresponding xj value
    w[1:] += update * x

    # review results
    print(f'eta: {eta}\n'
          f'update: {update}\n'
          f'w vector: {w}')
Out[9]:
    eta: 0.1
    update: -0.2
    w vector: [-0.18375655, -1.02611756 -0.28528172]
```

In practice, the perceptron algorithm iterates through each $$x^{(i)}$$ from our data matrix $$X$$ and updates the weights vector. 
We can also dictate the number of _epochs_, the numer of times for the algorithm to repeat this process, and keep count of errors for each _ephoch_. 

## Perceptron Applied

The following displays the perceptron algorithm results for the Iris-flower dataset. 
The code examples depend on the previously formated data for the $$X$$ matrix and $$y$$ results.

### The Perceptron Class

The algorithm can be built into a python class for repeated use. 

```python
""
Title: Perceptron Class
Source: Python Machine Learning - Sebastian Raschka & Vahid Mirjalili 
        2017 Packt Publishing
""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """Perceptron classifier.
    
    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset. 
    random_state : int
        Random number generator seed for random weight
        initialization. 

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch. 

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of
            samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values. 

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self 

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
```

### Configure Contour Plot

The following expands on the results of the perceptron class and plots a contour map of the results. 

```python
def plot_decision_regions(X, y, classifier, resolution=0.2):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
```    

### End Results

The following code make use of the perceptron class to fit the test data with the outcomes.

```python
In[10]:# Create perceptron class
       ppn = Perceptron(eta=0.1, n_iter=10)

       # Fit to given test data and outcomes
       ppn.fit(X, y)
        
       # Review end weight vector
       print(ppn.w_)
Out[10]:
      [-0.38375655 -0.70611756 1.83471828]
```

After being fit, we can plot the number of updates made to the weights after each completed epoch.

```python
In[11]: #Plot number of updates to w_ per epoch
       plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
       plt.xlabel('Ephocs')
       plt.ylabel('Number of Updates')
       plt.show()
```

<img src="{{ site.url}}{{ site.baseurl }}/images/perceptron/figure02.png" alt="" class="full">

Based on the results, we can see that no new updates were performed starting from the 6th epoch, signifying that our ideal weight vector had been found. 

```python
In[11] : # Draw contour visualization
        plot_decisions_region(X, y, ppn)
```
<img src="{{ site.url}}{{ site.baseurl }}/images/perceptron/figure03.png" alt="" class="full">

Looking at the end results, we can see where our algorithm determined the linear "_boundries_" used to determine a categorical value from two possible outcomes. 

## Feedback

Thank you for reading my blog post on the perceptron algorithm. If you find any deficiencies, or have any questions, please feel free to contact me via any of my listed platforms. 

