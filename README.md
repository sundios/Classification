
# ![alt text][logo]
[logo]:http://www.komahats.com/Koma_logo_bear.jpg

# Databases
# Authors
- Margaret Sant 
- Konrad Burchardt



## Table of Contents 

- [1. Introduction](#1-introduction)
- [2. The Digits Dataset](#1-the-digits-dataset)
- [2. The Digits Dataset](#1-the-digits-dataset)


### 1. Introduction

In this report we will go through the steps we took to train our digits
classifier. We will start with a brief explanation of the dataset, then
we will explain how we preprocessed the data and created our testing and
training datasets. After this we will go in detail into our 3 functions
we created, K-means for dimension reduction, Linear regression to find
our W opt and cross validation to find the right number of
k . After this we will explain our optimal number of
k and our finding. We will also include different case scenarios
with different parameters.


### 2. The Digits Dataset

The input patterns for our main dataset are 15 x 16 grayscale images of
handwritten digits 0—9, which were normalized in size and ratio to fill
the image space. The dataset is comprised of two-dimensional array of
size 15 x 16, making vector x_i lengths of 240. The values of the
vector components correspond to the greyscale values [ 0 to 6]
The rows consist of 200 instances of each digit, making 200 instances of
first “0” images, then “1” images, until “9”.

### 3. The objective

The objective is to train model that will classify digits patterns based
on a sample of hand-written digits. We use vector quantization with
K-Means clustering for image reduction, and Linear Regression for our
classifier. We need to also search for the optimal hyper parameter
k (number of clusters) by incorporating K-Fold Cross Validation
on our regressor, repeating for various feature dimensions (various
values of k). We need to get a miss classification rate below 5%.


### 4. Data Preprocessing – Creating the Train and Test Set

Our first preprocessing step was to create a column to indicate the
class (digit) of our training pattern. We would go from 0 to 9 and
change the number every 200 rows. The first 200 would be class = 0 for
“0” patterns and so on until we have class = 9 for “9” patterns. Then,
we used one-hot encoding to create a binary 10-dimensional target vector
z_i, in which a one corresponds to the class of pattern x_i  and
a zero everywhere else. 
Second, we equally divide our main dataset into x(train) and  x(test) so that 1000 patterns that
correspond to 200 of each class belong to both each. This way, our finalmodel is trained and tested equally for each class.



### 5. Dimension Reduction - Creating functions for feature extraction 
For dimension reduction, we use K-Means Clustering to extract a low dimensional feature representation of the data points:


<img src="http://kburchardt.com/images/1.png" alt="equation" width="400" style="margin:0 auto 0;"/>

This is an unsupervised machine learning technique that finds similarities in the data-points and clusters them based on closeness
(Euclidian distance) so each cluster C1, . . . , Ck represents a feature. 
First, an initiation function randomly assigns each data point in the training set to one of the k clusters. 
The first step of the algorithm re-assigns each data-point to the closest cluster centroid. The second step recalculates the location of the centroids based on the mean of the data-points in the respective clusters. Steps one and two repeat until no reassignment takes place. This equation describes the formula for computing the mean of each cluster:

<img src="http://kburchardt.com/images/2.png" alt="equation" width="200" style="margin:0 auto 0;"/>

μj s the centroid, aka the codebook vector. The minimal distance between the data-point and centroid is thus calculated as distance αi = ∥xi − μj ∥. This is the Euclidian distance. Our K-Means algorithm employs distances as a distance vector on both the train and the test set to obtain the reduced x to our feature vector f(x) = (a1. . . , ak) which will be fed into our Linear Regression classifier. We later explain how we split xtrain into validation and training folds to evaluate the models for various numbers of k to find the optimal kopt . We will also use K-Means on xtest using kopt for feature reduction before using our final Linear Regression classifier.

### 6. Linear Regression - Creating functions for Linear Regression
In this stage, we are using Linear Regression as a mechanism to transform the feature vectors f(x) into a hypothesis vector h(x) ∈ Rk. What we want is a decision function d: Rm → Rk which returns ideal class hypothesis vectors.

We use the machine learning method called Linear Regression as our d decision function to find the optimal weights Wopt for our features f(xi) to create the hypothesis vector, which is an approximate our target zi, our binary encoding for the digit classes explained above.
To start, we create a feature vector of “1”’s as a padding to deal with bias. So, f(xi) has size k+1, and Wopt is a matrix of size 10 x (1+k) . Then ,we create our linear regressor with the following equation:

<img src="http://kburchardt.com/images/3.png" alt="equation" width="400" style="margin:0 auto 0;"/>

where X is our f(x), and Y is our binary encoding zi. For our model, we set α to 0.5. And, N is the dimensions of our raw data. We fit the above equation to xtrain with f(x) and zi to create our hypothesis vector h(x), which we will use to test the (1) misclassification MISS and (2) mean squared error MSE scores of our train set using cross validation, described in the following section.


### 7. Cross Validation
In order to boost our model performance, we run the cross-validation scheme described above for various values of k. We do this to show whether the model is under- or overfitting. We only have xtrain available to us at this time, so we split the training data set S = (xi,yi)(i=,...,N) into two subsets, a \train” T = (xi,yi)i∈I and a “validation” V = (x′j,yi′)i∈I′.
This way we train the model and get a training error metric, while also sim- ulating “new data” in xvalidateto get a validation (or test) error metric. We use two different error score metrics, mean-squared-error MSE and misclassification MISS to see how our model performs:


<img src="http://kburchardt.com/images/4.png" alt="equation" width="400" style="margin:0 auto 0;"/>

where d is our linear regression decision function training on the training folds, zj our given encoded target vector, and ci is the hypothesis vectors.

### 8. Using Cross Validation to find optimal K
We use Cross validation on various numbers of ki, and graph our MSE and MISS results. For our problem, we use 8 folds for our cross validation, results shown on the graphs below:
Given k=[1,10,20,30,40,50,60,70,80,90, 100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]:


<img src="http://kburchardt.com/images/5.png" alt="equation" width="400" style="float:left;"/> <img src="http://kburchardt.com/images/6.png" alt="equation" width="400" style="margin:0 auto 0;"/>




