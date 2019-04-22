import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#load dataset and show first 5 rows
digits = np.loadtxt('mfeat-pix.txt')
data = pd.DataFrame(digits)

data = data.to_numpy()

# 1. ---------- Data Preprocessing ----------

#creating the y column (numbers) 
d={}
for i in range(0,10):
     d[(i)] = pd.DataFrame([i] * 200)    
numbers = pd.concat(d)

numbers = numbers.to_numpy()

#Encoding numbers 
numbers_encoded = []
for i in numbers:
    temp_encoded = np.zeros(10)
    label = i[0]
    position = label - 1
    if label == 0:
        position = 9
    temp_encoded[position] = 1
    
    numbers_encoded.append(temp_encoded)
numbers_encoded = np.array(numbers_encoded)
    
    
    
#binning numbers to digits
df = np.concatenate([numbers_encoded,data],axis =1)

#Creating Testing and training set ( 100 train and 100 test for each digit)
zero = df[0:200]
zero_train = zero[0:100]
zero_test = zero[100:200]

one = df[200:400]
one_train = one[0:100]
one_test = one[100:200]

two = df[400:600]
two_train = two[0:100]
two_test = two[100:200]

three = df[600:800]
three_train = three[0:100]
three_test = three[100:200]

four = df[800:1000]
four_train = four[0:100]
four_test = four[100:200]

five = df[1000:1200]
five_train = five[0:100]
five_test = five[100:200]

six = df[1200:1400]
six_train = six[0:100]
six_test = six[100:200]

seven = df[1400:1600]
seven_train = seven[0:100]
seven_test = seven[100:200]

eight = df[1600:1800]
eight_train = eight[0:100]
eight_test = eight[100:200]

nine = df[1800:2000]
nine_train = nine[0:100]
nine_test = nine[100:200]


test = np.concatenate([zero_test, one_test, two_test, 
                      three_test, four_test, five_test, 
                      six_test, seven_test, eight_test, nine_test])
    
train = np.concatenate([zero_train, one_train, two_train, 
                      three_train, four_train, five_train, 
                      six_train, seven_train, eight_train, nine_train])
    


#Slice train and test.
y_train = train[:,:10]
x_train = train[:,10:]

y_test = test[:,:10]
x_test = test[:,10:]


# 2. ---------- Kmeans and Linear regression  ----------


# ----- Kmeans  ------

def cluster_init(array, k):
 
    initial_assgnm = np.append(np.arange(k), np.random.randint(0, k, size=(len(array))))[:len(array)]
    np.random.shuffle(initial_assgnm)
    zero_arr = np.zeros((len(initial_assgnm), 1))

    for indx, cluster_assgnm in enumerate(initial_assgnm):
        zero_arr[indx] = cluster_assgnm
    upd_array = np.append(array, zero_arr, axis=1)

    return upd_array


def kmeans(array, k):
    
    cluster_array = cluster_init(array, k)

   
    while True:
        unique_clusters = np.unique(cluster_array[:, -1])

        centroid_dictonary = {}
        for cluster in unique_clusters:
            centroid_dictonary[cluster] = np.mean(cluster_array[np.where(cluster_array[:, -1] == cluster)][:, :-1], axis=0)


        start_array = np.copy(cluster_array)

    
        for row in range(len(cluster_array)):
            cluster_array[row, -1] = unique_clusters[np.argmin(
                [np.linalg.norm(cluster_array[row, :-1] - centroid_dictonary.get(cluster)) for cluster in unique_clusters])]
            
        if np.array_equal(cluster_array, start_array):
            break

    return centroid_dictonary


def ctrid_grav_plot(centroid_dictonary, limit_codevector_dim=10):
    

    counter = 0

    
    for each_key in centroid_dictonary.keys():
        codebook_vector = np.array(centroid_dictonary.get(each_key)).reshape(16,15)

    

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
        ax1, ax2 = ax
        ax1.imshow(codebook_vector, cmap='gray')
        ax1.set_title('$\mu$ Codebook Vector='+str(each_key), fontweight='bold')
        ax2.hist(codebook_vector)
        ax2.set_title('$\mu$ Codebook Vector Distribution', fontweight='bold')
        plt.show()
        counter += 1
        
        if counter == limit_codevector_dim:
            break
       
# ----- Linear Regression ------      
 
class LinearRegression(object):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        alpha = 0.5
        self.w = np.linalg.inv(X.T.dot(X)+ alpha*alpha*np.identity(X.shape[1])).dot(X.T).dot(y)
        return self
    
    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)
           
    def mse(self, X,y):
        return np.mean((y - self.predict(X))**2)
        
    def miss_score(self, X, y):
        return 1 - sum(np.argmax(np.insert(X, 0, 1, axis=1).dot(regr.w),axis=1) == np.argmax(y,axis=1))/len(y)
    


# 3. ---- Cross validation -----       
## ---------- Creating Folds for K-Fold ----------

from random import seed
from random import randrange

seed(1)
def cross_validation_split(train, folds=3):
	dataset_split = list()
	dataset_copy = list(train)
	fold_size = int(len(train) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split



#Plotting training set with 10 kmeans to check codebook vectors
    
ctrid_grav_plot(kmeans(x_train,10))


k=[1,10,20,30,40,50,60,70,80,90,100,120,130,140,150,160,170,180,190,200,210,220,230,240]

MSE_train = []
MISS_train = []
MSE_val = []
MISS_val = []


folds = cross_validation_split(train, 8)
for k_value in k:
    MSE_train_scores = []
    MISS_train_scores = []
    MSE_val_scores = []
    MISS_val_scores = []
   
    for i in range(len(folds)):
        train_data = []
        test_data = folds[i]
        
        for j in range(len(folds)):
            if j != i:
                train_data.extend(folds[j])
        
        test_data = np.array(test_data)
        train_data = np.array(train_data)
        
        X_train = train_data[:, 10:]
        Y_train = train_data[:, :10]
        
        X_test = test_data[:, 10:]
        Y_test = test_data[:, :10]
        
        
        centroid_dict_train_array = []
        
        centroid_dict_train = kmeans(X_train, k_value)
        centroid_dict_train_array.append(centroid_dict_train)

     #Feature reduction
        reduced_train_array = []
        for t in centroid_dict_train_array:
            reduced_train = []
            for datapoint in X_train:
                distance = []
                for c, codebookvector in t.items():
                    distance.append(np.linalg.norm(datapoint-codebookvector))
                reduced_train.append(distance)
            reduced_train_array.append(reduced_train)
            
            
    reduced_val_array = []
    for t in centroid_dict_train_array:
        reduced_val = []
        for datapoint in X_test:
            distance = []
            for c, codebookvector in t.items():
                distance.append(np.linalg.norm(datapoint-codebookvector))
            reduced_val.append(distance)
        reduced_val_array.append(reduced_val)
            
        
      #train   
        for r in reduced_train_array:
            
            regr = LinearRegression().fit(r,Y_train)
            mse_train = regr.mse(r,Y_train)
            miss_train = regr.miss_score(r,Y_train)
            MSE_train_scores.append(mse_train)
            MISS_train_scores.append(miss_train)
            #Validation
            mse_val = regr.mse(reduced_val,Y_test)
            miss_val = regr.miss_score(reduced_val,Y_test)
            MSE_val_scores.append(mse_val)
            MISS_val_scores.append(miss_val)
    
    a = np.mean(MSE_train_scores)
    MSE_train.append(a)
    b = np.mean(MISS_train_scores)
    MISS_train.append(b)
    c = np.mean(MSE_val_scores)
    MSE_val.append(c)
    d = np.mean(MISS_val_scores)
    MISS_val.append(d)
    

plt.plot(k,MISS_train,color='blue',linestyle='solid',label='MISS train')
plt.plot(k,MISS_val, color='red',linestyle='solid',label='MISS test')
plt.legend();
plt.xlabel('K')
plt.ylabel('accuracy')
 



  
#Kmeans = 55
centroids = kmeans(x_train,80) 

#reducing TRAIN
reduced_features_train =[]
for datapoint in x_train:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_train.append(distance)

#reducing TEST
reduced_features_test =[]
for datapoint in x_test:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_test.append(distance)


regr = LinearRegression().fit(reduced_features_train,y_train)
 

MISS = regr.miss_score(reduced_features_test,y_test)
MSE = regr.mse(reduced_features_test,y_test)
accuracy = sum(np.argmax(np.insert(reduced_features_test, 0, 1, axis=1).dot(regr.w),axis=1) == np.argmax(y_test,axis=1))/len(y_test)

print(MSE,MISS,accuracy)



#Kmeans = 65
centroids = kmeans(x_train,90) 

#reducing TRAIN
reduced_features_train =[]
for datapoint in x_train:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_train.append(distance)

#reducing TEST
reduced_features_test =[]
for datapoint in x_test:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_test.append(distance)


regr = LinearRegression().fit(reduced_features_train,y_train)
 

MISS = regr.miss_score(reduced_features_test,y_test)
MSE = regr.mse(reduced_features_test,y_test)
accuracy = sum(np.argmax(np.insert(reduced_features_test, 0, 1, axis=1).dot(regr.w),axis=1) == np.argmax(y_test,axis=1))/len(y_test)

print(MSE,MISS,accuracy)
     

#Kmeans = 
centroids = kmeans(x_train,100) 

#reducing TRAIN
reduced_features_train =[]
for datapoint in x_train:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_train.append(distance)

#reducing TEST
reduced_features_test =[]
for datapoint in x_test:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_test.append(distance)


regr = LinearRegression().fit(reduced_features_train,y_train)
 

MISS = regr.miss_score(reduced_features_test,y_test)
MSE = regr.mse(reduced_features_test,y_test)
accuracy = sum(np.argmax(np.insert(reduced_features_test, 0, 1, axis=1).dot(regr.w),axis=1) == np.argmax(y_test,axis=1))/len(y_test)

print(MSE,MISS,accuracy)


#Kmeans = 110
centroids = kmeans(x_train,110) 

#reducing TRAIN
reduced_features_train =[]
for datapoint in x_train:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_train.append(distance)

#reducing TEST
reduced_features_test =[]
for datapoint in x_test:
    distance =[]
    for i, codebookvector in centroids.items():
        distance.append(np.linalg.norm(datapoint-codebookvector))
    reduced_features_test.append(distance)


regr = LinearRegression().fit(reduced_features_train,y_train)
 

MISS = regr.miss_score(reduced_features_test,y_test)
MSE = regr.mse(reduced_features_test,y_test)
accuracy = sum(np.argmax(np.insert(reduced_features_test, 0, 1, axis=1).dot(regr.w),axis=1) == np.argmax(y_test,axis=1))/len(y_test)

print(MSE,MISS,accuracy)










    

    



        
        


    
    
    







