import numpy as np
import random
#https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/
#This is the question. Our task is to create a neural 
#network that is able to predict whether an unknown person is diabetic or not given data about his exercise habits, obesity, and smoking habits. 

# Person	Smoking	Obesity	Exercise	Diabetic
# Person 1	0		  1		  0			  1
# Person 2	0		  0		  1			  0
# Person 3	1		  0		  0			  0
# Person 4	1		  1		  0			  1
# Person 5	1		  1		  1			  1

#We will create a simple neural network with one input and one output layer in Python.

features = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1,]]) #Given Data (first three features here)
labels = np.array([1,0,0,1,1]) #The target (diabetic here)
labels = labels.reshape(5,1)
weights = np.random.rand(3,1) #Returns a matrix of 3x1, we used 3 cause we have 3 features in data, so we need a vector of 3 weights.
bias = np.random.rand(1) #Returns a no. between 0 and 1.
lr = 0.05 #Learning Rate
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidder(x):
    return sigmoid(x)*(1-sigmoid(x)) #https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e this is how.

for i in range(20000):
    input = features
    xweight = np.dot(features, weights) + bias
    predicted = sigmoid(xweight)

    error = predicted-labels
    print(error.sum())
    dcost_dpred = error
    dpred_dz = sigmoidder(predicted)

    z_delta = dcost_dpred * dpred_dz

    inputs = features.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num


single_point = np.array([1,0,0])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result) #Not Diabetic as almost 0
single_point = np.array([0,1,0])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result) #Diabetic as almost 1