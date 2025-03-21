import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn.model_selection import train_test_split


# import some data to play with
iris = datasets.load_iris()
features = iris.data 
classes=iris.target

training, test, training_true_classes, test_true_classes = train_test_split(
    features,classes,train_size=60,random_state=32)

training1 = training[training_true_classes == 0,:]
training2 = training[training_true_classes == 1,:]
training3 = training[training_true_classes == 2,:]

# Gaussian assumption, maximum likelihood estimate
mu1 = np.mean(training1, axis=0)
sigma1 = np.cov(training1, rowvar=False)

mu2 = np.mean(training2, axis=0)
sigma2 = np.cov(training2, rowvar=False)

mu3 = np.mean(training3, axis=0)
sigma3 = np.cov(training3, rowvar=False)

# Classify test samples
conf = np.zeros((3, 3))
for i,x in enumerate(test):
    y1 = multivariate_normal.pdf(x, mu1, sigma1)
    y2 = multivariate_normal.pdf(x, mu2, sigma2)
    y3 = multivariate_normal.pdf(x, mu3, sigma3)
    y = [y1, y2, y3]
    I = np.argmax(y)
    satir = test_true_classes[i]
    sutun = I
    conf[satir, sutun] += 1
    
print(conf)




