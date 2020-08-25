import numpy as np
import ActivationFunction

X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X,W1) + B1
Z1 = ActivationFunction.sigmoid(A1)
print(A1)
print(Z1)

W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(A1,W2) + B2
Z2 = ActivationFunction.sigmoid(A2)

print(A2)
print(Z2)

W3 = np.array([[0.1,0.3], [0.2,0.4]])
B3 = np.array(([0.1,0.2]))

A3 = np.dot(Z2, W3) + B3

def identify_function(x):
    return x

Y = identify_function(A3)

print(Y)


