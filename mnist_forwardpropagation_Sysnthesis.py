import sys, os
sys.path.append(os.pardir)
import numpy as np
from PIL import Image
from dataset.mnist import load_mnist
import pickle
import ActivationFunction

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'],  network['W2'],  network['W3']
    b1, b2, b3 = network['b1'],  network['b2'],  network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = ActivationFunction.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = ActivationFunction.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = ActivationFunction.softmax(a3)

    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
