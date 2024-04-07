
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Loading Dataset
Iris = load_iris()
X = Iris.data.reshape(150,4)
y = Iris.target.reshape(-1,1)
X = X.astype(np.float64)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
history = [[],[]]
#Initialising Weights and Biases (using LeChun Initialization)
W_i_h = np.random.uniform(-np.sqrt(6/4), np.sqrt(6/4), (5,4))
b_i_h = np.zeros((5,1))
W_h_o = np.random.uniform(-0.5, 0.5, (3,5))
b_h_o = np.zeros((3,1))
epochs = 30
learning_rate = 0.01

#training the model
for e in range(epochs):
    cce = 0
    acc = 0
    for x,y in zip(X_train,y_train):
        input = x.reshape(4,1)
        y = y.reshape(3,1)
        #forward
        #input to hidden layer pre-activation
        a_i_h = b_i_h + W_i_h @ input
        #hidden layer activation (Relu)
        x_i_h = np.where(a_i_h > 0, a_i_h, 0)
        #hidden to output layer pre-activation
        a_h_o = b_h_o + W_h_o @ x_i_h
        #output layer activation (Softmax)
        t = np.exp(a_h_o)
        softmax = t/np.sum(t)

        #error ( categorical cross entropy ) and accuracy
        cce += -np.sum( y * np.log(softmax) )/4
        acc += int(np.argmax(softmax) == np.argmax(y))
        grad = softmax - y
        
        #backwards
        #softmax derivative in Jacobian Form
        softmax_grad = softmax * (np.identity(3) - softmax.T)
        grad = np.dot(softmax_grad, grad)
        #updating weights between hidden and output layer
        W_h_o += -learning_rate * np.dot(grad,x_i_h.T)
        b_h_o += -learning_rate * grad
        #updating output gradient for next layer
        grad = np.dot(W_h_o.T, grad) 
        #Relu derivative
        grad = grad * np.where(a_i_h > 0, 1, 0)
        #updating Weights and Biases between input and hidden layer
        W_i_h += -learning_rate * np.dot(grad, input.T)
        b_i_h += -learning_rate * grad
        
    cce /= X_train.shape[0]
    acc /= y_train.shape[0]
    history[0].append(acc)
    history[1].append(cce)
    print("epoch : {}".format(e+1), " loss : {}".format(round(cce,2)), " accuracy : {} %".format(round(acc*100,2)))

print("Training Accuracy : {} %".format(round(acc*100)))

#predictions and performance
acc = 0
for x,y in zip(X_test, y_test):
    input = x.reshape(4,1)
    y = y.reshape(3,1)
    #input to hidden layer pre-activation
    a_i_h = b_i_h + W_i_h @ input
    #hidden layer activation (Relu)
    x_i_h = np.where(a_i_h > 0, a_i_h, 0)
    #hidden to output layer pre-activation
    a_h_o = b_h_o + W_h_o @ x_i_h
    #output layer activation (Softmax)
    t = np.exp(a_h_o)
    softmax = t/np.sum(t)
    #accuracy of predictions
    acc += int(np.argmax(softmax) == np.argmax(y))

print("Accuracy of Testing Data : {} %".format(round(acc*100/y_test.shape[0],2)))
plt.plot(range(1,epochs+1),history[0],color='teal',label="accuracy")
plt.plot(range(1,epochs+1),history[1],color='orange',label="loss")
plt.show()
