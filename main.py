#...............................#
## Created By
## Muhammad Saad
## 17/02/2022
## Lets begin builing our own cnn
#................................#


from telnetlib import XASCII
import numpy as np
from Convolution import *
from Pooling import *
from Softmax import *
from Data_Preprocessing import *


#Initilization of Layers of the CNN Network
con = Convolution(2)
pool= Pooling()
softmax = Softmax(2)


#calculating final loss of our network and accuracy
def cross_entropy(label, soft_out):
    loss = 0.0 
    grad = 0.0
    acc = 0
    for l in range (len(label)):
        if(label[l]==1):
            loss =  -np.log(soft_out[l])
            grad = -1/soft_out[l]
            if(np.round(soft_out[l])==label[l]):
                acc = 1

    

    return loss, grad, acc


#While going forward in cnn
def forward_pass(image, label):
    
    out = con.forward_pass(image)
    out = pool.forward_pass(out)
    out = softmax.forward_pass(out)
    loss, gradient, acc = cross_entropy(label,out)
    

    return loss, out, acc

#Doing backpropagation in our CNN
def backward_pass(out, label,alpha=0.05):
    grad = softmax.backward_pass(out,label,alpha)
    grad = pool.backward_pass(grad)
    con.backward_pass(grad,alpha)



def train(images, labels):
    print("Training the network now...")
    
    accuracy = 0   

    for i in range (len(images)):
        loss, out, acc = forward_pass(images[i],labels[i])
        accuracy = accuracy+acc
        if(i%500==0):
            print("Current loss is: ",loss)
        backward_pass(out,labels[i])
        
    print("Final accuracy of training is: ",accuracy)


def test(images, labels):
    print("Testing the network now...")
    
    accuracy = 0   

    for i in range (len(images)):
        loss, out, acc = forward_pass(images[i],labels[i])
        accuracy = accuracy+acc
        if(i%1000==0):
            print("Current loss is: ",loss)
        
    print("Final accuracy of testing is: ",accuracy)



def main():
    class_names, X, Y = load_data()
    train_X = X[:26000]
    train_Y = Y[:26000]

    test_X = X[26000:]
    test_Y = Y[26000:]

    print("Total Data: ", len(X))
    
    train(train_X,train_Y)
    test(test_X,test_Y)





if __name__==main():
    main()

