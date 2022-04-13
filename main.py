#...............................#
## Created By
## Muhammad Saad
## 15/03/2022
## Lets begin encrypting this CNN
#................................#




import numpy as np
import time 
from Convolution import *
from Pooling import *
from Softmax import *
from Data_Preprocessing import *
import pickle

# with open('./con_weights.pkl', 'rb') as file:
#     weights = pickle.load(file)
# with open('./fc_weights.pkl', 'rb') as file:
#     fc_weights = pickle.load(file)
# with open('./con_biases.pkl', 'rb') as file:
#     biases = pickle.load(file)
# with open('./fc_biases.pkl', 'rb') as file:
#     fc_biases = pickle.load(file)
#Initilization of Layers of the CNN Network
con = Convolution(8)
pool= Pooling()
softmax = Softmax(2)
#testing purposes
# con = Convolution(8, weights, biases)
# pool= Pooling()
# softmax = Softmax(2, fc_weights, fc_biases)


#calculating final loss of our network and accuracy
def cross_entropy(label, soft_out):
    loss = 0.0 
    grad = 0.0
    acc = 0
    for l in range (len(label)):
        if(label[l]==1):
            
            if(soft_out[l]==0):
                soft_out[l]=0.01

            loss =  -np.log(soft_out[l])
            #grad = -1/soft_out[l]
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
def backward_pass(out, label,alpha=0.001):
    grad = softmax.backward_pass(out,label,alpha)
    grad = pool.backward_pass(grad)
    con.backward_pass(grad,alpha)



def train(images, labels):
    print("----------------------------------------------------")
    print("................TRAINING IS STARTING................")
    print("----------------------------------------------------")

    batch_size = 1000
    accuracy = 0   
    t_loss=0.0
    count =1
    start = time.time()
    for epoch in range(3):
        print(f".............Epoch {epoch+1} Started...............")
        print("----------------------------------------------------")
        for i in range (len(images)):
            loss, out, acc = forward_pass(images[i],labels[i])
            t_loss = t_loss+loss
            accuracy = accuracy+acc
            if((i+1)%batch_size==0 and i!=0):
                print(f"[Step: {count}]|Loss: {t_loss/batch_size}|Accuracy: {accuracy/batch_size*100}%")
                print("-------------------------------------------------------")
                t_loss = 0.0
                accuracy = 0
                count = count+1
            backward_pass(out,labels[i])
        print(f"..............Epoch {epoch+1} Ended................")
        print("----------------------------------------------------")
    
    end = time.time()
    print(f"Total training time: {end-start} seconds")
    print("------------------------------------------------------")
    print("Storing trained parameters")
    with open('con_weights.pkl', 'wb') as file:
        pickle.dump(con.filters, file)
    with open('con_biases.pkl', 'wb') as file:
        pickle.dump(con.biases, file)
    with open('fc_weights.pkl', 'wb') as file:
        pickle.dump(softmax.weights, file)
    with open('fc_biases.pkl', 'wb') as file:
        pickle.dump(softmax.biases, file)
    print("------------------------------------------------------")
    

def test(images, labels):

    print("\n")
    print("\n")

    print("----------------------------------------------------")
    print("................TESTING IS STARTING.................")
    print("----------------------------------------------------")
    accuracy = 0
    t_loss = 0.0   
    start = time.time()
    for i in range (len(images)):
        loss, out, acc = forward_pass(images[i],labels[i])
        accuracy = accuracy+acc
        t_loss = t_loss+loss
    
    end = time.time()
    
    print("Test loss is: ",t_loss/len(images))
    print(f"Test Accuracy: {(accuracy/len(images))*100}%")
    print(f"Total test time: {end-start} seconds")

    print("------------------------------------------------------")



def main():
    print("----------------------------------------------------")
    print("................LOADING THE DATASET.................")
    print("----------------------------------------------------")

    testing = 0
    if testing == 0:
        class_names, X, Y = load_data()
        X = X/255
        train_X = X[:30000]
        train_Y = Y[:30000]

        test_X = X[30000:]
        test_Y = Y[30000:]
        print(f"Classes: {class_names}")
        print(f"Total Data Samples: {len(X)} ")
        print(f"Train Samples: {len(train_X)}")
        print(f"Test Samples: {len(test_X)}")

        train(train_X,train_Y)
        test(test_X,test_Y)
    if testing == 1:
        # with open('./Data/unseen_X.pkl', 'rb') as file:
        #     X = pickle.load(file)
        # with open('./Data/unseen_Y.pkl', 'rb') as file:
        #     Y = pickle.load(file)
        # X = X / 255
        class_names, X, Y = load_data()
        X = X/255
        train_X = X[:30000]
        train_Y = Y[:30000]

        test_X = X[30000:]
        test_Y = Y[30000:]
        
        
        test(test_X, test_Y)
      





if __name__==main():
    main()

