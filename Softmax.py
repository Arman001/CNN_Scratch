#........................#
#Created by Muhammad Saad 
#On 18/02/2022
#........................#

import numpy as np

class Softmax:


    def SoftMax(self,input):
        input = input - np.max(input)
        prob = np.exp(input)/np.sum(np.exp(input))
        return prob

    def ReLU(self, x):
        return x*(x>0)
    
    def Grad_ReLU(self,y):
        return (y > 0) * 1
        


    def __init__(self,final_out):
        self.final_out = final_out
        self.biases = np.random.rand(final_out)
        self.weights = np.zeros(final_out)
        self.start = 0
        self.output_len = 0
    
    # only testing
    # def __init__(self,final_out, weights, biases):
    #     self.final_out = final_out
    #     self.biases = biases
    #     self.weights = weights
    #     self.start = 0
    #     self.output_len = 0
    
    

    def forward_pass(self,input):
        if(self.start ==0):
            self.output_len = input.shape[0]*input.shape[1]*input.shape[2]
            self.weights = np.random.rand(self.output_len,self.final_out)/self.output_len
            self.start =1
        flatten_arr = np.ndarray.flatten(input)

        self.last_input = flatten_arr
        self.last_input_shape  = input.shape
        output = np.dot(flatten_arr,self.weights)
        output = output + self.biases
        
        output = self.SoftMax(output)
        return output

    
    ##Here lies the code for backpropagation 
    def backward_pass(self, softmax_out, label,  alpha):
        
        grad_biases = np.zeros(self.biases.shape)
        #gradient of output
        grad_output = softmax_out-label

        grad_output = grad_output.reshape(1,len(grad_output))
        self.last_input = self.last_input.reshape(len(self.last_input),1)
        
        #gradient of weights & biases
        grad_weights = self.last_input.dot(grad_output)
        grad_biases = softmax_out-label

        #gradient of input
        grad_input = self.weights.dot(grad_output.reshape(grad_output.shape[1],1))*self.last_input
        
        
        #updating
        self.weights = self.weights-alpha*grad_weights
        self.biases = self.biases - alpha*grad_biases

        
        return grad_input.reshape(self.last_input_shape)


       
#Completed finally after rigorous work on 19/02/22



        


