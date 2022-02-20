#..............................................#
## Created by
## Muhammad Saad
## On 17/02/22
## It is better to sketch out everything on paper
#..............................................#

import numpy as np

class Convolution:
    #Initialization of the class 
    def __init__(self,no_of_filters):
        #generally a 3x3 filter is used for convolution of smaller images
        self.no_of_filters = no_of_filters
        self.filter_size = 3
        #self.windows_list = []
        self.start = 0
        self.filters = np.random.rand(no_of_filters,self.filter_size,self.filter_size)/9
        self.biases = np.random.rand(no_of_filters)


    def ReLU(self, x):
        return x*(x>0)
    
    def Grad_ReLU(self,y):
        return (y > 0) * 1
        


    #Getting windwos 3x3 of matrix for doing dot product with filters
    #also stride is considerd as one
    #got successfully just need to apply it in convolution operation can be very useful in encrypted version and may be fast also
    #def window_extractor(self, input):
        
        # stride =1 
        # print(input.shape[1],"x",input.shape[2])
        # for i in range(input.shape[1]-(self.filter_size-1)): 
        #     for j in range(input.shape[2]-(self.filter_size-1)):
        #         window = []
        #         for k in range(self.filter_size):
        #             for l in range(self.filter_size):
        #                 ##window.append([i+k,j+l])


            
               ##self.windows_list.append(window)
        


    #while going forward in network
    def forward_pass(self, image):
        input_c = image.shape[0]
        self.last_input = image
        out_h = image.shape[1]-self.filter_size+1
        out_w = image.shape[2]-self.filter_size+1
        output = np.zeros((self.no_of_filters,out_h,out_w))

        for i in range(out_h): 
            for j in range(out_w):
                for k in range(self.filter_size):
                    for l in range(self.filter_size):
                        for out_c in range(self.no_of_filters):
                            for prev_c in range(input_c):
                                output[out_c][i][j]+=image[prev_c][i+k][j+l]*self.filters[out_c][k][l]
        

        self.last_output = output
        return self.ReLU(output)


    #while going back in network updating the filters

    def backward_pass(self, grad_pool, alpha):
        grad_output = (self.Grad_ReLU(self.last_output))@(grad_pool)
        grad_filters = np.zeros(self.filters.shape)
        out_h = self.last_input.shape[1]-grad_output.shape[1]+1
        out_w = self.last_input.shape[2]-grad_output.shape[2]+1


        #getting the gradients of filters and then updating
        for i in range(out_h): 
            for j in range(out_w):
                for k in range(grad_output.shape[1]):
                    for l in range(grad_output.shape[2]):
                        for out_c in range(self.no_of_filters):
                            for prev_c in range(self.last_input.shape[0]):
                                grad_filters[out_c][i][j]+=self.last_input[prev_c][i+k][j+l]*grad_output[out_c][k][l]
        self.filters = self.filters-alpha*grad_filters
        
        #getting biases gradient and updating the values
        grad_biases = np.zeros(self.biases.shape)
        for i in range(len(self.biases)):
            grad_biases[i] = np.sum(grad_output[i])
        self.biases = self.biases-alpha*grad_biases
        
        

#Almost completed at 20/02/22 at 10:40 AM 