#.......................#
#Created on 18/02/2022
#By Muhammad Saad
#.......................#

import numpy as np

class Pooling:
    def __init__(self):
        self.filter_size = 2
        self.stride = 2
        self.max_list = []
    

    #forward pass for pooling layer
    #I think it is kind of new to get indices but my method really works pretty fine
    def forward_pass(self, image):
        out_n = image.shape[0]
        out_h = (((image.shape[1])-self.filter_size)//self.stride)+1
        out_w = (((image.shape[2])-self.filter_size)//self.stride)+1
        self.input_shape = image.shape
        output = np.zeros((out_n,out_h, out_w))

        row = 0
        col =0 
        for i in range(0,image.shape[1],self.stride): 
            col = 0
            for j in range(0,image.shape[2],self.stride):
                max_indexes = [[0 for columns in range(3)] for rows in range(out_n)]
                for k in range(self.filter_size):
                    for l in range(self.filter_size):
                        for out_c in range(out_n):
                                if(output[out_c][row][col]<=image[out_c][i+k][j+l]):
                                    output[out_c][row][col]=image[out_c][i+k][j+l]
                                    max_indexes[out_c] = [(out_c,i+k,j+l)]

                self.max_list.append(max_indexes)             
                col = col+1
            
            row = row+1
        
        return output


    #backward pass for pooling layer
    def backward_pass(self, prev_grad):
        grad_pool = np.zeros(self.input_shape)
        count =0 
       
        for i in range (prev_grad.shape[1]):
            for j in range (prev_grad.shape[2]):
                for c in range (prev_grad.shape[0]):
                    grad_pool[(self.max_list[count][c][0][0])][(self.max_list[count][c][0][1])][(self.max_list[count][c][0][2])] = prev_grad[c][i][j]
                    
                
                count = count+1

        return grad_pool



    #Done on 19/02/22 6:03 PM

        
        

    