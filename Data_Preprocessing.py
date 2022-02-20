#........................#
#Created By Muhammad Saad#
#on 20/02/2022           #
#........................#



import pandas as pd
import numpy as np

def load_data():
    zeus_1 = pd.read_csv('./Data/Zeus.csv', delimiter=',')
    zeus = zeus_1.to_numpy()
    y_zeus  = np.zeros((len(zeus),2), dtype=int)
    y_zeus[:,0] = 1
    zeus = np.append(zeus,y_zeus, axis=1)

    neris_1 = pd.read_csv('./Data/Neris.csv', delimiter=',')
    neris = neris_1.to_numpy()
    y_neris  = np.zeros((len(neris),2), dtype=int)
    y_neris[:,1] = 1
    neris = np.append(neris,y_neris, axis=1)
    
    class_names = ["zeus", "neris"]

    X_data = np.concatenate((zeus,neris), axis=0)
    np.random.shuffle(X_data)
    X_new = X_data[:,:784]
    Y = X_data[:,784:]
    my_list = []
    for i in range (X_new.shape[0]):
        my_list.append(X_new[i].reshape(1,28,28))

    X = np.array(my_list)
    return class_names, X, Y



