#........................#
#Created By Muhammad Saad#
#on 20/02/2022           #
#........................#



import pandas as pd
import numpy as np

def load_data():
    cridex_1 = pd.read_csv('./Data/Cridex.csv', delimiter=',')
    cridex = cridex_1.to_numpy()
    cridex = cridex[:20000]
    y_cridex  = np.zeros((len(cridex),2), dtype=int)
    y_cridex[:,0] = 1
    cridex = np.append(cridex,y_cridex, axis=1)

    smb_1 = pd.read_csv('./Data/SMB.csv', delimiter=',')
    smb = smb_1.to_numpy()
    smb = smb[:20000]
    y_smb  = np.zeros((len(smb),2), dtype=int)
    y_smb[:,1] = 1
    smb = np.append(smb,y_smb, axis=1)
    
    class_names = ["cridex", "smb"]

    X_data = np.concatenate((cridex,smb), axis=0)
    np.random.shuffle(X_data)
    X_new = X_data[:,:784]
    Y = X_data[:,784:]
    my_list = []
    for i in range (X_new.shape[0]):
        my_list.append(X_new[i].reshape(1,28,28))

    X = np.array(my_list)
    return class_names, X, Y



