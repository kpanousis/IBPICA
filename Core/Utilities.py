'''
Created on Aug 27, 2016

@author: kon
'''

import numpy as np
from scipy.stats import pearsonr
import pickle
import scipy.io as sio

def get_realization(mean_values,variance_values,average=False):
    '''
    Get a realization of the matrix using either averaging or a single sample
    '''
    N,K=mean_values.shape
    if average:
        num_samples=10000
        ymat=np.zeros((N,K))
        for iter in range(num_samples):
            ymat+=np.random.normal(mean_values,variance_values)
        ymat/=num_samples
    else:
        ymat=np.random.normal(mean_values,variance_values)
    return ymat


def getPearson(originalMatrix,estimatedMatrix):
    '''
    get the pearson correlation between the estimated and original matrix
    '''
    return pearsonr(originalMatrix, estimatedMatrix)
    
    
    
def load_csv(path):
    '''
    Load the original matrix from the csv
    '''
    return np.genfromtxt(path,delimiter=',')

def load_B(path):
    return sio.loadmat(path)['Bnew']

def load_moments(path):
    '''
    Function to load my results
    '''
    with open(path,'rb') as f:
        y=pickle.load(f)
    return y[0],y[1]

def get_ibp_estimate(estimatedpath):
    variance_,mean_=load_moments(estimatedpath)
    mean_=mean_.reshape(180,3,mean_.shape[1]).mean(1)
    variance_=variance_.reshape(180,3,mean_.shape[1]).mean(1)
    estimated=get_realization(mean_, variance_, True)
    return estimated

def get_other_methods_estimate(path):
    return load_B(path)

if __name__ == '__main__':
    original_path='data/Bmat.csv'
    estimated_path='02016-08-26-18:01_batch_size_60_D_3000_data_helwig_snr_1_overlap_1_find_features/localparamsfinal.pickle'
    
    orig=load_csv(original_path)
    estimated=get_ibp_estimate(estimated_path)
    for tc in range(orig.shape[1]):
        print('Time course',tc+1)
        for tc1 in range(estimated.shape[1]):
            print(getPearson(orig[:,tc], estimated[:,tc1]))
    
    