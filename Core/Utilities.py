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
        num_samples=100
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
    estimated=get_realization(mean_, variance_, True)
    return estimated

def get_other_methods_estimate(path):
    return load_B(path)

def unmix(self,data):
    G_mean=self.t_mu
    G_prec=self.t_l
    G=G_mean
    G_sq=G_mean**2+1/G_prec
    l_hat=self.t_a/self.t_b
    y=np.dot(np.dot(np.linalg.pinv(np.dot(G.T,G)),G.T),data.T)
    
    for k in range(self.K):
        pass
        
    
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    
    from sklearn.decomposition import FastICA, PCA
    
    ###############################################################################
    # Generate sample data
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    
    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
    
    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise
    
    #S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations
    
    # Compute ICA
    ica = FastICA(n_components=3)
    S1_ = ica.fit_transform(X)  # Reconstruct signals
    #S += 0.2 * np.random.normal(size=S.shape)  # Add noise
    from sklearn import preprocessing
    #S=preprocessing.scale(S)
    estimated_path='02016-08-29-20:46_batch_size_100_D_3_data_female_inst_mix_find_features/localparamsfinal.pickle'
    
    #orig=load_csv(original_path)
    estimated=S1_
    for tc in range(S.shape[1]):
        print('Time course',tc+1)
        for tc1 in range(estimated.shape[1]):
            print(getPearson(S[:,tc], estimated[:,tc1]))
    
    