'''
Created on Jun 8, 2016

@author: kon
'''


from pymc3 import Model, Normal, Beta,Gamma,Deterministic,Bernoulli,DensityDist,Dirichlet,Categorical,Multinomial,MvNormal
import pymc3 as pm
import theano.tensor as T
import theano
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
import numpy.random as r

def initialize_prior_parameters(K,J,show_values=False):
    gamma_1=r.rand()
    gamma_2=r.rand()
    eta_1=r.rand()
    eta_2=r.rand()
    c=r.rand()
    f=r.rand()
    a=r.rand()
    b=r.rand()
    xi=np.ones((K,J))*(1.0/J)
    if show_values:
        print("gamma_1 init value: {} ".format(gamma_1))
        print("gamma_2 init value: {} ".format(gamma_2))
        print("eta_1 init value: {} ".format(eta_1))
        print("eta2_1 init value: {} ".format(eta_2))
        print("c init value: {} ".format(c))
        print("f init value: {} ".format(f))
        print("a init value: {} ".format(a))
        print("b init value: {} ".format(b))
        print("xi init value: \n {} ".format(xi))
    
    return gamma_1,gamma_2,eta_1,eta_2,c,f,a,b,xi


def create_model(K,J,N,D,a,b,c,f,gamma_1,gamma_2,eta_1,eta_2,xi,x,y):
    with pm.Model() as basic_model:
        
        lambda_k=Gamma("lambda",c,f,shape=K)
        
        phi=Gamma("phi",a,b)
        
        
        
        alpha=Gamma("alpha",gamma_1,gamma_2)
        
        u=Beta("u_k", alpha,1,shape=K)
        
        pi=Deterministic("pi_k", T.cumprod(u))
        
        z=Bernoulli("z",pi,shape=(D,K))

        s_kj=Gamma("s_kj",eta_1,eta_2,shape=(K,J))
        
                
        #mixture model
        varpi=Dirichlet("varpi",xi,shape=(K,J))
        
        components=Multinomial('zeta',n=1,p=varpi,shape=(N,K,J))
        
        print(components.shape)
        y=(Normal("y",mu=0,tau=s_kj,shape=(N,K,J))*components).sum(2)
       
        G=z*Normal("g",0,tau=lambda_k,shape=(D,K))
        
        mu=T.dot(G,y.T).T
        
        X=Normal('x_obs',mu=mu,tau=phi,observed=x)
    
        x=pm.variational.advi()
            
    
def ibp_ica():
    K=4
    J=3
    D=2
    N=100
    X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
    X = scale(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
    gamma_1,gamma_2,eta_1,eta_2,c,f,a,b,xi=initialize_prior_parameters(K, J, False)
    ann_input = theano.shared(X_train)
    ann_output = theano.shared(Y_train)
    N,D=X_train.shape
    
    create_model(K, J,N,D,a,b,c,f,gamma_1,gamma_2,eta_1,eta_2,xi,X_train,Y_train)
    
if __name__ == '__main__':
    ibp_ica()
