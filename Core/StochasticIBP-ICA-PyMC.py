'''
Created on Jun 8, 2016

@author: kon
'''


from pymc3 import Model, Normal, Beta,Gamma,Deterministic,Bernoulli,DensityDist,Dirichlet,Categorical
import numpy as np
import scipy as sp
from theano import tensor as T
from numpy import random as r

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


def create_model(K,J,N,D,a,b,c,f,gamma_1,gamma_2,eta_1,eta_2,xi):
    basic_model=Model()
    with basic_model:
        lambda_k=Gamma("lambda",c,f,shape=K)
        phi=Gamma("phi",a,b)
        e=Normal("error",0,T.inv(phi)*np.eye(N,D))
        alpha=Gamma("alpha",gamma_1,gamma_2)
        u=Beta("u_k", 1,alpha,shape=K)
        pi=Deterministic("pi_k", u*T.concatenate([[1],T.extra_ops.cumprod(u)[:-1]]))
        z=Bernoulli("z",pi,shape=(D,K))
        #Spike and slab distro
        slab=Normal("slab",0,T.inv(lambda_k),shape=(D,K))
        G=DensityDist("G",lambda value:z*slab+(1-z)*np.inf)
        S_inv=Gamma("s_inv",eta_1,eta_2,shape=(K,J))
        varpi=Dirichlet("varpi",xi,shape=(K,J))
        inter=Normal("inter",0,T.inv(S_inv),shape=(K,J))

        #mixture model
        components=Categorical('component',p=varpi.T,shape=(K,J))
        y=Normal("y",mu=0,sd=T.inv(S_inv)[components])
        print(G)
        print(y.shape)
#         if (z==0):
#             G=Deterministic("g", 1)
#         else:
#             G=Normal("g",0,T.inv(lambda_k))
#         print(G)
        
            
    
def ibp_ica():
    K=10
    J=8
    D=5
    N=10
    gamma_1,gamma_2,eta_1,eta_2,c,f,a,b,xi=initialize_prior_parameters(K, J, False)
    create_model(K, J,N,D,a,b,c,f,gamma_1,gamma_2,eta_1,eta_2,xi)
    
if __name__ == '__main__':
    ibp_ica()
