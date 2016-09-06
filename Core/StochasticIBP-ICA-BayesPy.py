'''
Created on Jun 7, 2016

@author: kon
'''
import numpy as np
from numpy import random as r
from bayespy.nodes import Gamma,GaussianARD,Dirichlet,Beta
from bayespy.inference import VB


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

def create_model(K,J):
    gamma_1,gamma_2,eta_1,eta_2,c,f,a,b,xi=initialize_prior_parameters(K, J,False)
    phi=Gamma(a,b)
    lambda_k=Gamma(c,f,plates=(K,1))
    alpha=Gamma(gamma_1,gamma_2)
    u_k=Beta([1,1])
    
    print(phi," ",lambda_k.plates," ",alpha, " ",u_k)
    
    
def ibp_ica():
    K=10
    J=8
    create_model(K, J)
    
if __name__ == '__main__':
    ibp_ica()