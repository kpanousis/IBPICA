'''
Created on Jun 29, 2016

@author: kon
'''
import theano
import numpy as np
import time
import pickle
import sys
import os
from scipy.special import psi
from scipy.stats import bernoulli
from numpy.linalg import inv
from numpy.linalg import det
from scipy.special import gammaln
from scipy.special import expit
from scipy import io as sio


class IBP_ICA:
    
    def __init__(self,K,D,J, S,N,rho=0.01):
        '''
        Initialize a IBP_ICA instance.
        
        Parameters
        ----------
        K: int
            The number of active features
        D: int
            The second dimension of the dataset
        J: int
            The number of mixtures
        S: int
            The length of the minibatch
        N: int
            The size of the original dataset
        rho: float
            The step size for each iteration
        '''
        
        self.K=K
        self.D=D
        self.J=J
        self.S=S
        self.N=N
        self.rho=rho
    
    
    #===========================================================================
    # Initialize posterior parameters and xi 
    #===========================================================================
    def init_params(self):
        '''
        Initialize the posterior parameters
        
        Returns
        -------
        params: list
            List containing the tilde_tau, omega, hat_tau, tilde_a, tilde_c, tilde_f, tilde_gamma_1 and tilde_gamma_2 parameters
        
        local_params: list
            The local parameters of the model, namely, tilde_s, tilde_m and zeta
        
        batch_params: list
            The global parameters that need batch update: tilde_eta_1, tilde_eta_2, tilde_xi, tilde_lambda, tilde_mu and tilde_b
        '''
        self.xi=2*np.ones((self.K,self.J))
        
        #=======================================================================
        # The scalar parameters
        #=======================================================================
        self.t_a=2.0
        self.t_b=2.0
        self.t_g_1=2.0
        self.t_g_2=2.0
        
        #=======================================================================
        # matrices
        #=======================================================================
        self.t_e_1=1.0*np.ones((self.K,self.J),dtype='float32')
        self.t_e_2=1.0*np.ones((self.K,self.J),dtype='float32')
        self.t_xi=1.0*np.ones((self.K,self.J),dtype='float32')
        self.t_l=1.0*np.ones((self.D,self.K),dtype='float32')
        self.t_mu=np.random.normal(0,1,(self.D,self.K))
        self.omega=-np.random.random((self.D,self.K))
        self.t_s=1.0*np.ones((self.N,self.K),dtype='float32')
        self.t_m=np.random.normal(0,1,(self.N,self.K))
        
        #=======================================================================
        # tensor
        #=======================================================================
        #zeta=np.random.random(size=(self.S,self.K,self.J))
        self.zeta=1.0*np.ones((self.N,self.K,self.J))
        for n in range(self.N):
            self.zeta[n,:,:]/=self.zeta[n,:,:].sum(1).reshape(-1,1)
        
        
        #=======================================================================
        # vectors
        #=======================================================================
        self.t_c=1.0*np.ones((self.K,1),dtype='float32')
        self.t_f=1.0*np.ones((self.K,1),dtype='float32')
        self.t_tau=1.0*np.ones((self.K,1),dtype='float32')
        self.h_tau=1.0*np.ones((self.K,1),dtype='float32')
        
#         print((t_a/t_b)*(t_mu**2+t_l).sum(0)+(zeta*(t_e_1/t_e_2)).sum(2))
        
        #=======================================================================
        # The order is very important 
        #=======================================================================
        self.params=[self.t_tau,self.omega,self.h_tau,self.t_c,self.t_f,self.t_g_1,self.t_g_2]
        self.local_params=[self.t_s,self.t_m,self.zeta]
        self.batch_params=[self.t_b,self.t_e_1,self.t_e_2,self.t_xi,self.t_l,self.t_mu,self.t_a]
        self.gamma_1=2
        self.gamma_2=2
        self.b=2
        self.a=2
        self.c=2
        self.f=2
        self.eta_1=2
        self.eta_2=2
    
    #===========================================================================
    # Update the number of features
    # METROPOLIS HASTINGS STEP
    #===========================================================================
    def feature_update(self,miniBatch):
        '''
        Function for using Gibbs sampling to update the number of features.
        To be implemented 
        '''
        print('Feature Update Step...')
        #some "global" calculations
        expectation_phi=self.t_a/self.t_b
        
        
        K_d_star=np.random.poisson(self.t_g_1/(self.t_g_2*(self.D-1)),size=self.D)
        
        #Remove redundant features
        self.unused_components()

        #=======================================================================
        # construct IBP
        #=======================================================================
        alpha=np.random.gamma(self.gamma_1,self.gamma_2)
        u=np.random.beta(alpha,1,size=self.K)
        pi_k=np.cumprod(u)
        z=np.zeros((self.D,self.K))
        for k in range(self.K):
            z[:,k]=bernoulli.rvs(pi_k[k],size=self.D)
        lambda_k=np.random.gamma(self.c,self.f,size=self.K)
        
        
        #===========================================================================
        # For each dimension update features
        #===========================================================================
        for d in range(self.D):
            
            if (K_d_star[d]==0):
                continue
            #Draw random samples  G_{d,:}^* from the prior
            G=np.zeros((1,K_d_star[d]))
            for k in range(K_d_star[d]):
                G[0,k]=z[d,k]*np.random.normal(0,lambda_k[k]**-1)
            
            
            #eye matrix
            I=np.eye(self.K,self.K)
             
            #E_g[G_{:,d}^T G_{:,d}]
            #may need a modification here
            expectation_g_transpose_g=np.dot((self.t_l[d,:]+self.t_mu[d,:]**2).reshape(-1,1),(self.t_l[d,:]+self.t_mu[d,:]**2).reshape(1,-1))
            
            #calculate M_d and M_d_star
            M_d=expectation_phi*expectation_g_transpose_g+I
            M_d_star=expectation_phi*np.dot(G[0,:].T,G[0,:])+np.eye(K_d_star[d],K_d_star[d])
                
            exp_sum=0
            exp_sum_star=0
            
            for n in range(self.S):   

                m_nd=expectation_phi*np.dot(inv(M_d),self.t_mu[d,:].T)*(miniBatch[n,d]-np.dot(self.t_mu[d,:].T,self.t_m[n,:].T))
                exp_sum+=np.dot(np.dot(m_nd.T,M_d),m_nd)

                m_nd_star=expectation_phi*np.dot(inv(M_d_star),G[0,:].T)*(miniBatch[n,d]-np.dot(self.t_mu[d,:],self.t_m[n,:].T)) 
                exp_sum_star+=np.dot(np.dot(m_nd_star.T,M_d_star),m_nd_star)
    
            
            theta_d=det(M_d)**(-self.S/2)*np.exp(0.5*exp_sum)
            theta_d_star=det(M_d_star)**(-self.S/2)*np.exp(0.5*exp_sum_star)
            
            #acceptance probability
            p_d_star=min(1,theta_d_star/theta_d)
            
            #accept proposal?
            if (np.random.rand()<p_d_star):
                
                if (K_d_star[d]>0):
                    self.expand_arrays(K_d_star[d])
                    self.K+=K_d_star[d]
        
    
    
    def mult_bound_calc(self):
        #qk calculation 
        q_k=np.exp(psi(self.h_tau)+(np.cumsum(psi(self.t_tau),0)-psi(self.t_tau))-np.cumsum(psi(self.t_tau+self.h_tau),0))
        q_k/=q_k.sum()
        
        #mult bound
        #check this later just to be sure
        second_sum=np.zeros((self.K,1))
        third_sum=np.zeros((self.K,1))
        
        for k in range(self.K):
            for m in range(k):
                temp=q_k[m+1:k+1].sum()
                second_sum[k,0]+=temp*psi(self.t_tau[m,0])
            for m in range(k+1):
                temp=q_k[m:k+1].sum()
                third_sum[k,0]+=temp*psi(self.t_tau[m,0]+self.h_tau[m,0])
             
        mult_bound=np.cumsum(q_k*psi(self.h_tau),0)+second_sum-third_sum-np.cumsum(q_k*np.log(q_k),0)

        return mult_bound,q_k
    
    
    #===========================================================================
    # Intermediate values for the non batch global parameters
    #===========================================================================
    def global_params_VI(self):
        '''
        Function to calculate the intermediate values for the non-batch global parameters.
        Implemented with simple VI updates
        
        Parameters
        ----------
        
        miniBatch: ndarray
            The minibatch of the dataset in which we calculate the gradients
        
        Returns
        -------
        gradients: list
            List of ndarrays containing the intermediate values for the non batch global parameters
        '''
#         self.params=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
#         self.local_params=[t_s,t_m,zeta]
#         self.batch_params=[t_b,t_e_1,t_e_2,t_xi,t_l,t_mu,t_a]
        #simple VI
        #print('Performing simple VI for global parameters...')
        
        q_z=expit(self.omega)
        
        mult_bound,q_k=self.mult_bound_calc()
        
         
        #tilde tau, that's tricky
        first_sum=np.zeros((self.K,1))
        second_sum=np.zeros((self.K,1))
        for k in range(self.K):
            for m in range(k+1,self.K):
                first_sum[k,0]+=(self.D-q_z[:,m].sum())*(q_k[k+1:m+1,0].sum())
            second_sum[k,0]=(q_z[:,k:]).sum()
             
        #here tilde tau
        #self.t_tau*=(1.0-self.rho)
        self.t_tau=(self.t_g_1/self.t_g_2+first_sum+second_sum)
         
        #omega
        #self.omega*=(1.0-self.rho)
        for k in range(self.K):
            self.omega[:,k]=((psi(self.t_tau[:k+1,0])-psi(self.t_tau[:k+1,0]+self.h_tau[:k+1,0])).sum()-mult_bound[k]-0.5*np.log(2*np.pi)+0.5*(psi(self.t_c[k,0])-np.log(self.t_f[k,0]))\
                    -0.5*(self.t_c[k,0]/self.t_f[k,0])*(self.t_mu[:,k]**2+self.t_l[:,k]))
                    
         
        #hat_tau
        #self.h_tau*=(1.0-self.rho)
        for k in range(self.K):
            self.h_tau[k]=(1.0+(self.D-q_z[:,k:].sum(0)).sum()*q_k[k])
         
             
        #tilde c
        #self.t_c*=(1.0-self.rho)
        self.t_c=(self.c+0.5*q_z.sum(0).reshape(-1,1))
         
        #tilde_f
        #self.t_f*=(1.0-self.rho)
        self.t_f=(self.f+(self.t_l+self.t_mu**2).sum(0).reshape(-1,1))
         
        #update t_g_1
        #self.t_g_1*=(1.0-self.rho)
        self.t_g_1=(self.gamma_1+self.K-1)
         
        #update t_g_2
        #self.t_g_2*=(1.0-self.rho)
        self.t_g_2=(self.gamma_2-(psi(self.t_tau[:self.K-1])-psi(self.t_tau[:self.K-1]+self.h_tau[self.K-1])).sum())
        
        self.params=[self.t_tau,self.omega,self.h_tau,self.t_c,self.t_f,self.t_g_1,self.t_g_2]

      
    #===========================================================================
    # Get the gradients for the global batch parameters
    #===========================================================================
    def getBatchGradients(self,x,s):
        '''
        Calculates the gradients for the batch parameters. For each s, get gradient as if we have seen x_s N(or S?) times
        
        Parameters
        ----------
        
        miniBatch: ndarray
            Array containing the subset of the data to calculate the gradients
        s: int
            Index to the current element of the minibatch
            
        Returns
        -------
        batch_params: list
            List of ndarrays with the gradients for the datapoint s
        '''
        
        #=======================================================================
        # some gradient ascent thingy here. 
        #=======================================================================
        #self.params=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
        #self.local_params=[t_s,t_m,zeta]
        #self.batch_params=[t_b,t_e_1,t_e_2,t_xi,t_l,t_mu,t_a]
        batch_gradients=[0]*len(self.batch_params)
        
        #tilde b update
        temp=0
        for d in range(self.D):
            temp+=(x[s,d]**2 -2*x[s,d]*(self.t_mu[d,:]*self.t_m[s,:]).sum()\
                                            +(self.t_mu[d,:]*self.t_m[s,:]).sum()**2\
                                            +(((self.t_mu[d,:]**2+self.t_l[d,:])*(self.t_m[s,:]**2+self.t_s)).sum())\
                                            -(self.t_mu[d,:]**2*self.t_m[s,:]**2).sum()
                                            )
        batch_gradients[0]=(1.0/self.S)*(self.b+0.5*self.N*temp)
        
        #t_e_1 update
        batch_gradients[1]=(1.0/self.S)*(self.eta_1+0.5*(self.N)*(self.zeta[s,:,:]))
        
        #t_e_2 update
        batch_gradients[2]=(1.0/self.S)*(self.eta_2+0.5*(self.N)*self.zeta[s,:,:]*(self.t_s[s,:]+self.t_m[s,:]**2).reshape(-1,1))
        
        #tilde xi update
        batch_gradients[3]=(1.0/self.S)*(self.xi+(self.N)*self.zeta[s,:,:])
         
        #tilde lambda update
        batch_gradients[4]=np.zeros((self.D,self.K))
        for d in range(self.D):
            batch_gradients[4][d,:]=self.N*(self.t_a/self.t_b)*(self.t_s[s,:]+self.t_m[s,:]**2).reshape(-1,)\
                                     +(self.t_c/self.t_f).reshape(-1,)
        
        #tilde mu update
        batch_gradients[5]=np.zeros((self.D,self.K))
        for d in range(self.D):
            batch_gradients[5][d,:]=(self.N/self.S)*(self.t_a/self.t_b)*(batch_gradients[4][d,:]**(-1))*self.t_m[s,:]*(x[s,d].T-(self.t_mu[d,:]*self.t_m[s,:]).sum())
        
        #normalize after using it in batch gradient of tilde mu else we have a nan bound
        batch_gradients[4]*=(1.0/self.S)
        
        #tilde a update
        batch_gradients[6]=(1.0/self.S)*(self.a+0.5*self.N*self.D)
        
      
        return batch_gradients
    
       
        
    #===========================================================================
    # Perform one iteration for the algorithm
    #===========================================================================
    def iterate(self,x,miniBatch_indices):
        '''
        Function that performs one iteration of the SVI IBP ICA algorithm. This includes:
            1) Update the local parameters
            2) Get the intermediate values for the batch gradients
            3) Update Params:
                3.1) Get intermediate values for the batch global params
                3.2) Gradient step for all the global parameters dependent on the sufficient stats
        
        Parameters
        ----------
        
        miniBatch: ndarray
            miniBatch of the dataset with which we calculate the noisy gradients 
            
        '''
        
        ###########################################
        # LOCAL PARAMETERS UPDATE                #
        ###########################################
        
        #update the local parameters
        self.updateLocalParams(x,miniBatch_indices)
        
        
        #============================================
        # GLOBAL PARAMETERS UPDATE                
        #============================================
        
       
        
        #for the batch global parameters get the intermediate values 
        #print('Batch Parameters intermediate values calculation...')
        intermediate_values_batch_all=[0]*len(self.batch_params)
        
        #for each datapoint calculate gradient and sum over all datapoints
        for s in range(len(miniBatch)):
            batch_params=self.getBatchGradients(x,miniBatch_indices[s])
           
            for i in range(len(intermediate_values_batch_all)):
                intermediate_values_batch_all[i]+=batch_params[i]
            
        #Ready for the final step for this iteration
        #print("Updating Global Parameters...")
            

        #update the batch global params
        for i in range(len(self.batch_params)):
            self.batch_params[i]*=(1-self.rho)
            self.batch_params[i]+=(self.rho)*intermediate_values_batch_all[i]
        self.t_b,self.t_e_1,self.t_e_2,self.t_xi,self.t_l,self.t_mu,self.t_a=self.batch_params
        
        
        
    
    #===========================================================================
    # Update the local parameters with gradient steps until convergence
    #===========================================================================
    def updateLocalParams(self,x,mb_indices):
        '''
        Update the local parameters for the IBP-ICA model (One gradient step)
        
        Parameters
        ----------
        
        miniBatch: ndarray
            The minibatch for this run of the algorithm
        '''
        
        #print("Updating local parameters...")

        self.t_s[mb_indices,:]=((self.t_a/self.t_b)*(np.diag(np.dot(self.t_mu.T,self.t_mu))+self.t_l.sum(0).T)\
                                        +(self.zeta[mb_indices,:,:]*(self.t_e_1/self.t_e_2)).sum(2))**-1

        self.t_m[mb_indices,:]=(self.t_s[mb_indices,:]*(self.t_a/self.t_b)*(np.dot(self.t_mu.T,x[mb_indices,:].T)).T)
        
        for n in range(self.S):
            self.zeta[mb_indices[n],:,:]=(np.exp(psi(self.t_xi)-psi(self.t_xi.sum(1)).reshape(-1,1)\
                                                   +0.5*(-np.log(2*np.pi)+psi(self.t_e_1)-np.log(self.t_e_2)-(self.t_e_1/self.t_e_2)*(self.t_m[mb_indices[n],:]**2+self.t_s[mb_indices[n],:]).reshape(-1,1))))
            self.zeta[mb_indices[n],:,:]/=self.zeta[mb_indices[n],:,:].sum(1).reshape(-1,1)
           
           
    #CALCULATION OF THE LOWER BOUND,
    #SWITCHED FROM THEANO, MUST CHECK SOME STUFF 
    def calculate_lower_bound(self,miniBatch,ind=None):
        a=self.a
        b=self.b
        c=self.c
        f=self.f
        g_1=self.gamma_1
        g_2=self.gamma_2
        e_1=self.eta_1
        e_2=self.eta_2
        xi=self.xi
        
        K=self.K
        D=self.D
        J=self.J
        N=miniBatch.shape[0]
        
        x=miniBatch
        
        if (not (ind is None)):
            t_m=self.t_m[ind,:]
            zeta=self.zeta[ind,:]
            t_s=self.t_s[ind,:]
        else:
            t_m=self.t_m
            zeta=self.zeta
            t_s=self.t_s
            
        omega=self.omega
        t_mu=self.t_mu
        t_a=self.t_a
        t_b=self.t_b
        t_g_1=self.t_g_1
        t_g_2=self.t_g_2
        t_e_1=self.t_e_1
        t_e_2=self.t_e_2
        t_xi=self.t_xi
        t_l=self.t_l
        
        
        t_c=self.t_c
        t_f=self.t_f
        t_tau=self.t_tau
        h_tau=self.h_tau
        
        mult_bound,_=self.mult_bound_calc()
        
        q_z=expit(omega)
        
        
        final=np.zeros((N,D))
        for n in range(N):
            for d in range(D):
                final[n,d]=(x[n,d]**2-2*x[n,d]*(t_m[n,:]*t_mu[d,:]).sum()\
                        + ((t_mu[d,:]*t_m[n,:])).sum()**2\
                        +((t_mu[d,:]**2+t_l[d,:])*(t_m[n,:]**2+t_s[n,:])-(t_mu[d,:]**2)*(t_m[n,:]**2)).sum())
                
        final*=-0.5*(t_a/t_b)
        
        final_y=np.zeros((N,K))
        for n in range(N):
            for k in range(K):
                for j in range(J):
                    final_y[n,k]+=zeta[n,k,j]*(0.5*(psi(t_e_1[k,j])-np.log(t_e_2[k,j]))\
                                                                     -0.5*(t_e_1[k,j]/t_e_2[k,j])*(t_s[n,k]+t_m[n,k]**2))
                    
        #=======================================================================
        # calculate all the individual terms for calculating the likelihood
        # this way it's easier to debug 
        # The names are pretty self-explanatory 
        #=======================================================================
        
        expectation_log_p_a=g_1*np.log(g_2)+(g_1-1)*(psi(t_g_1)-np.log(t_g_2))-g_2*(t_g_1/t_g_2)-gammaln(g_1)
        
        
        expectation_log_p_phi=a*np.log(b)+(a-1)*(psi(t_a)-np.log(t_b))-b*(t_a/t_b)-gammaln(a)
        
        
        expectation_log_u_k=psi(t_g_1)-np.log(t_g_2)+(t_g_1/t_g_2-1)*(psi(t_tau)-psi(t_tau+h_tau))
        
        
        expectation_log_lambda_k=c*np.log(f)+(c-1)*(psi(t_c)-np.log(t_f))-f*(t_c/t_f)-gammaln(c)
        
        
        expectation_log_varpi=-gammaln(xi).sum(1)+gammaln(xi.sum(1))+((xi-1)*(psi(t_xi)-psi(t_xi.sum(1)).reshape(-1,1))).sum(1)
        
        
        expectation_log_skj=e_1*np.log(e_2)+(e_1-1)*(-np.log(t_e_2)+psi(t_e_1))-e_2*(t_e_1/t_e_2)-gammaln(e_1)
        
        
        expectation_log_gdk=+0.5*(psi(t_c)-np.log(t_f)).sum()-0.5*((t_c/t_f).T*(t_mu**2+t_l)).sum(1)
        
        
        
        expectation_log_zdk=q_z*np.cumsum(psi(h_tau)-psi(t_tau+h_tau))+(1.0-q_z)*mult_bound.T
        
        
        expectation_log_varepsilon=zeta*(psi(t_xi)-psi(t_xi.sum(1)).reshape(-1,1))
        
        
        expectation_log_y=final_y
        
        
        expectation_log_x=0.5*(psi(t_a)-np.log(t_b))+final
         
      
        #=======================================================================
        # Combine all the terms to get the likelihood
        #=======================================================================
        likelihood=expectation_log_p_a \
                    +expectation_log_p_phi \
                    +expectation_log_u_k.sum() \
                    +expectation_log_lambda_k.sum() \
                    + expectation_log_varpi.sum() \
                    + expectation_log_skj.sum() \
                    + expectation_log_gdk.sum() \
                    + expectation_log_zdk.sum()\
                    + expectation_log_varepsilon.sum()\
                    + expectation_log_y.sum() \
                    + expectation_log_x.sum()
        
        
        
        #=======================================================================
        # calculate the entropy term  
        # Maybe should do individual lines again for easier debug.
        # Checked many times but check again
        # Each line is a different entropy term                   
        #=======================================================================
        entropy=t_g_1-np.log(t_g_2)+(1.0-t_g_1)*psi(t_g_1)+gammaln(t_g_1) \
                +t_a-np.log(t_b)+(1.0-t_a)*psi(t_a)+gammaln(t_a) \
                +(gammaln(t_tau)+gammaln(h_tau)-gammaln(h_tau+t_tau)-(t_tau-1.0)*psi(t_tau)-(h_tau-1.0)*psi(h_tau)+(t_tau+h_tau-2)*psi(t_tau+h_tau)).sum() \
                +(t_c-np.log(t_f)+gammaln(t_c)+(1.0-t_c)*psi(t_c)).sum() \
                +(gammaln(t_xi).sum(1)-gammaln(t_xi.sum(1))-(J-t_xi.sum(1))*psi(t_xi.sum(1))-((t_xi-1.0)*psi(t_xi)).sum(1)).sum() \
                +(t_e_1-np.log(t_e_2)+gammaln(t_e_1)+(1.0-t_e_1)*psi(t_e_1)).sum() \
                +0.5*(np.log(t_l).sum(1)).sum() \
                -((1.0-q_z)*np.log(1.0-q_z)+q_z*np.log(q_z)).sum() \
                +0.5*(np.log(t_s).sum(1)).sum() \
                -(zeta*np.log(zeta)).sum()
        
        #The evidence lower bound is the likelihood plus the entropy
        lower_bound=likelihood+entropy
        return lower_bound
        
           
    #===========================================================================
    # Calculate the lower bound and create gradient functions for all parameters
    #===========================================================================

    
    #===========================================================================
    # METROPOLIS HASTINGS PART
    #===========================================================================
    def unused_components(self):
        '''
        Check for unused components and reshape arrays if necessary
        '''
        used_k=self.search_used_components()
        if (len(used_k)!=self.K):
            print('Found Unused Components...')
            self.contract_arrays(used_k)
            print('Current K:',self.K)
            self.K=len(used_k)
            print('New K after contract',self.omega.shape[1])
            
    def search_used_components(self):
        '''
        Find which components are being used
        '''
        
        threshold=10**-3
        q=expit(self.omega)
        used_k=[]
        for k in range(self.K):
            if (np.where(q[:,k]>threshold)[0]).size>0:
                used_k.append(k)
        return used_k
    
    #===============================================================================
    # need to remove columns if feature is not active
    #===============================================================================
    def contract_arrays(self,used_k):
#         simpleVariationalVariables=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
#         localVar=[t_s,t_m,zeta]
#         batch_grad_vars=[t_b,t_e_1,t_e_2,t_xi,t_l,t_mu,t_a]
        #row delete
        
        self.xi=self.xi[used_k,:]
        
        
        #column delete
        self.t_s=self.t_s[:,used_k]
        self.t_m=self.t_m[:,used_k]
        self.zeta=self.zeta[:,used_k,:]
        
        self.t_e_1=self.t_e_1[used_k,:]
        self.t_e_2=self.t_e_2[used_k,:]
        self.t_xi=self.t_xi[used_k,:]
        self.t_l=self.t_l[:,used_k]
        self.t_mu=self.t_mu[:,used_k]
        
        self.t_tau=self.t_tau[used_k,:]
        self.omega=self.omega[:,used_k]
        self.h_tau=self.h_tau[used_k,:]
        self.t_c=self.t_c[used_k,:]
        self.t_f=self.t_f[used_k,:]
        
        self.batch_params=[self.t_b,self.t_e_1,self.t_e_2,self.t_xi,self.t_l,self.t_mu,self.t_a]
        self.local_params=[self.t_s,self.t_m,self.zeta]
        self.params=[self.t_tau,self.omega,self.h_tau,self.t_c,self.t_f,self.t_g_1,self.t_g_2]
        
        
    #=======================================================================
    # need to expand the parameters after the feature update
    #=======================================================================
    def expand_arrays(self,diff):
        
        print("Expand arrays..")
        self.xi=np.vstack((self.xi,np.ones((diff,self.J))))
        
        
        self.t_s=np.hstack((self.t_s,np.random.gamma(1,1,size=(self.N,diff))))
        self.t_m=np.hstack((self.t_m,np.random.normal(0,1,size=(self.N,diff))))
        self.zeta=np.hstack((self.zeta,np.ones((self.N,diff,self.J))*(1.0/self.J)))
        
        self.t_e_1=np.vstack((self.t_e_1,np.ones((diff,self.J))))
        self.t_e_2=np.vstack((self.t_e_2,np.ones((diff,self.J))))
        self.t_xi=np.vstack((self.t_xi,np.ones((diff,self.J))*(1.0/self.J)))
        self.t_l=np.hstack((self.t_l,np.ones((self.D,diff))))
        self.t_mu=np.hstack((self.t_mu,np.random.normal(0,1,(self.D,diff))))
        
        self.t_tau=np.vstack((self.t_tau,np.ones((diff,1))))
        self.omega=np.hstack((self.omega,np.random.random((self.D,diff))))
        self.h_tau=np.vstack((self.h_tau,np.ones((diff,1))))
        self.t_c=np.vstack((self.t_c,np.ones((diff,1))))
        self.t_f=np.vstack((self.t_f,np.ones((diff,1))))
        
        self.batch_params=[self.t_b,self.t_e_1,self.t_e_2,self.t_xi,self.t_l,self.t_mu,self.t_a]
        self.local_params=[self.t_s,self.t_m,self.zeta]
        self.params=[self.t_tau,self.omega,self.h_tau,self.t_c,self.t_f,self.t_g_1,self.t_g_2]
        
    #===========================================================================
    # Save the parameters to examine after experiments
    #===========================================================================
    def save_params(self,iteration,start_time,LL):
        
        if not os.path.exists(start_time):
            os.makedirs(start_time)
            
        with open(start_time+'/params'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(self.params, f)
        with open(start_time+'/localparams'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(self.local_params, f)
        with open(start_time+'/batchparams'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(self.batch_params, f)
        with open(start_time+'/bound_iter_'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(LL, f)
                
    
    #===========================================================================
    # Some naive creation of synthetic data
    #===========================================================================
    def create_synthetic_data(self,components=5):
        G=np.random.normal(0,1,size=(self.D,components))
        y=np.random.normal(0,1,size=(components,self.N))
        #return np.dot(G,y).T,G,y
        x=sio.loadmat('data/test_ibp_ica.mat')
        x=x["X"]
        return x
        
    
#===============================================================================
# Main 
#===============================================================================
    
if __name__ == '__main__':
    
    #some initial variables
    initN=1000
    initD=4
    initJ=3
    initS=50
    
    #x=datasets.load_iris()
    #x=x.data
    #initN,initD=x.shape
    #initialize IBP_ICA object
    z=IBP_ICA(5,initD,initJ,initS,initN)
    
    #create some synthetic data
    x   =z.create_synthetic_data(5)
    z.N=x.shape[0]
    z.D=x.shape[1]
    
    #init the posterior parameters
    z.init_params()
    
    #create lower bound and get gradients
    #z.calculate_lower_bound(x)
    #z.createGradientFunctions()
    
   
    iteration=0
    max_iter=500
    
    #keep the lower bound for each iteration
    LL=np.zeros((max_iter,int(z.N/z.S)))
    
    start_time=str(iteration)+(time.strftime("%Y-%m-%d-%H:%M").replace(" ","_")+'_batch_size_'+str(initS)+'_D_'+str(initD))
    global_min=0
    #repeat until maxi iterations
    st=time.time()
    while iteration<max_iter:
        if (iteration % 10==0):
            print('ELBO at iteration',iteration,":",z.calculate_lower_bound(x))
        
        iteration+=1
        print("Stochastic IBP-ICA iteration: ",iteration)
        #set step size for this iteration (Paisley et al.) 
        z.rho=0.0005
        
        #create all the random minibatches for this iteration
        random_indices=np.arange(z.N)
        np.random.shuffle(random_indices)
        random_indices=random_indices.reshape((int(z.N/z.S),z.S))
        
        #FOR ALL THE MINIBATCHES update the local parameters and the global parameters with SGD
        current_minibatch=0
        
        
        
        for miniBatch_indices in random_indices:
            #print('\n')
            #print('#########################################')
            #print('# Processing miniBatch',current_minibatch+1,'at iteration',iteration,'#')
            #print('#########################################')
            current_minibatch+=1
            miniBatch=x[miniBatch_indices,:]
            #print the lower bound before updating the parameters
            #elbo=z.calculate_lower_bound(miniBatch,miniBatch_indices)
            #elbo1=z.calculate_lower_bound(x)

            #print('-----------------------------------------------------------------------------------------------------')
            #print('The lower bound with the new batch before optimisation is',elbo)
            #print('The lower bound with the whole dataset before optimisation is',elbo1)
            #print('-----------------------------------------------------------------------------------------------------')
            
            #time.sleep(2)
            
            #perform one iteration of the algorithm 
            z.iterate(x,miniBatch_indices)

           
            
            #append lower bound to list
            LL[iteration-1,current_minibatch-1]=(z.calculate_lower_bound(miniBatch,miniBatch_indices))
            #elbo1=z.calculate_lower_bound(x)

            #print the lower bound to see how it goes
            #print('------------------------------------------------------------------')
            #print("Lower Bound at iteration",iteration,"and minibatch ",current_minibatch,"after optimisation is ",LL[iteration-1,current_minibatch-1])
            #print("Lower Bound (whole dataset) at iteration",iteration,"and minibatch ",current_minibatch,"after optimisation is ",elbo1)
            #print('------------------------------------------------------------------')
                        
            if (np.isnan(LL[iteration-1,current_minibatch-1])):
                sys.exit('Why is the bound nan? Please Debug.')
            
        #update the non batch global params
        z.global_params_VI()
        #print(LL[iteration-1,:])
        #time.sleep(5)
        z.feature_update(miniBatch)             
        print(z.K)
            
            
        z.save_params(iteration,start_time,LL[iteration-1,:])
        
    global_min=np.max(LL)
    global_min_ind=np.argmax(LL)
    print('------------------------------------------------------------------')
    print('The global max is ',global_min,'found in iteration',global_min_ind)
    print('------------------------------------------------------------------')
    print('Total running time:',time.time()-st)
    z.save_params("final",start_time,LL)

    print(LL)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#===============================================================================
# DEPRECATED STUFF. KEEP HERE JUST IN CASE
#===============================================================================
#         
#I think these are correct but I also have a nested scan in the deprecated stuff just in case
#         try_sum,_=theano.scan(lambda i, qk,ttau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i])*(T.sum(qk[:i])-T.cumsum(qk[:i-1]))),T.constant(0.0,dtype='float32')),
#                             sequences=T.arange(K),
#                             non_sequences=[q_k,t_tau],
#                             strict=True
#                             )
#         
#         try_sum2,_=theano.scan(lambda i, qk,ttau,htau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i+1]+htau[:i+1])*(T.cumsum(qk[:i])[::-1])),T.constant(0.0,dtype='float32')),
#                             sequences=T.arange(K),
#                             non_sequences=[q_k,t_tau,h_tau],
#                             strict=True
#                             )
#         
#     
#       
#         #the multinomial bound for calculating p(z_{dk}|\upsilon_k)
#         mult_bound=T.cumsum(q_k*T.psi(h_tau))+try_sum-try_sum2-T.cumsum(q_k*T.log(q_k))
#     def createGradientFunctions(self):
#         '''
#         The main function of the implementation. 
#         Create the variational lower bound and take derivatives with respect to the parameters.
#         
#         In the current version the original parameters have been replaced with original=exp(new_param) and we take derivatives 
#         with respect to new_param. This is the log trick and is used for imposing positivity constraints on the needed parameters.
#         
#         This function assigns theano.functions to the IBP_ICA object.
#         
#         '''
#         
#         print("Creating gradient functions...")
#         print("\t Initializing prior parameters...")
#         
#         #Some constant priors
#         a=T.constant(2.0,dtype='float32')
#         b=T.constant(2.0,dtype='float32')
#         c=T.constant(2.0,dtype='float32')
#         f=T.constant(2.0,dtype='float32')
#         g_1=T.constant(2.0,dtype='float32')
#         g_2=T.constant(2.0,dtype='float32')
#         e_1=T.constant(2.0,dtype='float32')
#         e_2=T.constant(2.0,dtype='float32')
# 
#         #Create some needed scalar variables
#         K=T.scalar('K',dtype='int32')
#         D=T.scalar('D',dtype='int32')      
#         S=T.scalar('S',dtype='int32')
#         J=T.scalar('J',dtype='int32')
#         xi=T.matrix('xi',dtype='float32')
#         
#         #The matrix for the data
#         x=T.matrix('x',dtype='float32')
#          
#         
#         #Parameters that do not need the log trick
#         omega=T.fmatrix('omega')
#         t_mu=T.fmatrix('t_mu')
#         t_m=T.fmatrix('t_m')
#          
#         #need to be positive scalars, so create some y variables so that e.g., t_a=exp(t_a_y)
#         t_a=T.scalar('t_a',dtype='float32')
#         t_b=T.scalar('t_b',dtype='float32')
#         t_g_1=T.scalar('t_g',dtype='float32')
#         t_g_2=T.scalar('t_g',dtype='float32')
#          
#         #tilde_eta_1,tilde_eta_2,tilde_xi,tilde_l,tilde_mu,omega,tilde_s, tilde_m and zeta are matrices
#         t_e_1,t_e_2,t_xi,t_l,t_s=T.fmatrices('t_e_1','t_e_2','t_xi','t_l','t_s')
#          
#         #the only tensor we got
#         zeta=T.ftensor3('zeta')
#      
#         #the rest are columns
#         t_c=T.col('t_c',dtype='float32')
#         t_f=T.col('t_f',dtype='float32')
#         t_tau=T.col('t_tau',dtype='float32')
#         h_tau=T.col('h_tau',dtype='float32')
#    
#         print('\t Creating the bound terms...')
#         #Calculate q_k as seen in Chatzis et al.
#         cs=T.cumsum(T.psi(t_tau),0)-T.psi(t_tau)
#         q_k=T.exp(T.psi(h_tau)+cs-T.cumsum(T.psi(t_tau+h_tau),0))
#         q_k/=T.sum(q_k)
#        
#         mult_bound=T.fcol('mult_bound')
#         
#     
#         
#         #calculate q(z_{dk}=1)
#         q_z=T.nnet.sigmoid(omega)
#         
#         
#         #IS THIS WRONG?
#         #THIS AND FINAL SEEM TO BE THE TERMS THAT CREATE PROBLEMS WHEN UPDATING T_A AND T_B
#         def inter_loop(n,xn,ts,tm,tmu,tl,ta,tb,K):
#             intermediate=x[n,:]**2-2*x[n,:]*T.sum(tm[n,:]*tmu,1).T\
#                         + (T.sum(tmu*tm[n,:],1)**2).T\
#                         +T.sum( (tmu**2+tl)*(tm[n,:]**2+ts[n,:])-(tmu**2)*(tm[n,:]**2),1).T   
#                         
#             return -0.5*(ta/tb)*(intermediate)
# 
#         final,_=theano.scan(fn=inter_loop,
#                             sequences=T.arange(S),
#                             non_sequences=[x,t_s,t_m,t_mu,t_l,t_a,t_b,K],
#                             strict=True
#                             )
#         
# 
#         
#         #=======================================================================
#         # Calculate the expectation of log p(y)
#         #=======================================================================
#         def y_calc(j,k,n,zt,te1,te2,ts,tm):
#             return zt[n,k,j]*(-0.5*T.log(2*np.pi)+0.5*(T.psi(te1[k,j])-T.log(te2[k,j]))-0.5*(te1[k,j]/(te2[k,j]))*(ts[n,k]+tm[n,k]**2))
#         
#         def deepest_loop(k,n,zt,te1,te2,ts,tm):
#             out,_=theano.scan(fn=y_calc,
#                               sequences=T.arange(J),
#                               non_sequences=[k,n,zt,te1,te2,ts,tm],
#                               strict=True
#                               )
#             return out 
#         
#         def not_so_deep_loop(n,zt,te1,te2,ts,tm):
#             out,_=theano.scan(fn=deepest_loop,
#                               sequences=T.arange(K),
#                               non_sequences=[n,zt,te1,te2,ts,tm],
#                               strict=True
#                               )
#             return out
#         
#         final_y,_=theano.scan(fn=not_so_deep_loop,
#                               sequences=T.arange(S),
#                               non_sequences=[zeta,t_e_1,t_e_2,t_s,t_m],
#                               strict=True
#                               )
#       
#         #=======================================================================
#         # calculate all the individual terms for calculating the likelihood
#         # this way it's easier to debug 
#         # The names are pretty self-explanatory 
#         #=======================================================================
#         
#         expectation_log_p_a=g_1*T.log(g_2)+(g_1-1)*(T.psi(t_g_1)-T.log(t_g_2))-g_2*(t_g_1/t_g_2)-T.gammaln(g_1)
#         
#         
#         expectation_log_p_phi=a*T.log(b)+(a-1)*(T.psi(t_a)-T.log(t_b))-b*(t_a/t_b)-T.gammaln(a)
#         
#         
#         expectation_log_u_k=T.psi(t_g_1)-T.log(t_g_2)+(t_g_1/t_g_2-1)*(T.psi(t_tau)-T.psi(t_tau+h_tau))
#         
#         
#         expectation_log_lambda_k=c*T.log(f)+(c-1)*(T.psi(t_c)-T.log(t_f))-f*(t_c/t_f)-T.gammaln(c)
#         
#         
#         expectation_log_varpi=-T.sum(T.gammaln(xi),1)+T.gammaln(T.sum(xi,1))+T.sum((xi-1)*(T.psi(t_xi)-(T.psi(T.sum(t_xi,1))).dimshuffle(0, 'x')),1)
#         
#         
#         expectation_log_skj=e_1*T.log(e_2)+(e_1-1)*(-T.log(t_e_2)+T.psi(t_e_1))-e_2*(t_e_1/t_e_2)-T.gammaln(e_1)
#         
#         
#         expectation_log_gdk=-0.5*K*T.log(2*np.pi)+T.sum(0.5*(T.psi(t_c)-T.log(t_f)))-T.sum(0.5*(t_c/t_f).T*(t_mu**2+t_l),1)
#         
#         
#         
#         expectation_log_zdk=q_z*T.cumsum(T.psi(h_tau)-T.psi(t_tau+h_tau))+(1.0-q_z)*mult_bound.T
#         
#         
#         expectation_log_varepsilon=zeta*(T.psi(t_xi)-T.psi(T.sum(t_xi,1)).dimshuffle(0,'x'))
#         
#         
#         expectation_log_y=final_y
#         
#         
#         expectation_log_x=-0.5*T.log(2*np.pi)+0.5*(T.psi(t_a)-T.log(t_b))+final
#          
#       
#         #=======================================================================
#         # Combine all the terms to get the likelihood
#         #=======================================================================
#         likelihood=expectation_log_p_a \
#                     +expectation_log_p_phi \
#                     +T.sum(expectation_log_u_k) \
#                     +T.sum(expectation_log_lambda_k) \
#                     +T.sum(expectation_log_varpi) \
#                     +T.sum(expectation_log_skj) \
#                     +T.sum(expectation_log_gdk) \
#                     +T.sum(expectation_log_zdk)\
#                     +T.sum(expectation_log_varepsilon)\
#                     +T.sum(expectation_log_y) \
#                     +T.sum(expectation_log_x)     
#                                           
#         #=======================================================================
#         # calculate the entropy term  
#         # Maybe should do individual lines again for easier debug.
#         # Checked many times but check again
#         # Each line is a different entropy term                   
#         #=======================================================================
#         entropy=t_g_1-T.log(t_g_2)+(1-t_g_1)*T.psi(t_g_1)+T.gammaln(t_g_1) \
#                 +t_a-T.log(t_b)+(1-t_a)*T.psi(t_a)+T.gammaln(t_a) \
#                 +T.sum(T.gammaln(t_tau)+T.gammaln(h_tau)-T.gammaln(h_tau+t_tau)-(t_tau-1)*T.psi(t_tau)-(h_tau-1)*T.psi(h_tau)+(t_tau+h_tau-2)*T.psi(t_tau+h_tau)) \
#                 +T.sum(t_c-T.log(t_f)+T.gammaln(t_c)+(1-t_c)*T.psi(t_c)) \
#                 +T.sum(T.sum(T.gammaln(t_xi),1)-T.gammaln(T.sum(t_xi,1))-(J-T.sum(t_xi,1))*T.psi(T.sum(t_xi,1))-(T.sum((t_xi-1.0)*T.psi(t_xi),1))) \
#                 +T.sum(t_e_1-T.log(t_e_2)+T.gammaln(t_e_1)+(1-t_e_1)*T.psi(t_e_1)) \
#                 +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.sum(T.log(t_l),1)) \
#                 -T.sum((1.0-q_z)*T.log(1.0-q_z))-T.sum(q_z*T.log(q_z)) \
#                 +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.sum(T.log(t_s),1)) \
#                 -T.sum(zeta*T.log(zeta))
# 
#         #The evidence lower bound is the likelihood plus the entropy
#         lower_bound=likelihood+entropy
#         
#         
#         #=======================================================================
#         # set local and global gradient variables
#         #=======================================================================
#         #to take derivatives wrt to the log
#         #Reminder: e.g., t_g_1=T.exp(t_g_1_y)
#         print('\t Calculating Derivatives of the lower bound...')
#         simpleVariationalVariables=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
#         localVar=[t_s,t_m,zeta]
#         batch_grad_vars=[t_b,t_e_1,t_e_2,t_xi,t_l,t_mu,t_a]
#         
#         
#         #=======================================================================
#         # calculate the derivatives
#         #=======================================================================
#         #derivatives_local=T.grad(lower_bound,localVar)
#         #derivatives_batch=T.grad(lower_bound,batch_grad_vars)
#         
#         print('\t Creating functions...')
#         #self.localGradientFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,J,D,S],derivatives_local,allow_input_downcast=True,on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
#         #self.batchGradientFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,J,D,S],derivatives_batch,allow_input_downcast=True,on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
#         self.lowerBoundFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,D,J,S,mult_bound],lower_bound,allow_input_downcast=True,on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
#     
#         #simple VI
#         print('Performing simple VI for global parameters...')
#         #qk calculation 
#         q_k=np.exp(psi(self.params[2])+(np.cumsum(psi(self.params[0]),0)-psi(self.params[0]))-np.cumsum(psi(self.params[0]+self.params[2]),0))
#         q_k/=q_k.sum()
#         q_z=1.0/(1.0+np.exp(-self.params[1]))
#         
#         #mult bound
#         #check this later just to be sure
#         second_sum=np.zeros((K,1))
#         third_sum=np.zeros((K,1))
#         for k in range(self.K):
#             for m in range(k-1):
#                 if (k==0):
#                     second_sum[k,0]=0
#                 second_sum[k,0]+=q_k[m+1:k].sum()*psi(self.params[0][m,0])
#             for m in range(k):
#                 third_sum+=q_k[m:k,0].sum()*psi(self.params[0][m,0]+self.params[2][m,0])
#             
#         mult_bound=np.cumsum(q_k*psi(self.params[2]),0)+second_sum-third_sum-np.cumsum(q_k*np.log(q_k),0)
#         
#         
#         #tilde tau, that's tricky
#         first_sum=np.zeros((K,1))
#         second_sum=np.zeros((K,1))
#         for k in range(self.K):
#             for m in range(k+1,self.K):
#                 first_sum[k,0]+=(self.D-q_z[:,m].sum())*q_k[k+1:m,0].sum()
#             second_sum[k,0]=(q_z[:,k:]).sum()
#             
#         self.params[0]=self.params[5]/self.params[6]+first_sum+second_sum
#         
#         #omega
#         for k in range(self.K):
#                 self.params[1][:,k]=np.sum(psi(self.params[0][:k+1,0])-psi(self.params[0][:k+1,0]+self.params[2][:k+1,0]),0)+mult_bound[k]-0.5*np.log(2*np.pi)+0.5*(psi(self.params[3][k,0])-np.log(self.params[4][k,0]))\
#                     -0.5*(self.params[3][k,0]/self.params[4][k,0])*(self.batch_params[5][:,k]**2+self.batch_params[4][:,k])
#         
#         #hat_tau
#         for k in range(self.K):
#             self.params[2][k]=1.0+(D-q_z[:,k:]).sum()*q_k[k]
#         
#             
#         #tilde c
#         self.params[3]=self.c+0.5*q_z.sum(0).reshape(-1,1)
#         
#         #tilde_f
#         self.params[4]=self.f+(self.batch_params[4]+self.batch_params[5]**2).sum(0).reshape(-1,1)
#         
#         #update t_g_1
#         self.params[5]=self.gamma_1+self.K-1
#         
#         #update t_g_2
#         self.params[6]=self.gamma_2-np.sum(psi(self.params[0][:self.K-1])-psi(self.params[0][:self.K-1]+self.params[2][self.K-1]))
#         
 #simple VI
#         print('Performing simple VI for global parameters...')
#         #qk calculation 
#         q_k=np.exp(psi(self.params[2])+(np.cumsum(psi(self.params[0]),0)-psi(self.params[0]))-np.cumsum(psi(self.params[0]+self.params[2]),0))
#         q_k/=q_k.sum()
#         q_z=1.0/(1.0+np.exp(-self.params[1]))
#         
#         #mult bound
#         #check this later just to be sure
#         second_sum=np.zeros((K,1))
#         third_sum=np.zeros((K,1))
#         for k in range(self.K):
#             for m in range(k-1):
#                 if (k==0):
#                     second_sum[k,0]=0
#                 second_sum[k,0]+=q_k[m+1:k].sum()*psi(self.params[0][m,0])
#             for m in range(k):
#                 third_sum+=q_k[m:k,0].sum()*psi(self.params[0][m,0]+self.params[2][m,0])
#             
#         mult_bound=np.cumsum(q_k*psi(self.params[2]),0)+second_sum-third_sum-np.cumsum(q_k*np.log(q_k),0)
#         
#         
#         #tilde tau, that's tricky
#         first_sum=np.zeros((K,1))
#         second_sum=np.zeros((K,1))
#         for k in range(self.K):
#             for m in range(k+1,self.K):
#                 first_sum[k,0]+=(self.D-q_z[:,m].sum())*q_k[k+1:m,0].sum()
#             second_sum[k,0]=(q_z[:,k:]).sum()
#             
#         self.params[0]*=(1-self.rho)
#         self.params[0]=self.rho*(self.params[5]/self.params[6]+first_sum+second_sum)
#         
#         #omega
#         self.params[1]*=(1-self.rho)
#         for k in range(self.K):
#                 self.params[1][:,k]=self.rho*(np.sum(psi(self.params[0][:k+1,0])-psi(self.params[0][:k+1,0]+self.params[2][:k+1,0]),0)+mult_bound[k]-0.5*np.log(2*np.pi)+0.5*(psi(self.params[3][k,0])-np.log(self.params[4][k,0]))\
#                     -0.5*(self.params[3][k,0]/self.params[4][k,0])*(self.batch_params[4][:,k]**2+self.batch_params[5][:,k]))
#         
#         #hat_tau
#         self.params[2]*=(1-self.rho)
#         for k in range(self.K):
#             self.params[2][k]=self.rho*(1.0+(D-q_z[:,k:]).sum()*q_k[k])
#         
#             
#         #tilde c
#         self.params[3]*=(1-self.rho)
#         self.params[3]=self.rho*(self.c+0.5*q_z.sum(0).reshape(-1,1))
#         
#         #tilde_f
#         self.params[4]*=(1-self.rho)
#         self.params[4]=self.rho*(self.f+(self.batch_params[4]**2+self.batch_params[5]).sum(0).reshape(-1,1))
#         
#         #update t_g_1
#         self.params[5]*=(1-self.rho)
#         self.params[5]=self.rho*(self.gamma_1+self.K-1)
#         
#         #update t_g_2
#         self.params[6]*=(1-self.rho)
#         self.params[6]=self.rho*(self.gamma_2-np.sum(psi(self.params[0][:self.K-2])-psi(self.params[0][:self.K-2]+self.params[2][self.K-2])))
#         
        
# ssum,_=theano.scan(fn=lambda k,curr,xn,ts,tm,tmu,tl,n: curr-2*(tm[n,k]*T.dot(tmu[:,k].T,x[n,:].T)+2*tm[n,k]*T.dot(tmu[:,k].T,T.sum(tm[n,k+1:]*tmu[:,k+1:],1))),
#                                sequences=T.arange(K),   
#                                outputs_info=T.constant(0.).astype('float32'),
#                                non_sequences=[xn,ts,tm,tmu,tl,n],
#                                )
#              
#              ssum=ssum[-1]

# batch_gradients=[0]*len(self.batch_params)
#         batch_gradients[0]=self.b+0.5*self.N*np.dot((miniBatch[s,:].T-np.dot(self.batch_params[-2],self.local_params[1][s,:].T)).T,miniBatch[s,:]-np.dot(self.batch_params[-2],self.local_params[1][s,:].T))
#         batch_gradients[1]=self.eta_1+0.5*self.N*np.exp(self.local_params[2][s,:,:])
#         batch_gradients[2]=self.eta_2+0.5*self.N*np.exp(self.local_params[2][s,:,:])*(np.exp(self.local_params[0][s,:])+self.local_params[1][s,:]**2).reshape(-1,1)
#         batch_gradients[3]=self.xi+self.N*np.exp(self.local_params[2][s,:,:])
#         
#         batch_gradients[4]=np.zeros((self.D,self.K))
#         for d in range(self.D):
#             batch_gradients[4][d,:]=self.N*(np.exp(self.batch_params[-1])/np.exp(self.batch_params[0]))*(np.exp(self.local_params[0][s,:])+self.local_params[1][s,:]**2).reshape(-1,)\
#                                      +(self.params[3]/self.params[4]).reshape(-1,)
#         batch_gradients[5]=np.zeros((self.D,self.K))
#         for d in range(self.D):
#             batch_gradients[5][d,:]=self.N*(batch_gradients[4][d,:]**(-1))*self.local_params[1][s,:]*(miniBatch[s,d].T-np.dot(self.batch_params[-2][d,:],self.local_params[1][s,:].T))
#         batch_gradients[6]=self.a+0.5*self.N*self.D
#         print(batch_gradients)
        # A nested scan for calculating the expectations of the last term of log p(x|...)
#         def normalCalc(d,n,xn,ts,tm,tmu,tl,ta,tb,K):
#             ssum,_=theano.scan(fn=lambda k,curr,ts,tm,tmu,tl,n: curr+T.cumsum(tm[n,k]*tm[n,k+1:]*tmu[:,k].T*tmu[:,k+1])+(2*tm[n,k]**2+ts[n,k]**2)*(T.sum(tl[:,k])+tmu[:,k].T*tmu[:,k])
#                                sequences=T.arange(K),
#                                outputs_info=T.constant(0.)
#                                non_sequences=[ts,tmu,tl,n]
#                                )
#             return -0.5*(ta/tb)*(x[n,d]**2-2*T.sum(tm[n,:]*tmu[d,:]*x[n,d])+T.sum((t_m[n,:]**2+2*ts[n,:]**2)*(tl[d,:]+tmu[d,:]**2)))
#             return -0.5*(t_a/t_b)*(xn[n,d]**2-2*xn[n,d]*T.sum(tmu[d,:]*tm[n,:])\
#                     +(tmu[d,:]*tm[n,:]).sum()**2\
#                     +((tmu[d,:]**2+tl[d,:])*(tm[n,:]**2+ts[n,:])-(tmu[d,:]**2*tm[n,:]**2)).sum())
            #return -0.5*(ta/tb)*(T.dot(x[n,:],x[n,:])-2*(T.dot(T.dot(tm[n,:],tmu.T),x[n,:].T))+T.dot(tm[n,:]**2+ts[n,:],T.sum(t_l,0)+T.diag(T.dot(tmu.T,tmu))))
            #return -0.5*(ta/tb)*(T.dot(x[n,:]-T.dot(tmu,tm[n,:].T),(x[n,:]-T.dot(tmu,tm[n,:].T)).T))
#             outputs,_=theano.scan(fn=normalCalc,
#                                 sequences=T.arange(D),
#                                 non_sequences=[n,xn,ts,tm,tmu,tl,ta,tb,K],
#                                 strict=True
#                                 )
#             return outputs
#         
 #=======================================================================
        # Calculate Gdk expectation
        # Again these loops are just to be sure
        #=======================================================================
#         def gcalc(k,d,tc,tf,tmu,tl):
#             return +0.5*(T.psi(tc[k,0])-T.log(tf[k,0]))-0.5*(tc[k,0]/tf[k,0])*(tmu[d,k]**2+tl[d,k])
#         
#         def inter_g_loop(n,tc,tf,tmu,tl):
#             
#             outputs,_=theano.scan(fn=gcalc,
#                                   sequences=T.arange(K),
#                                   non_sequences=[n,tc,tf,tmu,tl],
#                                   strict=True
#                                   )
#             return outputs
#         
#         finalg,_=theano.scan(fn=inter_g_loop,
#                                  sequences=T.arange(D),
#                                  non_sequences=[t_c,t_f,t_mu,t_l],
#                                  strict=True
#                                  )
#         
#         
#print(z.lowerBoundFunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))
    #print(z.gradyfunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))
    #print(z.gradx(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))
    #print(z.gradent(K=z.K,t_s=z.local_params[0]))
    #print('sup',z.gradafunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))

#=======================================================================
        # create the derivative functions    
        #=======================================================================
        #BUT THE INPUT MUST BE THE ORIGINAL VARIABLES
#         gradVariables=[t_tau1,omega,h_tau1,t_a1,t_c1,t_f1,t_g_11,t_g_21]
#         localVar=[t_s1,t_m,zeta1]
#         batch_grad_vars=[t_e_11,t_e_21,t_xi1,t_l1,t_mu,t_b1]
        
        
        #testing some stuff, here the gradent is the derivative of the entropy term containing t_s wrt to t_s. With the current values, this gives 0.5 because it replaces
        #ts[n,k] with the initial value
#         der=T.grad(expectation_log_y.sum(),t_s)
#         derx=T.grad(expectation_log_x.sum(),t_s)
#         derent=T.grad(0.5*T.sum(K*(T.log(2*np.pi)+1)+T.sum(T.log(t_s),1)),t_s)
#         self.gradyfunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],der,allow_input_downcast=True,on_unused_input='warn')
#         self.gradx=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],derx,allow_input_downcast=True,on_unused_input='warn')        
#         self.gradent=theano.function([t_s,K],derent,allow_input_downcast=True,on_unused_input='warn')
#         
       
        #DEBUG
#         dera=T.grad(likelihood,t_a)
#         self.gradafunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],dera,allow_input_downcast=True,on_unused_input='warn')
#         
#         self.multFunction=theano.function([t_tau,h_tau,K],mult_bound,allow_input_downcast=True)
#         self.qfunv=theano.function([t_tau,h_tau],q_k,allow_input_downcast=True)
        
        
        #END DEBUG
       
        #create the Theano variables
        #tilde_a,tilde_b,tilde_gamma_1 and tilde_gamma_2 are scalars
#         t_a=T.scalar('t_a',dtype='float64')
#         t_b=T.scalar('t_b',dtype='float64')
#         t_g_1=T.scalar('t_g_1',dtype='float64')
#         t_g_2=T.scalar('t_g_2',dtype='float64')
#         
#         #tilde_eta_1,tilde_eta_2,tilde_xi,tilde_l,tilde_mu,omega,tilde_s, tilde_m and zeta are matrices
#         t_e_1,t_e_2,t_xi,t_l,t_mu,omega,t_s,t_m=T.fmatrices('t_e_1','t_e_2','t_xi','t_l','t_mu','omega','t_s','t_m')
#         
#         #the only tensor we got
#         zeta=T.ftensor3('zeta')
#     
#         #the rest are columns
#         t_c=T.col('t_c',dtype='float64')
#         t_f=T.col('t_f',dtype='float64')
#         t_tau=T.col('t_tau',dtype='float64')
#         h_tau=T.col('h_tau',dtype='float64')
#          
         
                
#         take the log of the original parameters
#          lt_g_1=T.log(t_g_1)
#          lt_g_2=T.log(t_g_2)
#          lt_e_1=T.log(t_e_1)
#          lt_e_2=T.log(t_e_2)
#          lt_a=T.log(t_a)
#          lt_b=T.log(t_b)
#          lt_c=T.log(t_c)
#          lt_f=T.log(t_f)
#          lt_l=T.log(t_l)     
#          lzeta=T.log(zeta)
#          lt_xi=T.log(t_xi)
#          lh_tau=T.log(h_tau)
#          lt_tau=T.log(t_tau)
#          lt_s=T.log(t_s)
     #calculate q_k
#         q_k,up=theano.scan(lambda i, t_tau,h_tau: T.exp(T.psi(eh_tau[i])+T.sum(T.psi(et_tau[:i-1]))-T.sum(T.psi(et_tau[:i]+eh_tau[:i]))),
#                             sequences=T.arange(K),
#                             non_sequences=[t_tau,h_tau])
#         
#         q_k/=T.sum(q_k)

    #DIFFERENT FOR THE BATCH
#     def calculate_intermediate_batch_params(self,batch_gradients):
#         batch=[0]*len(self.batch_params)
#         for i in range(len(batch)):
#             batch[i]=np.array(self.batch_params[i],copy=True)
#         
#         for iteration in range(20):
#             for i in range(len(self.batch_params)):
#                 if (i==len(self.batch_params)-2):
#                     batch[i]+=0.01*batch_gradients[i]
#                     continue
#                 batch[i]+=np.exp(np.log(batch[i])+0.01*batch_gradients[i])
#         print(batch)
#         time.sleep(20)
#         return batch
            
   #=======================================================================
        # Nested scan for the calculation of the multinomial bound
        #=======================================================================
#         def oneStep(s_index,curr,tau,q_k,k,add_index):
#             return ifelse(T.gt(k,0),curr+T.psi(tau[s_index,0])*T.sum(q_k[s_index+add_index:k+1]),curr)
#          
#         def inner_loop(k,tau,q_k,add_index):
#            
#             zero=T.constant(0).astype('int64')
#  
#             s_indices=ifelse(T.gt(k,zero),T.arange(k),T.arange(0,1))
#  
#             n_steps=ifelse(T.gt(k-add_index,zero),k-add_index,T.constant(1).astype('int64'))
#  
#             outputs,_=theano.scan(fn=oneStep,
#                                         sequences=s_indices,
#                                         outputs_info=T.constant(.0),
#                                         non_sequences=[tau,q_k,k,add_index],
#                                         )
#             #print("outputs",outputs.type)
#             outputs=outputs[-1]
#             return outputs
#          
#         add_index=T.constant(1).astype(('int64'))
#         sec_sum,updates=theano.scan(fn=inner_loop,
#                                       sequences=T.arange(K),
#                                       non_sequences=[t_tau,q_k,add_index],
#                                       )
#          
#         #all I need to change for the second sum are the indices
#         add_index_zero=T.constant(0).astype('int64')
#         third_sum,up=theano.scan(fn=inner_loop,
#                                       sequences=T.arange(K),
#                                       non_sequences=[t_tau,q_k,add_index_zero],
#                                       )
 # log_x_calc,_=theano.scan(lambda i, ts,tm,tmu,tl,x,ta,tb,D,N: -0.5*D*T.log(2*np.pi)+0.5*T.log(T.psi(ta)-T.log(tb))-0.5*(ta/tb)*(T.dot(x[i,:],x[i,:].T)-2*T.dot(T.dot(tmu,tm[i,:].T).T,x[i,:].T)\
#                                                                                                                                         +T.sum(T.nlinalg.trace(t_s[i,:].dimshuffle('x',0))\
#                                                                                                                                                +T.dot(tm[i,:],tm[i,:].T)*(T.nlinalg.trace(t_l)+T.diag(T.dot(tmu.T,tmu))))),
#                            sequences=T.arange(S),
#                            non_sequences=[t_s,t_m,t_mu,t_l,x,t_a,t_b,D,N]
#                            )
        
#         log_g_calc,_=theano.scan(lambda i, tc,tf,tl,tmu,K: -0.5*T.log(T.prod(2*np.pi*(-T.psi(tc)+T.log(tf))))-0.5*T.sum(T.nlinalg.trace(tc/tf*t_l[i,:])\
#                                                                                                                            +((tc/tf))*T.dot(tmu[i,:].T,tmu[i,:])),
#                                  sequences=T.arange(D),
#                                  non_sequences=[t_c,t_f,t_l,t_mu,K]
#                                  )
#         def normalCalc(x_n,t_m_n,t_s_n,curr,t_mu,t_l,t_b,t_a):            
#             return curr-0.5*(t_b/(t_a-1))*(T.dot(x_n, x_n.T)-2*T.dot(T.dot(t_m_n.T,t_mu.T),x_n)+T.sum(T.dot((t_m_n**2+t_s_n).T,(t_l+T.diag(T.dot(t_mu.T,t_mu))).T)))
        
#         last_term,_=theano.scan(lambda i, curr,x,t_m,t_s,t_mu,t_l,t_b,t_a:  curr-0.5*(t_a/t_b)*(T.dot(x[i], x[i].T)-2*T.dot(T.dot(t_m[i].T,t_mu.T),x[i])\
#                                                                                                 +T.sum((t_m[i]**2+t_s[i])*(t_l+T.diag(T.dot(t_mu.T,t_mu))))),
#                                 sequences=T.arange(x.shape[0]),
#                                 outputs_info=T.constant(.0).astype('float64'),
#                                 non_sequences=[x,t_m,t_s,t_mu,t_l,t_b,t_a])
#         last_term=last_term[-1]
