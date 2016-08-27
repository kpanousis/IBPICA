'''
Created on Jun 29, 2016

@author: kon
'''
from __future__ import division
from scipy import signal
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
from sklearn import preprocessing

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
    def init_params(self,data):
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
        self.xi=5*np.ones((self.K,self.J))
        
        #=======================================================================
        # The scalar parameters
        #=======================================================================
        #self.t_a=1.0
        #self.t_b=1.0
        self.t_g_1=1.0
        self.t_g_2=1.0
        
        #=======================================================================
        # matrices
        #=======================================================================
        self.t_e_1=np.random.random((self.K,self.J))
        self.t_e_2=np.random.random((self.K,self.J))
        self.t_xi=np.random.random((self.K,self.J))
        #self.t_l=np.random.random((self.D,self.K))
        #self.t_mu=np.random.normal(0,1,(self.D,self.K)).astype('float64')
        self.omega=-np.random.random((self.D,self.K)).astype('float64')
        #self.t_s=np.random.random((self.N,self.K))
        #self.t_m=np.random.normal(0,1,(self.N,self.K)).astype('float64')
        
        #=======================================================================
        # tensor
        #=======================================================================
        self.zeta=1.0*np.random.random((self.N,self.K,self.J)).astype('float64')
        for n in range(self.N):
            self.zeta[n,:,:]/=self.zeta[n,:,:].sum(1).reshape(-1,1).astype('float64')
        
        #=======================================================================
        # vectors
        #=======================================================================
        self.t_c=1*np.random.random((self.K,1))
        self.t_f=1.0*np.random.random((self.K,1))
        self.t_tau=1*np.random.random((self.K,1))
        self.h_tau=1.0*np.random.random((self.K,1))
        
#         print((t_a/t_b)*(t_mu**2+t_l).sum(0)+(zeta*(t_e_1/t_e_2)).sum(2))
        
        
        
        y=self.initiliaze_posterior(data)
        #self.initialize_mixtures(y)
        
        #=======================================================================
        # The order is very important 
        #=======================================================================
        self.params=[self.t_tau,self.omega,self.h_tau,self.t_c,self.t_f,self.t_g_1,self.t_g_2]
        self.local_params=[self.t_s,self.t_m,self.zeta]
        self.batch_params=[self.t_b,self.t_e_1,self.t_e_2,self.t_xi,self.t_l,self.t_mu,self.t_a]
        #print(self.batch_params)
        #print(self.local_params)
        self.gamma_1=1
        self.gamma_2=1
        self.b=1
        self.a=1
        self.c=1
        self.f=1
        self.eta_1=1
        self.eta_2=1
    
    #===========================================================================
    # kmeans implementation for initialisation
    #===========================================================================
    def kmeans(self,y):
        eta=np.ones((self.N,))
        x=np.sort(y)
        i=np.argsort(y)
        lst=[1]
        for i in range(self.J-1):
            lst.append(2)
        seeds=np.array(lst)*self.N/(2*self.J)
        seeds=np.ceil(np.cumsum(seeds)).astype(np.int32)
        
        last_i=np.ones((self.N,))
        m=x[seeds]
        
        
        d=np.zeros((self.J,self.N))
        
        for iter in range(100):
            for j in range(self.J):
                d[j,:]=(y-m[j])**2

            i=np.argmin(d,0)
            if (i-last_i).sum()==0:
                break;
            else:
                for j in range(self.J):
                    if (eta[np.where(i==j)].sum()==0):
                        m[j]=(eta[np.where(i==j)]*y[np.where(i==j)]).sum()
                    else:
                        m[j]=(eta[np.where(i==j)]*y[np.where(i==j)]).sum()/eta[np.where(i==j)].sum()
                last_i=i
        
        #compute variances and mixing proportions
        v=np.zeros((self.J,))
        gammas=np.zeros((self.J,self.N))
        mix_prob=np.zeros((self.J,))
        for j in range(self.J):
            v[j]=(eta[np.where(i==j)]*(y[np.where(i==j)]-m[j])**2).sum()/((eta[np.where(i==j)]).sum()+np.finfo(float).eps)
            if v[j]==0:
                v[j]=self.N
            
            mix_prob[j]=y[np.where(i==j)].size/self.N
            gammas[j,:]=(1.0/(2*np.pi*v[j]))*np.exp(-((y-m[j])**2)/(2*v[j]))*mix_prob[j]
        sumg=np.repeat(gammas.sum(0).reshape(-1,1),self.J,1).T

        gammas/=sumg
        return gammas, v, m, mix_prob/mix_prob.sum()
        
    def initialize_mixtures(self,y):
        
        print('Initializing mixture parameters using kmeans...')
        self.t_e_1=np.zeros((self.K,self.J))
        self.t_e_2=np.zeros((self.K,self.J))  
        self.t_xi=np.zeros((self.K,self.J))
        self.zeta=np.zeros((self.N,self.K,self.J))
        for k in range(self.K):
            m_0=y[:,k].mean()
            v_0=(0.3*(y.max()-y.min()))**2
            tau_0=1/v_0
            
            gammas, kv, m, pi=self.kmeans(y[:,k])
            mean_precision=(1./kv).mean()
            var_precision=np.std(1/kv)**2
            b_0=(var_precision/mean_precision).mean()
            c_0=(mean_precision**2/var_precision).mean()
            
            la=pi
            mm=m

            v=kv.mean()*np.arange(1,self.J+1)/(self.N/self.J)
            
            b=np.zeros(self.J)
            c=np.zeros(self.J)
            for s in range(self.J):
                if kv[s]>0:
                    precision=(1/kv[s])
                else:
                    precision=mean_precision
                b[s]=var_precision/precision
                c[s]=precision**2/var_precision
            
            
            #print(renspo.shape)
            #print(renspo)
            self.t_xi[k,:]=la
            self.t_e_1[k,:]=b
            self.t_e_2[k,:]=c
            self.zeta[:,k,:]=gammas.T
       
        
    def initiliaze_posterior(self,data):
        
        
        #overcomplete so random
        if (self.D<self.K):
            G=np.random.normal(0,1,(self.D,self.K))
            y=np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),(data-data.mean(0).reshape(-1,1)).T).T
            m_G_hat=G
            mu_y_hat=y
            var_noise=np.var(data)/5.0
            l_hat=(1.0/var_noise).mean()*np.ones((self.D,1))
            
        else:
            
            #whitening and centering
            init_data=preprocessing.scale(data)
        
            #svd centered matrix
            [U,l,V]=np.linalg.svd(init_data)

                
            m_G_hat=V[:,:self.K]
            G=m_G_hat
            
            
            if (G.shape[0]==G.shape[1]):
                print('Square matrix..')
                var_noise=np.var(init_data)/5.0
                isovalue=(1./var_noise).mean()
            else:
                values=np.diag(l)
                explained=values[:self.K,:]
                unexplained=values[self.K:,:]
                mean_crp=unexplained.mean()**2/self.D
                isovalue=1.0/mean_crp
            
            noise=np.ones((self.D,1))*isovalue
            noise_prec_vect=noise
            l_hat=noise_prec_vect
        
            y=np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),init_data.T).T
            mu_y_hat=y
        
        self.t_s=np.random.gamma(1,1,(self.N,self.K))
        self.t_m=mu_y_hat
        self.t_a=0.5*self.N*self.D
        self.t_b=(l_hat*self.t_a).sum()
        self.t_a=self.N*self.D/2
        self.t_b=(l_hat*self.t_a).sum()
        self.t_mu=m_G_hat
        self.t_l=1*np.random.gamma(1,1,(self.D,self.K))
        return y
    
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
        
        #sample for each dimension
        K_d_star=np.random.poisson(self.t_g_1/(self.t_g_2*(self.D-1)),size=self.D)
        
        
        print('Number of features before expansion:',self.K)
            
        
        
        #===========================================================================
        # For each dimension update features
        #===========================================================================
        for d in range(self.D):
            
            if (K_d_star[d]==0):
                continue
            
            #onstruct the ibp
            z=np.zeros((self.D,K_d_star[d]))
            alpha=np.random.gamma(self.gamma_1,self.gamma_2,size=K_d_star[d])
            u=np.random.beta(alpha,1)
            pi_k=np.cumprod(u)
            
            for k in range(K_d_star[d]):
                z[:,k]=bernoulli.rvs(pi_k[k],size=self.D)
                
            lambda_k=np.random.gamma(self.c,self.f,size=K_d_star[d])

            #Draw random samples  G_{d,:}^* from the prior
            G=np.zeros((1,K_d_star[d]))
            for k in range(K_d_star[d]):
                if z[d,k]:
                    if (K_d_star[d]>1):
                        G[0,k]=z[d,k]*np.random.randn()*lambda_k[k]**-0.5
                    else:
                        G[0,k]=z[d,k]*np.random.randn(K_d_star[d])*lambda_k[k]**-0.5
            #G=np.random.randn(1,K_d_star[d])*lambda_k**-.5
            #E_g[G_{:,d}^T G_{:,d}]
            #may need a modification here
            expectation_g_transpose_g=np.dot((self.t_l[d,:]+self.t_mu[d,:]**2).reshape(-1,1),(self.t_l[d,:]+self.t_mu[d,:]**2).reshape(1,-1))
            
            #calculate M_d and M_d_star
            M_d=expectation_phi*expectation_g_transpose_g+np.eye(self.K,self.K)
            M_d_star=expectation_phi*np.dot(G.T,G)+np.eye(K_d_star[d],K_d_star[d])
                
            exp_sum=0
            exp_sum_star=0
            
            for n in range(self.S):   

                m_nd=expectation_phi*np.dot(inv(M_d),self.t_mu[d,:].T)*(miniBatch[n,d]-np.dot(self.t_mu[d,:].T,self.t_m[n,:].T))
                exp_sum+=np.dot(np.dot(m_nd.T,M_d),m_nd)

                m_nd_star=expectation_phi*np.dot(inv(M_d_star),G.T)*(miniBatch[n,d]-np.dot(self.t_mu[d,:],self.t_m[n,:].T)) 
                exp_sum_star+=np.dot(np.dot(m_nd_star.T,M_d_star),m_nd_star)
    
            
            theta_d=det(M_d)**(-self.S/2)*np.exp(0.5*exp_sum)
            theta_d_star=det(M_d_star)**(-self.S/2)*np.exp(0.5*exp_sum_star)
            
            #acceptance probability
            p_d_star=min(1,theta_d_star/theta_d)
            
           
            #accept proposal?
            if (np.random.rand()<p_d_star):
                #Remove redundant features
                self.unused_components()
                 
                self.expand_arrays(K_d_star[d])
                self.K+=K_d_star[d]
        
        print('Number of features after expansion',self.K)
    
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
        self.t_f=(self.f+0.5*(self.t_l+self.t_mu**2).sum(0).reshape(-1,1))
         
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
    # Update the local parameters using simple VI
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
                                                   +0.5*(psi(self.t_e_1)-np.log(self.t_e_2)-(self.t_e_1/self.t_e_2)*(self.t_m[mb_indices[n],:]**2+self.t_s[mb_indices[n],:]).reshape(-1,1))))
            self.zeta[mb_indices[n],:,:]/=self.zeta[mb_indices[n],:,:].sum(1).reshape(-1,1).astype('float64')
                       
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
        self.omega=np.hstack((self.omega,-np.random.gamma(1,1,(self.D,diff))))
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
            
        with open(start_time+'/params'+str(iteration)+'.pickle', 'wb') as f:  
                pickle.dump(self.params, f)
        with open(start_time+'/localparams'+str(iteration)+'.pickle', 'wb') as f: 
                pickle.dump(self.local_params, f)
        with open(start_time+'/batchparams'+str(iteration)+'.pickle', 'wb') as f: 
                pickle.dump(self.batch_params, f)
        with open(start_time+'/bound_iter_'+str(iteration)+'.pickle', 'wb') as f:  
                pickle.dump(LL, f)
                
#     def unmix(self,x):
#         G,G_sq=self.t_mu,self.t_mu**2+1/self.t_l
#         noise=self.t_a/self.t_b
#         mn=0
#         old_x=np.dot(np.dot(np.linalg.inv(np.dot(G.T,G)),G.T),x.T)
#         noiseG=np.repeat(noise, 1,self.K)*G
#         data_bit=np.dot(noiseG.T,x.T)
#         
#         preweight=noise*G_sq
#         weight_G=np.repeat(preweight,1,self.N)
#         
#         
#         for i in range(self.K):
            
    #===========================================================================
    # Some naive creation of synthetic data
    #===========================================================================
    def create_synthetic_data(self,dataset,components=5):
        x=sio.loadmat(dataset)
        x=x["X"]
        x=x.reshape(180,3,3000)[:,0,:]
        #ind=np.arange(500,1000)
        #ind2=np.arange(3000,4500)
        #al=np.append(ind,ind2)
        #ind=np.arange(500,1000,1)
        #ind2=np.arange(3000,4500,1)
        #ind_all=list(ind)
        #ind_all.extend(list(ind2))
        #x=x[:,ind_all]
        return x.T,0,0
#         
    
#===============================================================================
# Main 
#===============================================================================
    
if __name__ == '__main__':
    
    #some initial variables
    initN=1000
    initD=4
    initJ=5
    initS=60
    dataset='data/helwig_snr_1_overlap_1'
    data='helwig_snr_1_overlap_1_find_features'
    #x=datasets.load_iris()
    #x=x.data
    #initN,initD=x.shape
    #initialize IBP_ICA object
    z=IBP_ICA(5,initD,initJ,initS,initN)
    
    #create some synthetic data
    x,_,_=z.create_synthetic_data(dataset)
    #x=x[:1000,:]
    z.N=x.shape[0]
    z.D=x.shape[1]
    
    
    #init the posterior parameters
    z.init_params(x)
    
    print('N:',z.N,'D:',z.D)
   
    iteration=0
    max_iter=1000
    elbo_tolerance=10**-3
    #keep the lower bound for each iteration
    LL=np.empty((max_iter,int(z.N/z.S)))
    LL_iterations=np.empty((max_iter,1))
    
    start_time=str(iteration)+(time.strftime("%Y-%m-%d-%H:%M").replace(" ","_")+'_batch_size_'+str(initS)+'_D_'+str(z.D)+'_data_'+data+'_one_subject')
    global_min=0
    #repeat until maxi iterations
    st=time.time()
    while iteration<max_iter:
        if (iteration % 5 ==0):
            print('ELBO at iteration',iteration,":",LL[iteration-1,-1])
        
        iteration+=1
        print("Stochastic IBP-ICA iteration: ",iteration)
        #set step size for this iteration (Paisley et al.) 
        z.rho=(iteration+1000.0)**-.75
       
        
        #create all the random minibatches for this iteration
        random_indices=np.arange(z.N)
        np.random.shuffle(random_indices)
        random_indices=random_indices.reshape((int(z.N/z.S),z.S))
        
        #FOR ALL THE MINIBATCHES update the local parameters and the global parameters with SGD
        current_minibatch=0
        
        
        
        for miniBatch_indices in random_indices:
            current_minibatch+=1
            miniBatch=x[miniBatch_indices,:]
                        
            #perform one iteration of the algorithm 
            z.iterate(x,miniBatch_indices)
            
            #append lower bound to list
            LL[iteration-1,current_minibatch-1]=(z.calculate_lower_bound(miniBatch,miniBatch_indices))
            
                    
            if (np.isnan(LL[iteration-1,current_minibatch-1])):
                sys.exit('Why is the bound nan? Please Debug.')
            
        #update the non batch global params
        z.global_params_VI()
        LL_iterations[iteration-1,0]=(z.calculate_lower_bound(miniBatch,miniBatch_indices))

        #check convergence
        if iteration>1:
            if abs(LL_iterations[iteration-1,0]-LL_iterations[iteration-2,0])<elbo_tolerance:
                print("Reached ELBO tolerance level..")
                break
        if (iteration %5==0):
            z.feature_update(miniBatch)             
            
        #save params for each iteration
        z.save_params(iteration,start_time,LL[iteration-1,:])
        
    #Print some stuff and save all
    global_min=np.max(LL)
    global_min_ind=np.argmax(LL)
    print('------------------------------------------------------------------')
    print('The global max is ',global_min,'found in iteration',global_min_ind)
    print('------------------------------------------------------------------')
    print('Total running time:',time.time()-st)
    z.save_params("final",start_time,LL)

    print(LL)
    
    
    
    
    
    
    
  