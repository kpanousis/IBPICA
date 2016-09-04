'''
Created on Jun 5, 2016

@author: kon
'''
from theano import tensor as T
import theano 
from theano.ifelse import ifelse

from theano.tensor.shared_randomstreams import RandomStreams
from numpy import random as r
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import bernoulli
from scipy.special import psi
from scipy.special import gamma
from scipy.special import beta
from numpy.linalg import norm
from sklearn.decomposition import PCA   
import collections
from recordclass import recordclass
import time

temp=0
        for k in range(self.K):
            temp+=-2*self.t_m[s,k]*np.dot(self.t_mu[:,k].T,x[s,:].T)+2*(self.t_m[s,k]*self.t_m[s,k+1:]*np.dot(self.t_mu[:,k].T,self.t_mu[:,k+1:])).sum()\
            +(self.t_s[s,k]+self.t_m[s,k]**2)*(np.trace(np.diag(self.t_l[:,k]))+np.dot(self.t_mu[:,k].T,self.t_mu[:,k]))

class IBP_ICA:
    
    def __init__(self,K,D,J,S ):
        self.K=K
        self.D=D
        self.J=J
        self.S=S
       
    #===============================================================================
    # Initialize the prior parameters
    #===============================================================================
    def initialize_prior_parameters(self,K,J,show_values=False):
        gamma_1=2.0
        gamma_2=2.0
        eta_1=2.0
        eta_2=2.0
        c=2.0
        f=2.0
        a=2.0
        b=2.0
        xi=np.ones((K,J))*(1.0/J)
        p=recordclass("prior",["gamma_1","gamma_2","eta_1","eta_2","c","f","a","b","xi"])
        prior=p(gamma_1,gamma_2,eta_1,eta_2,c,f,a,b,xi)
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
        
        return prior
    
    #===============================================================================
    # Initialize the parameters of the posterior distributions
    #===============================================================================
    def initialize_posterior_parameters(self,K,D,N,J,show_values=False):
        
        p=recordclass("posterior",['tilde_a','tilde_b','tilde_xi','tilde_eta_1','tilde_eta_2','tilde_lambda','tilde_mu','tilde_gamma_1','tilde_gamma_2',\
                                   'omega','tilde_c','tilde_f','tilde_tau','hat_tau','zeta',"tilde_s","tilde_m"])
        posterior=p(2.0,2.0,2.0*np.ones((K,J)),2*np.ones((K,J)),2*np.ones((K,J)),r.gamma(1,1,size=(D,K)),r.normal(0,1,size=(D,K)),2.0,2.0,r.random((D,K)),2*np.ones((K,1)),2*np.ones((K,1)),2*np.ones((K,1)),\
                    2*np.ones((K,1)),np.ones((N,K,J)),r.gamma(1,1,size=(N,K)),r.normal(0,1,size=(N,K)))
        
        for s in range(self.S):
            posterior.zeta[s,:,:]/=posterior.zeta[s,:,:].sum(1).reshape(-1,1)
        if show_values:
            print("tilde_a init value: {}".format(params.tilde_a))
            print("tilde_b init value: {}".format(params.tilde_b))
            print("tilde_xi init value: {}".format(params.tilde_xi))
            print("tilde_eta_1 init value: {}".format(params.tilde_eta_1))
            print("tilde_eta_2 init value: {}".format(params.tilde_eta_2))
            print("tilde_lambda init value: {}".format(params.tilde_lambda))
            print("tilde_mu init value: {}".format(params.tilde_mu))
            print("tilde_gamma_1 init value: {}".format(params.tilde_gamma_1))
            print("tilde_gamma_2 init value: {}".format(params.tilde_gamma_2))
            print("omega init value: {}".format(params.omega))
            print("tilde_c init value: {}".format(params.tilde_c))
            print("tilde_f init value: {}".format(params.tilde_f))
            print("tilde_tau init value: {}".format(params.tilde_tau))
            print("hat_tau init value: {}".format(params.hat_tau))
        
        return posterior
    
            
    #===============================================================================
    # For each dimension D calculate the proposed number of new features
    #===============================================================================
    def feature_update(self,K_d,D,N,prior,posterior,x=1):   
        
        #some "global" calculations
        expectation_phi=posterior.tilde_a/posterior.tilde_b
        K=max(K_d)
        
         #=======================================================================
        # construct IBP
        #=======================================================================
        alpha=r.gamma(prior.gamma_1,prior.gamma_2)
        u=r.beta(alpha,1,size=K)
        pi_k=np.cumprod(u)
        z=np.zeros((D,K))
        for k in range(K):
            z[:,k]=bernoulli.rvs(pi_k[k],size=D)
        lambda_k=r.gamma(prior.c,prior.f,size=K)
            
        #Draw random samples  G_{d,:}^* from the prior
        G=np.zeros((D,K))
        for k in range(K):
            G[:,k]=z[:,k]*r.normal(0,lambda_k[k]**-1,size=D)
    
        #assign inf to non active features
        #find_active=np.where(z==0)
        #G[find_active]=np.inf
        
        #===========================================================================
        # For each dimension update features
        #===========================================================================
        for d in range(D):
            #eye matrix
            I=np.eye(K,K)
             
            #E_g[G_{:,d}^T G_{:,d}]
            #may need a modification here
            expectation_g_transpose_g=np.dot(posterior.tilde_mu[d,:],posterior.tilde_mu[d,:].T)
            
            #calculate M_d and M_d_star
            M_d=expectation_phi*expectation_g_transpose_g+I
            M_d_star=expectation_phi*np.dot(G[d,:],G[d,:].T)+I
                
            exp_sum=0
            exp_sum_star=0
            for n in range(N):
                   
                 m_nd=expectation_phi*np.dot(inv(M_d),posterior.tilde_mu[d,:].T)*(x[n,d]-np.dot(posterior.tilde_mu[d,:].T,posterior.tilde_m[n,:].T))
                 m_nd_star=expectation_phi*np.dot(inv(M_d_star),G[d,:].T)*(x[n,d]-np.dot(posterior.tilde_mu[d,:],posterior.tilde_m[n,:].T)) 
                 exp_sum+=np.dot(np.dot(m_nd.T,M_d),m_nd)
                 exp_sum_star+=np.dot(np.dot(m_nd_star.T,M_d_star),m_nd_star)
    
            theta_d=det(M_d)**(-N/2)*np.exp((1/2)*exp_sum)
            theta_d_star=det(M_d_star)**(-N/2)*np.exp((1/2)*exp_sum_star)
            
            #acceptance probability
            p_d_star=min([1,theta_d_star/theta_d])
            
            #accept proposal?
            K_d_star=0
            if (r.rand()<p_d_star):
                K_d_star=r.poisson(posterior.tilde_gamma_1/(posterior.tilde_gamma_2*(D-1)))
        
            #=======================================================================
            # DO something for K here
            #=======================================================================
            K_d[d]+=K_d_star
        return K_d
    
    
    
    #===============================================================================
    # Update the local parameters of the model until l2 convergence
    #===============================================================================
    def update_local_parameters(self,x,K,J,posterior):
        
        S=len(x)
        
        tolerance=10**-4
        
        expectation_phi=posterior.tilde_a/posterior.tilde_b
        
        old_tilde_s=np.array(posterior.tilde_s,copy=True)
        old_tilde_m=np.array(posterior.tilde_m,copy=True)
        old_zeta=np.array(posterior.zeta,copy=True)
        iteration=0
        while True:
            iteration+=1
            for s in range(S):
                for k in range(K):
                    posterior.tilde_s[s,k]=((posterior.zeta[s,k,:]*(posterior.tilde_eta_1[k,:]/posterior.tilde_eta_2[k,:])).sum()+expectation_phi*((posterior.tilde_mu[:,k]**2+posterior.tilde_mu[:,k]).sum()))**-1
                    posterior.tilde_m[s,k]=expectation_phi*posterior.tilde_s[s,k]*np.dot(posterior.tilde_mu[:,k].T,x[s,:].T)
                    
                    #===============================================================
                    # edw paizei ena major lathaki/ leipei to y
                    #===============================================================
                    for j in range(J):
                        posterior.zeta[s,k,j]=np.exp(psi(posterior.tilde_xi[k,j])-psi(posterior.tilde_xi[k,:].sum())+(1/2)*(psi(posterior.tilde_eta_1[k,j])-np.log(posterior.tilde_eta_2[k,j])) \
                        -(1/2)*(posterior.tilde_s[s,k]+posterior.tilde_m[s,k]**2)*(posterior.tilde_eta_1[k,j]/posterior.tilde_eta_2[k,j]))
                    
                    posterior.zeta[s,k,:]/=(posterior.zeta[s,k,:]).sum()
            
            #converged?
            if (abs(norm(old_tilde_s)-norm(posterior.tilde_s))<tolerance):
                 if (abs(norm(old_tilde_m)-norm(posterior.tilde_m))<tolerance):
                     if abs(norm(old_zeta)-norm(posterior.zeta))<tolerance:
                         break;

            old_tilde_s=np.array(posterior.tilde_s,copy=True)
            old_tilde_m=np.array(posterior.tilde_m,copy=True)
            old_zeta=np.array(posterior.zeta,copy=True)
        return posterior
    
    
    #===============================================================================
    # Update the intermediate global parameters to proceed to the gradient step
    #===============================================================================
    def intermediate_global_parameters(self,x,N,K,J,D,prior,posterior,print_values=False):
        S=len(x)
        
        p=collections.namedtuple("params_hat",['hat_a','hat_b','hat_xi','hat_eta_1','hat_eta_2','hat_lambda','hat_mu','hat_gamma_1','hat_gamma_2','hat_omega','hat_c','hat_f','hat_tau_1','hat_tau_2'])
        
        hat_gamma_1=prior.gamma_1+K-1
        hat_gamma_2=prior.gamma_2-(psi(posterior.tilde_tau)-psi(posterior.tilde_tau+posterior.hat_tau)).sum()
        hat_a=prior.a+(N*D)/2
        
        hat_omega=np.zeros((D,K))
        hat_b=np.zeros((S,))
        hat_c=np.zeros((K,1))
        hat_f=np.zeros((K,1))
        hat_tau_1=np.zeros((K,1))
        hat_tau_2=np.zeros((K,1))
        hat_eta_1=np.zeros((S,K,J))
        hat_eta_2=np.zeros((S,K,J))
        hat_xi=np.zeros((S,K,J))
        hat_lambda=np.zeros((S,D,K))
        hat_mu =np.zeros((S,D,K))
        
        q_k=self.calculate_q(K, posterior.tilde_tau, posterior.hat_tau)
        q_z=self.calculate_posterior_z(posterior.omega)
    
        for k in range(K):
            mult_bound=self.multinomial_bound(k, q_k, posterior.tilde_tau, posterior.hat_tau)
            
            
            hat_omega[:,k]=(psi(posterior.tilde_tau[:k+1])-psi(posterior.tilde_tau[:k+1]+posterior.hat_tau[:k+1])).sum()+mult_bound[k,0] \
                            +(1/2)*(psi(posterior.tilde_c[k])-np.log(posterior.tilde_f[k]))-(1/2)*(posterior.tilde_lambda[:,k]+posterior.tilde_mu[:,k]**2)*(posterior.tilde_c[k]/posterior.tilde_f[k])
            
            hat_c[k]=prior.c+(1/2)*q_z[:,k].sum()
            hat_f[k]=prior.f+(1/2)*(posterior.tilde_lambda[:,k]+posterior.tilde_mu[:,k]**2).sum()
            
            #hat_tau_1 and hat_tau_2 calculation
            cur_sum=0
            for m in range(k+1,K):
                cur_sum+=(D-(q_z[:,m]).sum())*((q_k[k+1:m+1]).sum())
            hat_tau_1[k]=cur_sum
            hat_tau_1[k]+=q_z[:,k:].sum()+posterior.tilde_gamma_1/posterior.tilde_gamma_2
            
            cur_sum=0
            for m in range(k,K):
                cur_sum+=D-q_z[:,m].sum()
            cur_sum*=q_k[k]
            hat_tau_2[k]=1.0+cur_sum
            
            for s in range(S):
                hat_eta_1[s,:,:]=prior.eta_1+(N/2)*posterior.zeta[s,:,:]
                hat_eta_2[s,:,:]=prior.eta_2+(N/2)*posterior.zeta[s,:,:]*(posterior.tilde_s[s,k]+posterior.tilde_m[s,k]**2)
                hat_xi[s,:,:]=prior.xi+N*posterior.zeta[s,:,:]
                hat_lambda[s,:,k]=N*(posterior.tilde_a/posterior.tilde_b)*(posterior.tilde_s[s,k]+posterior.tilde_m[s,k]**2)+(posterior.tilde_c[k]/posterior.tilde_f[k])
                for d in range(D):
                    hat_mu[s,d,k]=N*(hat_lambda[s,d,k]**-1)*(posterior.tilde_m[s,k])*(x[s,d]-np.dot(posterior.tilde_mu[d,:],posterior.tilde_m[s,:].T))*(posterior.tilde_a/posterior.tilde_b)
                hat_b[s]=prior.b+(N/2)*np.dot((x[s,:].T-np.dot(posterior.tilde_mu,posterior.tilde_m[s,:].T)),(x[s,:].T-np.dot(posterior.tilde_mu,posterior.tilde_m[s,:].T)).T)
    
        params_hat=p(hat_a,hat_b,hat_xi,hat_eta_1,hat_eta_2,hat_lambda,hat_mu,hat_gamma_1,hat_gamma_2,hat_omega,hat_c,hat_f,hat_tau_1,hat_tau_2)
    
        if (print_values):
            print(params_hat)
            
        return params_hat
                
            
            
    #===============================================================================
    # Calculate the posterior of z, q(z_{dk}=1)
    #===============================================================================
    def calculate_posterior_z(self,posterior_omega):
        return 1.0/(1.0+np.exp(-posterior_omega))
    
    
    #===============================================================================
    # Calculate the bound based on multinomial expansion
    #===============================================================================
    def multinomial_bound(self,k,q_k,tau_1,tau_2):
        if (k==0):
            sec_sum=0
            third_sum=0
        else:
            sec_sum=np.sum(psi(tau_1[:k-1])*(np.sum(q_k[:k])-np.cumsum(q_k[:k-1])))
            third_sum=np.sum(psi(tau_1[:k]+tau_2[:k])*(np.cumsum(q_k[:k])[::-1]))
            
        mult_bound=(q_k[k]*psi(tau_2))+sec_sum-third_sum-np.cumsum(q_k[k]*np.log(q_k[k]))
#         return mult_bound
#         mult_bound=(q_k[:k+1]*psi(tau_2[:k+1])).sum()
#         cur_sum=0
#         q_sum=0
#         for m in range(k-1):
#             for n in range(m+1,k):
#                 q_sum+=q_k[n]
#             cur_sum+=q_sum*psi(tau_1[m])
#             q_sum=0
#             
#         mult_bound+=cur_sum 
#         cur_sum=0
#         q_sum=0
#         for m in range(k):
#             for n in range(m,k):
#                 q_sum+=q_k[n]
#             cur_sum+=q_sum*psi(tau_1[m]+tau_2[m])
#             q_sum=0
#             
#         mult_bound-=cur_sum+(q_k[:k+1]*np.log(q_k[:k+1])).sum()
        return mult_bound
        
    #===============================================================================
    # Calculate q for the calculation of the multinonmial expansion bound
    #===============================================================================
    def calculate_q(self,K,tau_1,tau_2):
        q_k=np.zeros(K)
        for k in range(K):
            q_k[k]=np.exp(psi(tau_2[k])+(psi(tau_1[:k])).sum()-(psi(tau_1[:k+1]+tau_2[:k+1])).sum())
        q_k/=q_k.sum()
        return q_k
    
    #===============================================================================
    # Make the final gradient step for the global parameters
    #===============================================================================
    def gradient_step(self,rho,S,params_hat,posterior,print_values=False):
        
        posterior.tilde_a=(1.0-rho)*posterior.tilde_a+rho*params_hat.hat_a
        posterior.tilde_gamma_1=(1.0-rho)*posterior.tilde_gamma_1+rho*params_hat.hat_gamma_1
        posterior.tilde_gamma_2=(1.0-rho)*posterior.tilde_gamma_2+rho*params_hat.hat_gamma_2
        posterior.tilde_c=(1.0-rho)*posterior.tilde_c+rho*params_hat.hat_c
        posterior.tilde_f=(1.0-rho)*posterior.tilde_f+rho*params_hat.hat_f
        posterior.tilde_tau=(1.0-rho)*posterior.tilde_tau+rho*params_hat.hat_tau_1
        posterior.hat_tau=(1.0-rho)*posterior.hat_tau+rho*params_hat.hat_tau_2
        posterior.omega=(1.0-rho)*posterior.omega+rho*params_hat.hat_omega
    
        
        posterior.tilde_eta_1=(1.0-rho)*posterior.tilde_eta_1+(rho/S)*params_hat.hat_eta_1.sum(0)
        posterior.tilde_eta_2=(1.0-rho)*posterior.tilde_eta_2+(rho/S)*(params_hat.hat_eta_2).sum(0)
        posterior.tilde_xi=(1.0-rho)*posterior.tilde_xi+(rho/S)*(params_hat.hat_xi).sum(0)
        posterior.tilde_lambda=(1.0-rho)*posterior.tilde_lambda+(rho/S)*(params_hat.hat_lambda).sum(0)
        posterior.tilde_mu=(1.0-rho)*posterior.tilde_mu+(rho/S)*params_hat.hat_mu.sum(0)
        posterior.tilde_b=(1.0-rho)*posterior.tilde_b+(rho/S)*(params_hat.hat_b).sum(0)
        
        if (print_values):
            print(posterior)
            
        return posterior
    
    #===============================================================================
    # Init some of the local parameters based on the global dependence 
    #===============================================================================
#     def init_local(self,x,S,K,J,posterior,custom_init=False):
#         
#         if (not custom_init):
#             posterior.zeta=np.zeros((S,K,J))
#             tilde_s_inv=np.zeros((S,K))
#             posterior.tilde_m=np.zeros((S,K))
#             for s in range(S):
#                 posterior.zeta[s,:,:]=np.exp(psi(posterior.tilde_xi)-psi(posterior.tilde_xi.sum(1).reshape(-1,1).repeat(axis=1,repeats=J))\
#                                              -(1/2)*(psi(posterior.tilde_eta_1)-np.log(posterior.tilde_eta_2)\
#                 +posterior.tilde_eta_1/posterior.tilde_eta_2))
#                 tilde_s_inv[s,:]=np.sum(posterior.zeta[s,:,:]*(posterior.tilde_eta_1/posterior.tilde_eta_2),1)+(posterior.tilde_a/posterior.tilde_b)*(np.diag(np.dot(posterior.tilde_mu.T,posterior.tilde_mu)))
#                 for k in range(K):
#                     posterior.tilde_m[s,k]=(tilde_s_inv[s,k]**-1)*(posterior.tilde_a/posterior.tilde_b)*np.dot(posterior.tilde_mu[:,k].T,x[s,:])
#             posterior.tilde_s=tilde_s_inv**-1
#         else:
#             posterior.tilde_s_inv=r.gamma(1,1,size=(S,K))
#             pca=PCA(n_components=K)
#             pca.fit(x)
#             for s in range(S):
#                 posterior.zeta[s,:,:]=np.exp(psi(tilde_xi)-psi(sum(tilde_xi,2)))+psi(tilde_eta_1)-np.log(tilde_eta_2)-tilde_eta_1/tilde_eta_2
#             
#         return posterior
    
    
   
            
    
    def unused_components(self,K,N,D,J,xi,posterior):
        '''
        Check for unused components and reshape arrays if necessary
        '''
        used_k=self.search_used_components(posterior.omega,K)
        if (len(used_k)!=K):
            xi,posterior=self.contract_arrays(used_k, J, N, D, xi, posterior)
        return len(used_k),xi,posterior
        
    
    def search_used_components(self,omega,K):
        '''
        Find which components are being used
        '''
        
        threshold=10**-4
        q=1.0/(1.0+np.exp(-omega))
        used_k=[]
        for k in range(K):
            if (np.where(q[:,k]>threshold)[0]).size>0:
                used_k.append(k)
        return used_k
        
    def beta_calc(a):
        pr=1.0
        for i in range(len(a)):
            pr*=gamma(a[:,i])
        b_a=pr/gamma(a.sum(1))
    
        return b_a 
    
    
    def lowerBoundCalc(self):
        '''
        Create the lower bound and use T.grad to get the gradients for free
        '''
        
        print("Creating lower bound functions...")
        
        print("\t Initializing prior parameters...")
        a=T.scalar('a',dtype='float64')
        b=T.scalar('b',dtype='float64')
        c=T.scalar('c',dtype='float64')
        f=T.scalar('f',dtype='float64')
        g_1=T.scalar('g_1',dtype='float64')
        g_2=T.scalar('g_2',dtype='float64')
        e_1=T.scalar('e_1',dtype='float64')
        e_2=T.scalar('e_2',dtype='float64')

        K=T.scalar('K',dtype='int64')
        D=T.scalar('D',dtype='int64')      
        S=T.scalar('S',dtype='int64')
        xi=T.matrix('xi',dtype='float64')
        
        #data matrix
        x=T.matrix('x',dtype='float64')
        
        #create the Theano variables
        #tilde_a,tilde_b,tilde_gamma_1 and tilde_gamma_2 are scalars
        t_a=T.scalar('t_a',dtype='float64')
        t_b=T.scalar('t_b',dtype='float64')
        t_g_1=T.scalar('t_g_1',dtype='float64')
        t_g_2=T.scalar('t_g_2',dtype='float64')
        
        #tilde_eta_1,tilde_eta_2,tilde_xi,tilde_l,tilde_mu,omega,tilde_s, tilde_m and zeta are matrices
        t_e_1,t_e_2,t_xi,t_l,t_mu,omega,t_s,t_m=T.fmatrices('t_e_1','t_e_2','t_xi','t_l','t_mu','omega','t_s','t_m')
        
        #the only tensor we got
        zeta=T.ftensor3('zeta')
    
        #the rest are columns
        t_c=T.col('t_c',dtype='float64')
        t_f=T.col('t_f',dtype='float64')
        t_tau=T.col('t_tau',dtype='float64')
        h_tau=T.col('h_tau',dtype='float64')
        
        
               
        #take the log of the original parameters
        lt_g_1=T.log(t_g_1)
        lt_g_2=T.log(t_g_2)
        lt_e_1=T.log(t_e_1)
        lt_e_2=T.log(t_e_2)
        lt_a=T.log(t_a)
        lt_b=T.log(t_b)
        lt_c=T.log(t_c)
        lt_f=T.log(t_f)
        lt_l=T.log(t_l)     
        lzeta=T.log(zeta)
        lt_xi=T.log(t_xi)
        lh_tau=T.log(h_tau)
        lt_tau=T.log(t_tau)
        lt_s=T.log(t_s)
        
        #exp of the log
        et_g_1=T.exp(lt_g_1)
        et_g_2=T.exp(lt_g_2)
        et_e_1=T.exp(lt_e_1)
        et_e_2=T.exp(lt_e_2)
        et_a=T.exp(lt_a)
        et_b=T.exp(lt_b)
        et_c=T.exp(lt_c)
        et_f=T.exp(lt_f)
        et_l=T.exp(lt_l)     
        ezeta=T.exp(lzeta)
        et_xi=T.exp(lt_xi)
        eh_tau=T.exp(lh_tau)
        et_tau=T.exp(lt_tau)
        et_s=T.exp(lt_s)
        
        cs=T.cumsum(T.psi(et_tau),0)-T.psi(et_tau)
        q_k=T.exp(T.psi(eh_tau)+cs-T.cumsum(T.psi(et_tau+eh_tau),0))
        q_k/=T.sum(q_k)
       
        
    
        #check these FOR CORRECTNESS
        try_sum,_=theano.scan(lambda i, qk,ttau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i-1])*(T.sum(qk[:i])-T.cumsum(qk[:i-1]))),T.constant(0.0,dtype='float64')),
                            sequences=T.arange(K),
                            non_sequences=[q_k,et_tau],
                            strict=True
                            )
        
        try_sum2,_=theano.scan(lambda i, qk,ttau,htau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i]+htau[:i])*(T.cumsum(qk[:i])[::-1])),T.constant(0.0,dtype='float64')),
                            sequences=T.arange(K),
                            non_sequences=[q_k,et_tau,eh_tau],
                            strict=True
                            )
        
      
        #the multinomial bound for calculating p(z_{dk}|\upsilon_k)
        mult_bound=T.cumsum(q_k*T.psi(eh_tau))+try_sum-try_sum2-T.cumsum(q_k*T.log(q_k))

        #calculate q(z_{dk}=1)
        q_z=1.0/(1.0+T.exp(-omega))
        
       

        #=======================================================================
        # calculation of the weird last term
        #=======================================================================
        def normalCalc(d,n,xn,ts,tm,tmu,tl):
            return xn[n,d]**2-2*xn[n,d]*T.sum(tmu[d,:]*tm[n,:])\
                    +T.sum(tmu[d,:]*tm[n,:])**2\
                    +((tmu[d,:]**2+tl[d,:])*(tm[n,:]**2+ts[n,:])).sum()-(tmu[d,:]**2*tm[n,:]**2).sum()
             
        def inter_loop(n,xn,ts,tm,tmu,tl):
            
            outputs,_=theano.scan(fn=normalCalc,
                                sequences=T.arange(D),
                                non_sequences=[n,xn,ts,tm,tmu,tl],
                                strict=True
                                )
            return outputs
        
        final,_=theano.scan(fn=inter_loop,
                            sequences=T.arange(S),
                            non_sequences=[x,et_s,t_m,t_mu,et_l],
                            strict=True
                            )
        
        
        #=======================================================================
        # Calculate Gdk expectation
        #=======================================================================
        def gcalc(k,d,tc,tf,tmu,tl):
            return +0.5*(T.psi(tc[k,0])-T.log(tf[k,0]))-0.5*(tc[k,0]/tf[k,0])*(tmu[d,k]**2+tl[d,k])
        
        def inter_g_loop(n,tc,tf,tmu,tl):
            
            outputs,_=theano.scan(fn=gcalc,
                                  sequences=T.arange(K),
                                  non_sequences=[n,tc,tf,tmu,tl],
                                  strict=True
                                  )
            return outputs
        
        finalg,_=theano.scan(fn=inter_g_loop,
                                 sequences=T.arange(D),
                                 non_sequences=[et_c,et_f,t_mu,et_l],
                                 strict=True
                                 )
        
        #=======================================================================
        # Calculate the expectation of logy
        #=======================================================================
        def y_calc(j,k,n,zt,te1,te2,ts,tm):
            return zt[n,k,j]*(-0.5*T.log(2*np.pi)+0.5*(T.psi(te1[k,j])-T.log(te2[k,j]))-0.5*(te1[k,j]/te2[k,j])*(ts[n,k]+tm[n,k]**2))
        
        def deepest_loop(k,n,zt,te1,te2,ts,tm):
            out,_=theano.scan(fn=y_calc,
                              sequences=T.arange(J),
                              non_sequences=[k,n,zt,te1,te2,ts,tm],
                              strict=True
                              )
            return out 
        
        def not_so_deep_loop(n,zt,te1,te2,ts,tm):
            out,_=theano.scan(fn=deepest_loop,
                              sequences=T.arange(K),
                              non_sequences=[n,zt,te1,te2,ts,tm],
                              strict=True
                              )
            return out
        
        final_y,_=theano.scan(fn=not_so_deep_loop,
                              sequences=T.arange(S),
                              non_sequences=[ezeta,et_e_1,et_e_2,et_s,t_m],
                              strict=True
                              )
#      
        #=======================================================================
        # calculate the likelihood term 
        #=======================================================================
        expectation_log_p_a=g_1*T.log(g_2)+(g_1-1)*(T.psi(et_g_1)-T.log(et_g_2))-g_2*(et_g_1/et_g_2)-T.log(T.gamma(g_1))
        
        expectation_log_p_phi=a*T.log(b)+(a-1)*(T.psi(et_a)-T.log(et_b))-b*(et_a/et_b)-T.log(T.gamma(a))
        
        expectation_log_u_k=T.psi(et_g_1)-T.log(et_g_2)+(et_g_1/et_g_2-1)*(T.psi(et_tau)-T.psi(et_tau+eh_tau))
        
        expectation_log_lambda_k=c*T.log(f)+(c-1)*(T.psi(et_c)-T.log(et_f))-f*(et_c/et_f)-T.log(T.gamma(c))
        
        expectation_log_varpi=-T.log(T.prod(T.gamma(xi),1))/T.gamma(T.sum(xi,1))+T.sum((xi-1)*(T.psi(et_xi)-(T.psi(T.sum(et_xi,1))).dimshuffle(0, 'x')),1)
        
        expectation_log_skj=e_1*T.log(e_2)-(e_1+1)*(T.log(et_e_2)-T.psi(et_e_1))-e_2*(et_e_1/et_e_2)-T.log(T.gamma(e_1))
        
        expectation_log_gdk=-0.5*T.log(2*np.pi)+finalg
        
        expectation_log_zdk=q_z*T.cumsum(T.psi(eh_tau)-T.psi(et_tau+eh_tau))+(1.0-q_z)*mult_bound
        
        expectation_log_varepsilon=ezeta*(T.psi(et_xi)-T.psi(T.sum(et_xi,1)).dimshuffle(0,'x'))
        
        expectation_log_y=final_y
        
        expectation_log_x=-0.5*T.log(2*np.pi)+0.5*(T.psi(et_a)-T.log(et_b))-0.5*(et_a/et_b)*final
      
        #=======================================================================
        # Combine all the terms to get the likelihood
        #=======================================================================
        likelihood=expectation_log_p_a \
                    +expectation_log_p_phi \
                    +T.sum(expectation_log_u_k) \
                    +T.sum(expectation_log_lambda_k) \
                    +T.sum(expectation_log_varpi) \
                    +T.sum(expectation_log_skj) \
                    +T.sum(expectation_log_gdk) \
                    +T.sum(expectation_log_zdk)\
                    +T.sum(expectation_log_varepsilon)\
                    +T.sum(expectation_log_y) \
                    +T.sum(expectation_log_x)                           
        #=======================================================================
        # calculate the entropy term                     
        #=======================================================================
        entropy=et_g_1-T.log(et_g_2)+(1-et_g_1)*T.psi(et_g_1)+T.log(T.gamma(et_g_1)) \
                +et_a-T.log(et_b)+(1-et_a)*T.psi(et_a)+T.log(T.gamma(et_a)) \
                +T.sum(T.log(T.gamma(et_tau)*T.gamma(eh_tau)/T.gamma(eh_tau+et_tau))-(et_tau-1)*T.psi(et_tau)-(eh_tau-1)*T.psi(eh_tau)+(et_tau+eh_tau-2)*T.psi(et_tau+eh_tau)) \
                +T.sum(et_c-T.log(et_f)+T.log(T.gamma(et_c))+(1-et_c)*T.psi(et_c)) \
                +T.sum(T.prod(T.gamma(et_xi),1)/T.gamma(T.sum(et_xi,1))-(J-T.sum(et_xi,1))*T.psi(T.sum(et_xi,1))-(T.sum((et_xi-1.0)*T.psi(et_xi),1))) \
                +T.sum(et_e_1+T.log(et_e_2*T.gamma(et_e_1))-(1+et_e_1)*T.psi(et_e_1)) \
                +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.log(T.prod(et_l,1))) \
                -T.sum((1.0-q_z)*T.log(1.0-q_z))-T.sum(q_z*T.log(q_z)) \
                +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.sum(T.log(et_s),1)) \
                -T.sum(ezeta*T.log(ezeta))

        lower_bound=likelihood+entropy
        
        prior=[a,b,c,f,g_1,g_2,e_1,e_2,xi]
        var=[t_s,t_m,zeta,t_tau,omega,h_tau,t_a,t_c,t_f,t_g_1,t_g_2,t_e_1,t_e_2,t_xi,t_l,t_mu,t_b]
        
       
        self.lowerBoundFunction=theano.function(prior+var+[x,K,D,S],lower_bound,allow_input_downcast=True)
        self.likelihoodFunction=theano.function(prior+var+[x,K,D,S],likelihood,allow_input_downcast=True)
        self.entropyFunction=theano.function(prior+var+[K,D,S],entropy,allow_input_downcast=True,on_unused_input='warn')
        
    
   
        
    #===============================================================================
    # need to remove columns if feature is not active
    #===============================================================================
    def contract_arrays(self,used_k,J,N,D,xi,posterior):
        
        #row delete
        xi=xi[used_k,:]
        posterior.tilde_xi=posterior.tilde_xi[used_k,:]
        posterior.tilde_eta_1=posterior.tilde_eta_1[used_k,:]
        posterior.tilde_eta_2=posterior.tilde_eta_2[used_k,:]
        
        #column delete
        posterior.tilde_mu=posterior.tilde_mu[:,used_k]
        posterior.tilde_s=posterior.tilde_s[:,used_k]
        posterior.tilde_m=posterior.tilde_m[:,used_k]
        posterior.omega=posterior.omega[:,used_k]
        posterior.zeta=posterior.zeta[:,used_k,:]
        posterior.tilde_lambda=posterior.tilde_lambda[:,used_k]
        posterior.tilde_tau=posterior.tilde_tau[used_k]
        posterior.hat_tau=posterior.hat_tau[used_k]
        posterior.tilde_c=posterior.tilde_c[used_k]
        posterior.tilde_f=posterior.tilde_f[used_k]
        
        return xi,posterior
    
    #=======================================================================
    # need to expand the parameters after the feature update
    #=======================================================================
    def expand_arrays(self,diff,J,N,D,xi,posterior):
        #vstack
        xi=np.vstack((xi,np.ones((diff,J))))
        posterior.tilde_xi=np.vstack((posterior.tilde_xi,np.ones((diff,J))*(1.0/J)))
        posterior.tilde_eta_1=np.vstack((posterior.tilde_eta_1,np.ones((diff,J))))
        posterior.tilde_eta_2=np.vstack((posterior.tilde_eta_2,np.ones((diff,J))))
        
        #hstack
        posterior.tilde_mu=np.hstack((posterior.tilde_mu,r.normal(0,1,size=(D,diff))))
        posterior.tilde_s=np.hstack((posterior.tilde_s,r.gamma(1,1,size=(N,diff))))
        posterior.tilde_m=np.hstack((posterior.tilde_m,r.normal(0,1,size=(N,diff))))
        posterior.omega=np.hstack((posterior.omega,np.ones((D,diff))))
        inter_zeta=np.exp(psi(posterior.tilde_xi[-diff:])-psi(posterior.tilde_xi[-diff:].sum(1).reshape(-1,1).repeat(axis=1,repeats=J)))
        posterior.zeta=np.hstack((posterior.zeta,inter_zeta[np.newaxis,:,:].repeat(axis=0,repeats=N)))
        posterior.tilde_lambda=np.hstack((posterior.tilde_lambda,np.ones((D,diff))))
        posterior.tilde_tau=np.hstack((posterior.tilde_tau,np.ones(diff)))
        posterior.hat_tau=np.hstack((posterior.hat_tau,np.ones(diff)))
        posterior.tilde_c=np.hstack((posterior.tilde_c,np.ones(diff)))
        posterior.tilde_f=np.hstack((posterior.tilde_f,np.ones(diff)))
        
        return xi,posterior
    
    #===========================================================================
    # Some naive creation of synthetic data
    #===========================================================================
    def create_synthetic_data(self,N,D,K):
        G=np.random.normal(0,1,size=(self.D,self.K))
        y=np.random.normal(0,1,size=(self.K,N))
        return np.dot(G,y).T,G,y
    
     #===============================================================================
    # Run the full stochastic IBP ICA algorithm as described in ....
    #===============================================================================
    def run_stochastic_IBP_ICA(self,x):
        
       
        #initialize K
        #K_d=r.randint(4,8,self.D)
        #K=max(K_d)
        
        #self.K=K
        #initialize the prior parameters of the model
        prior=self.initialize_prior_parameters(self.K,self.J)
        
        #init the posterior parameters 
        #These will be updated with gradient steps
        posterior=self.initialize_posterior_parameters(self.K, self.D, self.S, self.J, False)
        
     
        i=1;
        
        #sample the data 
        random_indices=np.random.randint(0,len(x),S)
        miniBatch=x[random_indices,:]
        
        #init zeta for local calculations (if needed initialize the others)
        #posterior=self.init_local(miniBatch,self.S, self.K, self.J, posterior)
        
        #track the lower bound
        LL=[]
        
        
        while True:
            
            #set step size
            rho=(1.0+i)**-.75
            i+=1    
           
            #keep old value to know if we need to extend the variables
            #K_old=max(K_d)
            
            #Perform tha MH step to update the number of generated latent features
            #K_d=feature_update(K_d, D, S, prior,posterior,x_s)
         
         
            #extract the max number of active features
            #K=max(K_d)
    
            prior1=[prior.a,prior.b,prior.c,prior.f,prior.gamma_1,prior.gamma_2,prior.eta_1,prior.eta_2,prior.xi]
            var=[posterior.tilde_s,posterior.tilde_m,posterior.zeta,posterior.tilde_tau,posterior.omega,\
                 posterior.hat_tau,posterior.tilde_a,posterior.tilde_c,posterior.tilde_f,posterior.tilde_gamma_1,posterior.tilde_gamma_2,posterior.tilde_eta_1,posterior.tilde_eta_2,\
                 posterior.tilde_xi,posterior.tilde_lambda,posterior.tilde_mu,posterior.tilde_b]     
            
            print("LowerBound:",self.lowerBoundFunction(*(prior1+var),x=miniBatch,K=self.K,D=self.D,S=self.S))
            #print("The new K is:",K)
    
            #if the K is different we need to update the shapes
            #if K_old!=K:
            #    diff=K-K_old
            #    prior.xi,posterior=expand_arrays(diff,J,S,D,prior.xi,posterior)
            
            #update the local parameters
            print('Updating the local parameters...')
            posterior=self.update_local_parameters(miniBatch, self.K, self.J, posterior)
            
            
            prior1=[prior.a,prior.b,prior.c,prior.f,prior.gamma_1,prior.gamma_2,prior.eta_1,prior.eta_2,prior.xi]
            var=[posterior.tilde_s,posterior.tilde_m,posterior.zeta,posterior.tilde_tau,posterior.omega,\
                 posterior.hat_tau,posterior.tilde_a,posterior.tilde_c,posterior.tilde_f,posterior.tilde_gamma_1,posterior.tilde_gamma_2,posterior.tilde_eta_1,posterior.tilde_eta_2,\
                 posterior.tilde_xi,posterior.tilde_lambda,posterior.tilde_mu,posterior.tilde_b]     
            
            print("LowerBound:",self.lowerBoundFunction(*(prior1+var),x=miniBatch,K=self.K,D=self.D,S=self.S))
             
            #calculate the intermediate global parameters for the gradient step
            print('Calculating Intermediate Parameters...')
            params_hat=self.intermediate_global_parameters(miniBatch, self.S,self.K,self.J, self.D, prior,posterior,False)
            
            
            #perform the gradient step         
            posterior=self.gradient_step(rho, self.S, params_hat,posterior,False)
            
            prior1=[prior.a,prior.b,prior.c,prior.f,prior.gamma_1,prior.gamma_2,prior.eta_1,prior.eta_2,prior.xi]
            var=[posterior.tilde_s,posterior.tilde_m,posterior.zeta,posterior.tilde_tau,posterior.omega,\
                 posterior.hat_tau,posterior.tilde_a,posterior.tilde_c,posterior.tilde_f,posterior.tilde_gamma_1,posterior.tilde_gamma_2,posterior.tilde_eta_1,posterior.tilde_eta_2,\
                 posterior.tilde_xi,posterior.tilde_lambda,posterior.tilde_mu,posterior.tilde_b]     
            
            print("LowerBound:",self.lowerBoundFunction(*(prior1+var),x=miniBatch,K=self.K,D=self.D,S=self.S))
            
            
            #search for unused components, remove them and reshape the arrays as necessary
            self.K,prior.xi,posterior=self.unused_components(self.K, self.S, self.D, self.J, prior.xi, posterior)
            
            #append the new bound to the list
            #LL.append(self.lowerBoundFunction(*(prior+posterior),x=miniBatch,K=self.K,D=self.D,S=self.S))
            LL.append(0)
            print('--------------------------------------------------------------')
            print('The log-likelihood for this iteration is',LL[-1])
            print('--------------------------------------------------------------')
                    
            
            #Check for convergence
            if (np.isnan(LL[-1])):
                print("An error occured, the lower bound is NaN. Please Debug.")
                break
            elif (LL[-1]>0):
                print('\n########################################################################')
                print("# What? Positive log likelihood? That can't be possible. Please Debug. #")
                print('########################################################################\n')
    
                break
            
            if (len(LL)>1):
                if (LL[-1]<LL[-2]):
                    print('\n###################################')
                    print("# Error! The bound is decreasing? #")
                    print('###################################\n')
                    break
                elif (abs(LL[-1]-LL[-2])<0):#np.finfo(np.float32).eps):
                    print('\n############################################################')
                    print("# Terminating due to insufficient increase in lower bound! #")
                    print('############################################################\n')
    
                    break
        
            print("The new bound is:")
            print(LL[-1])
#===============================================================================
#  Main
#===============================================================================
if __name__ == '__main__':
    
    K=5
    N=10000
    D=4
    J=8
    S=50
    
    z=IBP_ICA(K,D,J,S)
    
    x,G,y=z.create_synthetic_data(N,D,K)
    z.lowerBoundCalc()
    z.run_stochastic_IBP_ICA(x)
    