'''
Created on Jun 5, 2016

@author: kon
'''
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
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


#===============================================================================
# Initialize the prior parameters
#===============================================================================
def initialize_prior_parameters(K,J,show_values=False):
    gamma_1=1
    gamma_2=1
    eta_1=1
    eta_2=1
    c=1
    f=1
    a=1
    b=1
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
def initialize_posterior_parameters(K,N,D,J,show_values=False):
    
    p=recordclass("posterior",['tilde_a','tilde_b','tilde_xi','tilde_eta_1','tilde_eta_2','tilde_lambda','tilde_mu','tilde_gamma_1','tilde_gamma_2',\
                               'omega','tilde_c','tilde_f','tilde_tau','hat_tau','zeta',"tilde_s","tilde_m"])
    posterior=p(1,1,np.ones((K,J))*(1.0/J),np.ones((K,J)),np.ones((K,J)),r.gamma(1,1,size=(D,K)),r.normal(0,1,size=(D,K)),1,1,r.random((D,K)),np.ones(K),np.ones(K),np.ones(K),np.ones(K),0,0,0)
    
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
def feature_update(K_d,D,N,prior,posterior,x=1):   
    
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
def update_local_parameters(x,K,J,posterior):
    
    S=len(x)
    
    tolerance=10**-6
    
    expectation_phi=posterior.tilde_a/posterior.tilde_b
    
    old_tilde_s=posterior.tilde_s
    old_tilde_m=posterior.tilde_m
    old_zeta=posterior.zeta
    
    while True:
        for s in range(S):
            for k in range(K):
                
                posterior.tilde_s[s,k]=(np.dot(posterior.tilde_eta_1[k,:]/posterior.tilde_eta_2[k,:],posterior.zeta[s,k,:])+expectation_phi*(np.dot(posterior.tilde_mu[:,k].T,posterior.tilde_mu[:,k])))**-1
                posterior.tilde_m[s,k]=expectation_phi*posterior.tilde_s[s,k]*np.dot(posterior.tilde_mu[:,k].T,x[s,:].T)
                
                #===============================================================
                # edw paizei ena major lathaki/ leipei to y
                #===============================================================
                for j in range(J):
                    posterior.zeta[s,k,j]=np.exp(psi(posterior.tilde_xi[k,j])-psi(posterior.tilde_xi[k,:].sum())+(1/2)*(psi(posterior.tilde_eta_1[k,j])-np.log(posterior.tilde_eta_2[k,j])) \
                    -(1/2)*(posterior.tilde_s[s,k]+posterior.tilde_m[s,k]**2)*(posterior.tilde_eta_1[k,j]/posterior.tilde_eta_2[k,j]))
                
                posterior.zeta[s,k,:]/=(posterior.zeta[s,k,:]).sum()
        
        #converged?
        if (abs(norm(old_tilde_s)-norm(posterior.tilde_s))<tolerance) and abs(norm(old_tilde_m)-norm(posterior.tilde_m))<tolerance and abs(norm(old_zeta)-norm(posterior.zeta))<tolerance:
            break;
        
        old_tilde_s=posterior.tilde_s
        old_tilde_m=posterior.tilde_m
        old_zeta=posterior.zeta
    return posterior


#===============================================================================
# Update the intermediate global parameters to proceed to the gradient step
#===============================================================================
def intermediate_global_parameters(x,N,K,J,D,prior,posterior,print_values=False):
    S=len(x)
    
    p=collections.namedtuple("params_hat",['hat_a','hat_b','hat_xi','hat_eta_1','hat_eta_2','hat_lambda','hat_mu','hat_gamma_1','hat_gamma_2','hat_omega','hat_c','hat_f','hat_tau_1','hat_tau_2'])
    
    hat_gamma_1=prior.gamma_1+K-1
    hat_gamma_2=prior.gamma_2-(psi(posterior.tilde_tau[:-1])-psi(posterior.tilde_tau[:-1]+posterior.hat_tau[:-1])).sum()
    hat_a=prior.a+(N*D)/2
    
    hat_omega=np.zeros((D,K))
    hat_b=np.zeros((N,))
    hat_c=np.zeros(K)
    hat_f=np.zeros(K)
    hat_tau_1=np.zeros(K)
    hat_tau_2=np.zeros(K)
    hat_eta_1=np.zeros((S,K,J))
    hat_eta_2=np.zeros((S,K,J))
    hat_xi=np.zeros((S,K,J))
    hat_lambda=np.zeros((S,D,K))
    hat_mu =np.zeros((S,D,K))
    
    q_k=calculate_q(K, posterior.tilde_tau, posterior.hat_tau)
    q_z=calculate_posterior_z(posterior.omega)

    for k in range(K):
        mult_bound=multinomial_bound(k, q_k, posterior.tilde_tau, posterior.hat_tau)

        hat_omega[:,k]=(psi(posterior.tilde_tau[:k+1])-psi(posterior.tilde_tau[:k+1]+posterior.hat_tau[:k+1])).sum()+mult_bound \
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
                hat_mu[s,d,k]=N*(hat_lambda[s,d,k]**-1)*(posterior.tilde_m[s,k])*(x[s,d]-np.dot(posterior.tilde_mu[d,:],posterior.tilde_m[s,:].T))
            hat_b[s]=prior.b+(N/2)*np.dot((x[s,:].T-np.dot(posterior.tilde_mu,posterior.tilde_m[s,:].T)),(x[s,:].T-np.dot(posterior.tilde_mu,posterior.tilde_m[s,:].T)).T)

    params_hat=p(hat_a,hat_b,hat_xi,hat_eta_1,hat_eta_2,hat_lambda,hat_mu,hat_gamma_1,hat_gamma_2,hat_omega,hat_c,hat_f,hat_tau_1,hat_tau_2)

    if (print_values):
        print(params_hat)
        
    return params_hat
            
        
        
#===============================================================================
# Calculate the posterior of z, q(z_{dk}=1)
#===============================================================================
def calculate_posterior_z(posterior_omega):
    return 1.0/(1.0+np.exp(-posterior_omega))

#===============================================================================
# Calculate the bound based on multinomial expansion
#===============================================================================
def multinomial_bound(k,q_k,tau_1,tau_2):
    mult_bound=(q_k[:k+1]*psi(tau_2[:k+1])).sum()
    cur_sum=0
    q_sum=0
    for m in range(k-1):
        for n in range(m+1,k):
            q_sum+=q_k[n]
        cur_sum+=q_sum*psi(tau_1[m])
        q_sum=0
        
    mult_bound+=cur_sum 
    cur_sum=0
    q_sum=0
    for m in range(k):
        for n in range(m,k):
            q_sum+=q_k[n]
        cur_sum+=q_sum*psi(tau_1[m]+tau_2[m])
        q_sum=0
        
    mult_bound-=cur_sum+(q_k[:k+1]*np.log(q_k[:k+1])).sum()
    return mult_bound
    
#===============================================================================
# Calculate q for the calculation of the multinonmial expansion bound
#===============================================================================
def calculate_q(K,tau_1,tau_2):
    q_k=np.zeros(K)
    for k in range(K):
        q_k[k]=np.exp(psi(tau_2[k])+(psi(tau_1[:k])).sum()-(psi(tau_1[:k+1]+tau_2[:k+1])).sum())
    q_k/=q_k.sum()
    return q_k

#===============================================================================
# Make the final gradient step for the global parameters
#===============================================================================
def gradient_step(rho,S,params_hat,posterior,print_values=False):
    
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
def init_local(x,S,K,J,posterior,custom_init=False):
    
    if (not custom_init):
        posterior.zeta=np.zeros((S,K,J))
        tilde_s_inv=np.zeros((S,K))
        posterior.tilde_m=np.zeros((S,K))
        for s in range(S):
            posterior.zeta[s,:,:]=np.exp(psi(posterior.tilde_xi)-psi(posterior.tilde_xi.sum(1).reshape(-1,1).repeat(axis=1,repeats=J))\
                                         -(1/2)*(psi(posterior.tilde_eta_1)-np.log(posterior.tilde_eta_2)\
            +posterior.tilde_eta_1/posterior.tilde_eta_2))
            tilde_s_inv[s,:]=np.sum(posterior.zeta[s,:,:]*(posterior.tilde_eta_1/posterior.tilde_eta_2),1)+(posterior.tilde_a/posterior.tilde_b)*(np.diag(np.dot(posterior.tilde_mu.T,posterior.tilde_mu)))
            for k in range(K):
                posterior.tilde_m[s,k]=(tilde_s_inv[s,k]**-1)*(posterior.tilde_a/posterior.tilde_b)*np.dot(posterior.tilde_mu[:,k].T,x[s,:])
        posterior.tilde_s=tilde_s_inv**-1
    else:
        posterior.tilde_s_inv=r.gamma(1,1,size=(S,K))
        pca=PCA(n_components=K)
        pca.fit(x)
        for s in range(S):
            posterior.zeta[s,:,:]=np.exp(psi(tilde_xi)-psi(sum(tilde_xi,2)))+psi(tilde_eta_1)-np.log(tilde_eta_2)-tilde_eta_1/tilde_eta_2
        
    return posterior


#===============================================================================
# Run the full stochastic IBP ICA algorithm as described in ....
#===============================================================================
def run_stochastic_IBP_ICA(x):
    
    #Decide on the number of components
    J=10
    
    #set the sample size
    S=10
    
    D=x.shape[1]
    
    #initialize K
    K_d=r.randint(4,8,D)
    K=max(K_d)
        
    #initialize the prior parameters of the model
    prior=initialize_prior_parameters(K,J,False)
    
    #init the posterior parameters 
    #These will be updated with gradient steps
    posterior=initialize_posterior_parameters(K, D, S, J, False)
    
 
    i=1;
    
    #sample the data 
    random_indices=r.randint(0,len(x),S)
    x_s=x[random_indices,:]
    
    #init zeta for local calculations (if needed initialize the others)
    posterior=init_local(x_s,S, K, J, posterior)
    
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
        K=max(K_d)

        
        print("The new K is:",K)

        #if the K is different we need to update the shapes
        #if K_old!=K:
        #    diff=K-K_old
        #    prior.xi,posterior=expand_arrays(diff,J,S,D,prior.xi,posterior)
        
        #update the local parameters
        posterior=update_local_parameters(x_s, K, J, posterior)
        
         
        #calculate the intermediate global parameters for the gradient step
        params_hat=intermediate_global_parameters(x_s, len(x_s),K,J, D, prior,posterior,False)
        
        
        #perform the gradient step         
        posterior=gradient_step(rho, S, params_hat,posterior,False)
        
        #search for unused components, remove them and reshape the arrays as necessary
        K,prior.xi,posterior=unused_components(K, S, D, J, prior.xi, posterior)
        
        #append the new bound to the list
        LL.append(lower_bound_calculation(x_s,prior, posterior, K, J,S, D))
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
            elif (abs(LL[-1]-LL[-2])<np.finfo(np.float32).eps):
                print('\n############################################################')
                print("# Terminating due to insufficient increase in lower bound! #")
                print('############################################################\n')

                break
    
        print("The new bound is:")
        print(LL[-1])
        

def unused_components(K,N,D,J,xi,posterior):
    '''
    Check for unused components and reshape arrays if necessary
    '''
    used_k=search_used_components(posterior.omega,K)
    if (len(used_k)!=K):
        xi,posterior=contract_arrays(used_k, J, N, D, xi, posterior)
    return len(used_k),xi,posterior
    

def search_used_components(omega,K):
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

def calculate_lower_bound(x,pr,pos,K,J,N,D,debug=False):
    q_z=calculate_posterior_z(posterior.omega)
    
    likelihood=pr.g_1*np.log(pr.g_2)+(pr.g_1-1)*(psi(pos.t_g_1)-np.log(pos.t_g_2))-pr.g_2*(post.t_g_1/pos.t_g_2)-np.log(gamma(pr.g_1)) \
                    +pr.a*np.log(pr.b)+(pr.a-1)*(psi(pos.t_a)-np.log(pos.t_b))-pr.b*(pos.t_a/pos.t_b)-np.log(gamma(pr.a)) \
                    +np.sum(T.psi(pos.t_g_1)-np.log(pos.t_g_2)+(pos.t_g_1/pos.t_g_2-1)*(psi(pos.t_tau)-psi(pos.t_tau+pos.h_tau))) \
                    +T.sum(c*T.log(f)+(c-1)*(T.psi(t_c)-T.log(t_f))-f*(t_c/t_f)-T.log(T.gamma(c))) \
                    +T.sum(T.sum((xi-1)*(T.psi(t_xi)-T.psi(T.sum(t_xi,1))),1)) \
                    +T.sum(e_1*T.log(e_2)-(e_1+1)*(T.log(t_e_2)-T.psi(t_e_1))-e_2*(t_e_1/t_e_2)) \
                    +T.sum(0.5*(T.psi(t_c)-T.log(t_f))-0.5*(t_mu**2+t_l)*(t_c/t_f)) \
                    +T.sum(q_z*T.cumsum(T.psi(h_tau)-T.psi(t_tau+h_tau))+(1.0-q_z)*mult_bound)\
                    +T.sum(0.5*zeta*(T.psi(t_e_1)-T.log(t_e_2)-(t_m**2+t_s)*(t_e_1/t_e_2))) \
                    +T.sum(0.5*(T.psi(t_a)-T.log(t_b)) \
                           -0.5*(T.dot(T.transpose(x), x)-T.dot(T.transpose(x),T.dot(T.transpose(t_mu),t_m))-T.dot(T.dot(T.transpose(t_m),T.transpose(t_mu)),x)+T.dot(T.dot(T.transpose(t_m),T.nlinalg.trace(t_l)+T.dot(T.transpose(t_mu),t_mu)),t_m))*(-t_a/t_b))

def lower_bound_calculation(x,prior,posterior,K,J,N,D,debug=True):
    '''
    Calculate the variational lower bound in order to assess convergence
    '''
    q_z=calculate_posterior_z(posterior.omega)
   
    first_term=prior.gamma_1*np.log(prior.gamma_2)+(prior.gamma_1-1)*(psi(posterior.tilde_gamma_1)-np.log(posterior.tilde_gamma_2))-prior.gamma_2*(posterior.tilde_gamma_1/posterior.tilde_gamma_2)-np.log(gamma(prior.gamma_1))
    
    second_term=prior.a*np.log(prior.b)+(prior.a-1)*(psi(posterior.tilde_a)-np.log(posterior.tilde_b))-prior.b*(posterior.tilde_a/posterior.tilde_b)-np.log(gamma(prior.a))
    
    third_term=K*(psi(posterior.tilde_gamma_1)-np.log(posterior.tilde_gamma_2))+(posterior.tilde_gamma_1/posterior.tilde_gamma_2-1)*(psi(posterior.tilde_tau)-psi(posterior.tilde_tau+posterior.hat_tau)).sum()
    
    fourth_term=K*prior.c*np.log(prior.f)+(prior.c-1)*(psi(posterior.tilde_c)-psi(posterior.tilde_c+posterior.tilde_f)).sum()-prior.f*(posterior.tilde_c/posterior.tilde_f).sum()-np.log(gamma(prior.c))
    
    fifth_term=-(np.log(beta_calc(prior.xi))).sum()+np.multiply(prior.xi-1,(psi(posterior.tilde_xi)-psi(posterior.tilde_xi.sum(axis=1)).reshape(-1,1).repeat(axis=1,repeats=J))).sum()
    
    sixth_term=(prior.eta_1*np.log(prior.eta_2)).sum()-((prior.eta_1+1)*(np.log(posterior.tilde_eta_2)).sum()-psi(posterior.tilde_eta_1)).sum()-prior.eta_2*(posterior.tilde_eta_1/posterior.tilde_eta_2).sum()
    
    seventh_term=psi(posterior.tilde_c).sum()-np.log(posterior.tilde_f).sum()+((posterior.tilde_mu**2+posterior.tilde_lambda)*(posterior.tilde_c/posterior.tilde_f)).sum()
    
    eight_term=0
    q_k=calculate_q(K, posterior.tilde_tau, posterior.hat_tau)
    for d in range(D):
        for k in range(K):
            eight_term+=q_z[d,k]*(psi(posterior.hat_tau[:k]).sum()-psi(posterior.hat_tau[:k]+posterior.tilde_tau[:k]).sum())+((1-q_z[d,k])*multinomial_bound(k, q_k, posterior.tilde_tau, posterior.hat_tau)).sum()
    
    ninth_term=np.log(posterior.tilde_eta_2).sum()-psi(posterior.tilde_eta_1).sum()+(posterior.tilde_s+posterior.tilde_m**2).sum()+(posterior.tilde_eta_1/posterior.tilde_eta_2).sum()
    
    
    last_term=0
    for n in range(N):
        last_term+=(1/2)*(psi(posterior.tilde_a)-np.log(posterior.tilde_b))-(1/2)*((posterior.tilde_a/posterior.tilde_b)*(np.dot(x[n,:],x[n,:].T))-np.dot(np.dot(x[n,:],posterior.tilde_mu),posterior.tilde_m[n,:]).sum()\
        -(np.dot(np.dot(posterior.tilde_m[n,:].T,posterior.tilde_mu.T),x[n,:])).sum()\
        +np.dot(np.dot(np.dot(posterior.tilde_m[n,:],posterior.tilde_mu.T),posterior.tilde_mu),posterior.tilde_m[n,:].T).sum())
   
    if debug:
        print("1",first_term)
        print("2",second_term)
        print("3",third_term)
        print("4",fourth_term)
        print("5",fifth_term)
        print("6",sixth_term)
        print("7",seventh_term)
        print("8",eight_term)
        print("9",last_term)
    return first_term+second_term+third_term+fourth_term+fifth_term+sixth_term+seventh_term+eight_term+ninth_term+last_term+entropy_calculation(posterior, K, J, D, True)                                                                                           
    
    
def entropy_calculation(posterior,K,J,D,print_values=True):
    H_q_y=(K/2)*(1+np.log(2*np.pi))+(1/2)*np.log(posterior.tilde_s).sum()
    H_q_a=posterior.tilde_gamma_1-np.log(posterior.tilde_gamma_2)+np.log(gamma(posterior.tilde_gamma_1))+(1-posterior.tilde_gamma_1)*psi(posterior.tilde_gamma_1)
    H_q_phi=posterior.tilde_a+np.log(posterior.tilde_b)+np.log(gamma(posterior.tilde_a))+(1-posterior.tilde_a)*psi(posterior.tilde_a)
    
    H_q_u=0
    H_q_z=0
    H_lambda=0
    H_varpi=0
    H_s=0
    H_zeta=posterior.zeta.sum()
    q_z=1.0/(1.0+np.exp(-posterior.omega))
    for k in range(K):
        H_q_u+=np.log(beta(posterior.tilde_tau[k]+posterior.hat_tau[k],1))-(posterior.tilde_tau[k]-1)*psi(posterior.tilde_tau[k])-(posterior.hat_tau[k]-1)*psi(posterior.hat_tau[k])\
        +(posterior.tilde_tau[k]+posterior.hat_tau[k]-2)*psi(posterior.tilde_tau[k]+posterior.hat_tau[k])
        H_lambda+=posterior.tilde_c[k]-np.log(posterior.tilde_f[k])+np.log(gamma(posterior.tilde_c[k]))+(1-posterior.tilde_c[k])*psi(posterior.tilde_c[k])
        for d in range(D):
            H_q_z+=-(1-q_z[d,k])*np.log(1-q_z[d,k])-q_z[d,k]*np.log(q_z[d,k])
        H_varpi+=-(J-posterior.tilde_xi[k,:].sum())*psi(posterior.tilde_xi[k,:].sum())
        for j in range(J):
            H_varpi+=np.log(beta(posterior.tilde_xi[k,j],1))-(posterior.tilde_xi[k,j]-1)*psi(posterior.tilde_xi[k,j])
            H_s+=posterior.tilde_eta_1[k,j]-np.log(posterior.tilde_eta_2[k,j])+np.log(gamma(posterior.tilde_eta_1[k,j]))+(1-posterior.tilde_eta_1[k,j])*psi(posterior.tilde_eta_1[k,j])
    
    if (print_values):
        print("H(q_y):",H_q_y,"H(q_a):",H_q_a,"H(q_phi):",H_q_phi,"H(q_u):",H_q_u,"H(q_z):",H_q_z,"H(q_lambda):",H_lambda,"H(q_varpi):",H_varpi,"H(q_s):",H_s,"H(q_zeta):",H_zeta)
        
    return H_q_y+H_q_a+H_q_phi+H_q_u+H_q_z+H_lambda+H_varpi+H_s+H_zeta
    
def create_synthetic_data(D,K,N):
    G=r.normal(size=(D,K))
    y=r.normal(size=(K,N))
    return np.dot(G,y)
    
#===============================================================================
# need to remove columns if feature is not active
#===============================================================================
def contract_arrays(used_k,J,N,D,xi,posterior):
    
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
def expand_arrays(diff,J,N,D,xi,posterior):
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
        
#===============================================================================
#  Main
#===============================================================================
if __name__ == '__main__':
    x=create_synthetic_data(10, 9, 100000).T
    run_stochastic_IBP_ICA(x)
    
    
#     gamma_1,gamma_2,eta_1,eta_2,c,f,a,b,xi=initialize_prior_parameters(5,5,False)
#     
#     tilde_gamma_1,tilde_gamma_2,tilde_eta_1,tilde_eta_2,tilde_c,tilde_f,tilde_a,tilde_b,tilde_xi,tilde_tau,hat_tau,omega,tilde_mu,tilde_lambda=initialize_posterior_parameters(K, D, N, 3, False)
#     feature_update(K, D, N,gamma_1,gamma_2,c,f,tilde_a,tilde_b)