'''
Created on Jun 29, 2016

@author: kon
'''
import theano
import theano.tensor as T
import numpy as np
import time
import pickle
import sys
from numpy.linalg import norm
from numpy import newaxis
from theano.ifelse import ifelse
from theano import pp
from theano.compile.io import Out
from theano.gradient import consider_constant
from scipy.special import psi
theano.config.exception_verbosity='high'
#theano.config.optimizer='fast_run'
theano.config.traceback.limit=20
from six.moves import cPickle

class IBP_ICA:
    
    def __init__(self,K,D,J, S,N,rho):
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
        xi=2*np.ones((self.K,self.J))
        
        #=======================================================================
        # The scalar parameters
        #=======================================================================
        rt_a=np.log(2.0)
        rt_b=np.log(2.0)
        t_g_1=2.0
        t_g_2=2.0
        
        #=======================================================================
        # matrices
        #=======================================================================
        rt_e_1=np.log(2.0)*np.ones((self.K,self.J),dtype='float32')
        rt_e_2=np.log(2.0)*np.ones((self.K,self.J),dtype='float32')
        rt_xi=np.log(2.0)*np.ones((self.K,self.J),dtype='float32')
        rt_l=np.log(2.0)*np.ones((self.D,self.K),dtype='float32')
        t_mu=np.ones((self.D,self.K))
        omega=np.ones((self.D,self.K))
        rt_s=np.log(2.0)*np.ones((self.S,self.K),dtype='float32')
        t_m=np.ones((self.S,self.K))
        
        #=======================================================================
        # tensor
        #=======================================================================
        #zeta=np.random.random(size=(self.S,self.K,self.J))
        rzeta=1.0*np.ones((self.S,self.K,self.J))
        for s in range(self.S):
            rzeta[s,:,:]/=rzeta[s,:,:].sum(1).reshape(-1,1)
        rzeta=np.log(rzeta)
        
        
        #=======================================================================
        # vectors
        #=======================================================================
        t_c=2.0*np.ones((self.K,1),dtype='float32')
        t_f=2.0*np.ones((self.K,1),dtype='float32')
        t_tau=2.0*np.ones((self.K,1),dtype='float32')
        h_tau=2.0*np.ones((self.K,1),dtype='float32')
        
#         print((t_a/t_b)*(t_mu**2+t_l).sum(0)+(zeta*(t_e_1/t_e_2)).sum(2))
        
        #=======================================================================
        # The order is very important 
        #=======================================================================
        self.params=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
        self.local_params=[rt_s,t_m,rzeta]
        self.batch_params=[rt_b,rt_e_1,rt_e_2,rt_xi,rt_l,t_mu,rt_a]
        self.xi=xi
        self.gamma_1=2
        self.gamma_2=2
        self.a=2
        self.c=2
        self.f=2
    
    #===========================================================================
    # Update the number of features
    #TO BE IMPLEMENTED
    #===========================================================================
    def feature_update(self):
        '''
        Function for using Gibbs sampling to update the number of features.
        To be implemented 
        '''
        pass
    
    
    #===========================================================================
    # Intermediate values for the non batch global parameters
    #===========================================================================
    def global_params_VI(self):
        '''
        Function to calculate the intermediate values for the non-batch global parameters.
        As before since we cant set the gradient to zero to get the update for the parameters, 
        we need to perform gradient ascent to get the value(?)
        
        Parameters
        ----------
        
        miniBatch: ndarray
            The minibatch of the dataset in which we calculate the gradients
        
        Returns
        -------
        gradients: list
            List of ndarrays containing the intermediate values for the non batch global parameters
        '''
        
        #simple VI
        print('Performing simple VI for global parameters...')
        #qk calculation 
        q_k=np.exp(psi(self.params[2])+(np.cumsum(psi(self.params[0]),0)-psi(self.params[0]))-np.cumsum(psi(self.params[0]+self.params[2]),0))
        q_k/=q_k.sum()
        q_z=1.0/(1.0+np.exp(-self.params[1]))
        
        #mult bound
        #check this later just to be sure
        second_sum=np.zeros((K,1))
        third_sum=np.zeros((K,1))
        for k in range(self.K):
            for m in range(k-1):
                if (k==0):
                    second_sum[k,0]=0
                second_sum[k,0]+=q_k[m+1:k].sum()*psi(self.params[0][m,0])
            for m in range(k):
                third_sum+=q_k[m:k,0].sum()*psi(self.params[0][m,0]+self.params[2][m,0])
            
        mult_bound=np.cumsum(q_k*psi(self.params[2]),0)+second_sum-third_sum-np.cumsum(q_k*np.log(q_k),0)
        
        
        #tilde tau, that's tricky
        first_sum=np.zeros((K,1))
        second_sum=np.zeros((K,1))
        for k in range(self.K):
            for m in range(k+1,self.K):
                first_sum[k,0]+=(self.D-q_z[:,m].sum())*q_k[k+1:m,0].sum()
            second_sum[k,0]=(q_z[:,k:]).sum()
            
        self.params[0]=self.params[5]/self.params[6]+first_sum+second_sum
        
        #omega
        for k in range(self.K):
                self.params[1][:,k]=np.sum(psi(self.params[0][:k+1,0])-psi(self.params[0][:k+1,0]+self.params[2][:k+1,0]),0)+mult_bound[k]-0.5*np.log(2*np.pi)+0.5*(psi(self.params[3][k,0])-np.log(self.params[4][k,0]))\
                    -0.5*(self.params[3][k,0]/self.params[4][k,0])*(np.exp(self.batch_params[4][:,k])**2+np.exp(self.batch_params[5][:,k]))
        
        #hat_tau
        for k in range(self.K):
            self.params[2][k]=1.0+(D-q_z[:,k:]).sum()*q_k[k]
        
            
        #tilde c
        self.params[3]=self.c+0.5*q_z.sum(0).reshape(-1,1)
        
        #tilde_f
        self.params[4]=self.f+(np.exp(self.batch_params[4])**2+np.exp(self.batch_params[5])).sum(0).reshape(-1,1)
        
        #update t_g_1
        self.params[5]=self.gamma_1+self.K-1
        
        #update t_g_2
        self.params[6]=self.gamma_2-np.sum(psi(self.params[0][:self.K-2])-psi(self.params[0][:self.K-2]+self.params[2][self.K-2]))
        
      
    #===========================================================================
    # Get the gradients for the global batch parameters
    #===========================================================================
    def getBatchGradients(self,miniBatch,s):
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
        
        #Arrange the local parameters as if we have x_s N times
        t_s=self.local_params[0][s,:].reshape(1,-1)
        t_s=np.repeat(t_s,self.S,axis=0)
         
        t_m=self.local_params[1][s,:].reshape(1,-1)
        t_m=np.repeat(t_m,self.S,axis=0)
         
        #stack the same elements to get the N*zeta_{skj}
        zeta=self.local_params[2][s,:,:]
        zeta=zeta[newaxis,:,:]
        zeta=np.repeat(zeta,self.S,axis=0)
         
        #temporary local params for gradient calculation
        local_params=[t_s,t_m,zeta]
         
        #Get observation x_s from the dataset and repeat N times
        x=miniBatch[s,:].reshape(1,-1)
        x=np.repeat(x,self.S,axis=0)
        
        #=======================================================================
        # some gradient ascent thingy here. 
        #=======================================================================

        batch_gradients=self.batchGradientFunction(*(local_params+self.params+self.batch_params),x=x,xi=self.xi,K=self.K,D=self.D,S=self.S,J=self.J)
      
        return batch_gradients
    



    #===========================================================================
    # Gradient step for updating the parameters
    #===========================================================================
    def updateParams(self,miniBatch,batch_update):
        '''
        Update the global parameters with a gradient step.
        Basically perform the last step of the current iteration for the Stochastic Variational Inference algorithm.
        
        Parameters
        ----------
        
        miniBatch: ndarray
            The minibatch of the dataset to calculate the gradients
        
        batch_update: list
            List of ndarrays containing the intermediate values for the update of the batch global parameters
            
        '''
        
        #Ready for the final step for this iteration
        print("Updating Global Parameters...")

        #update the batch global params
        for i in range(len(self.batch_params)):
            self.batch_params[i]*=(1-rho)
            self.batch_params[i]+=(self.rho/S)*batch_update[i]
       
        
    #===========================================================================
    # Perform one iteration for the algorithm
    #===========================================================================
    def iterate(self,miniBatch):
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
        self.updateLocalParams(miniBatch)
        
        
        #============================================
        # GLOBAL PARAMETERS UPDATE                
        #============================================
        
        #for the batch global parameters get the intermediate values 
        print('Batch Parameters intermediate values calculation...')
        intermediate_values_batch_all=[0]*len(self.batch_params)
        
        #for each datapoint calculate gradient and sum over all datapoints
        for s in range(len(miniBatch)):
           batch_params=self.getBatchGradients(miniBatch,s)
           for i in range(len(intermediate_values_batch_all)):
               intermediate_values_batch_all[i]+=batch_params[i]
            
        #update the parameters
        self.updateParams(miniBatch, intermediate_values_batch_all)
        
        
    
    #===========================================================================
    # Update the local parameters with gradient steps until convergence
    #===========================================================================
    def updateLocalParams(self,miniBatch):
        '''
        Update the local parameters for the IBP-ICA model (One gradient step)
        
        Parameters
        ----------
        
        miniBatch: ndarray
            The minibatch for this run of the algorithm
        '''
        
        print("Updating local parameters...")

        gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.S,J=self.J)
        self.local_params[0]+=self.rho*gradients[0]

        gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.S,J=self.J)
        self.local_params[1]+=self.rho*gradients[1]
        
        gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.S,J=self.J)
        self.local_params[2]+=self.rho*gradients[2]

             
    #===========================================================================
    # Calculate the lower bound and create gradient functions for all parameters
    #===========================================================================
    def createGradientFunctions(self):
        '''
        The main function of the implementation. 
        Create the variational lower bound and take derivatives with respect to the parameters.
        
        In the current version the original parameters have been replaced with original=exp(new_param) and we take derivatives 
        with respect to new_param. This is the log trick and is used for imposing positivity constraints on the needed parameters.
        
        This function assigns theano.functions to the IBP_ICA object.
        
        '''
        
        print("Creating gradient functions...")
        print("\t Initializing prior parameters...")
        
        #Some constant priors
        a=T.constant(2.0,dtype='float32')
        b=T.constant(2.0,dtype='float32')
        c=T.constant(2.0,dtype='float32')
        f=T.constant(2.0,dtype='float32')
        g_1=T.constant(2.0,dtype='float32')
        g_2=T.constant(2.0,dtype='float32')
        e_1=T.constant(2.0,dtype='float32')
        e_2=T.constant(2.0,dtype='float32')

        #Create some needed scalar variables
        K=T.scalar('K',dtype='int32')
        D=T.scalar('D',dtype='int32')      
        S=T.scalar('S',dtype='int32')
        J=T.scalar('J',dtype='int32')
        xi=T.matrix('xi',dtype='float32')
        
        #The matrix for the data
        x=T.matrix('x',dtype='float32')
         
        
        #Parameters that do not need the log trick
        omega=T.fmatrix('omega')
        t_mu=T.fmatrix('t_mu')
        t_m=T.fmatrix('t_m')
         
        #need to be positive scalars, so create some y variables so that e.g., t_a=exp(t_a_y)
        t_a_y=T.scalar('t_a_y',dtype='float32')
        t_b_y=T.scalar('t_b_y',dtype='float32')
        t_g_1=T.scalar('t_g_1',dtype='float32')
        t_g_2=T.scalar('t_g_2',dtype='float32')
         
        #tilde_eta_1,tilde_eta_2,tilde_xi,tilde_l,tilde_mu,omega,tilde_s, tilde_m and zeta are matrices
        t_e_1_y,t_e_2_y,t_xi_y,t_l_y,t_s_y=T.fmatrices('t_e_1_y','t_e_2_y','t_xi_y','t_l_y','t_s_y')
         
        #the only tensor we got
        zeta_y=T.ftensor3('zeta_y')
     
        #the rest are columns
        t_c=T.col('t_c',dtype='float32')
        t_f=T.col('t_f',dtype='float32')
        t_tau=T.col('t_tau',dtype='float32')
        h_tau=T.col('h_tau',dtype='float32')
         
        #Reparametrize the original variables to be the exp of something in order to impose positivity constraint
        t_a=T.exp(t_a_y)
        t_e_1=T.exp(t_e_1_y)
        t_e_2=T.exp(t_e_2_y)
        t_l=T.exp(t_l_y)  
        t_b=T.exp(t_b_y)     
   
        zeta=T.exp(zeta_y)
        t_xi=T.exp(t_xi_y)
        t_s=T.exp(t_s_y)
        
        print('\t Creating the bound terms...')
        #Calculate q_k as seen in Chatzis et al.
        cs=T.cumsum(T.psi(t_tau),0)-T.psi(t_tau)
        q_k=T.exp(T.psi(h_tau)+cs-T.cumsum(T.psi(t_tau+h_tau),0))
        q_k/=T.sum(q_k)
       
        
    
        #I think these are correct but I also have a nested scan in the deprecated stuff just in case
        try_sum,_=theano.scan(lambda i, qk,ttau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i])*(T.sum(qk[:i])-T.cumsum(qk[:i-1]))),T.constant(0.0,dtype='float32')),
                            sequences=T.arange(K),
                            non_sequences=[q_k,t_tau],
                            strict=True
                            )
        
        try_sum2,_=theano.scan(lambda i, qk,ttau,htau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i+1]+htau[:i+1])*(T.cumsum(qk[:i])[::-1])),T.constant(0.0,dtype='float32')),
                            sequences=T.arange(K),
                            non_sequences=[q_k,t_tau,h_tau],
                            strict=True
                            )
        
    
      
        #the multinomial bound for calculating p(z_{dk}|\upsilon_k)
        mult_bound=T.cumsum(q_k*T.psi(h_tau))+try_sum-try_sum2-T.cumsum(q_k*T.log(q_k))

        #calculate q(z_{dk}=1)
        q_z=1.0/(1.0+T.exp(-omega))
        
        
        #IS THIS WRONG?
        #THIS AND FINAL SEEM TO BE THE TERMS THAT CREATE PROBLEMS WHEN UPDATING T_A AND T_B
        def inter_loop(n,xn,ts,tm,tmu,tl,ta,tb,K):
            
             ssum,_=theano.scan(fn=lambda k,curr,xn,ts,tm,tmu,tl,n: curr-2*(tm[n,k]*T.dot(tmu[:,k].T,x[n,:].T)+2*tm[n,k]*T.dot(tmu[:,k].T,T.sum(tm[n,k+1:]*tmu[:,k+1:],1))),
                               sequences=T.arange(K),   
                               outputs_info=T.constant(0.).astype('float32'),
                               non_sequences=[xn,ts,tm,tmu,tl,n],
                               )
             
             ssum=ssum[-1]
             return -0.5*(ta/tb)*(T.dot(x[n,:],x[n,:].T)+ssum+T.sum((tm[n,:]**2+ts[n,:])*(T.sum(tl,0)+T.diag(T.dot(tmu.T,tmu)))))

        final,_=theano.scan(fn=inter_loop,
                            sequences=T.arange(S),
                            non_sequences=[x,t_s,t_m,t_mu,t_l,t_a,t_b,K],
                            strict=True
                            )
        

        
        #=======================================================================
        # Calculate the expectation of log p(y)
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
                              non_sequences=[zeta,t_e_1,t_e_2,t_s,t_m],
                              strict=True
                              )
      
        #=======================================================================
        # calculate all the individual terms for calculating the likelihood
        # this way it's easier to debug 
        # The names are pretty self-explanatory 
        #=======================================================================
        
        expectation_log_p_a=g_1*T.log(g_2)+(g_1-1)*(T.psi(t_g_1)-T.log(t_g_2))-g_2*(t_g_1/t_g_2)-T.gammaln(g_1)
        
        
        expectation_log_p_phi=a*T.log(b)+(a-1)*(T.psi(t_a)-T.log(t_b))-b*(t_a/t_b)-T.gammaln(a)
        
        
        expectation_log_u_k=T.psi(t_g_1)-T.log(t_g_2)+(t_g_1/t_g_2-1)*(T.psi(t_tau)-T.psi(t_tau+h_tau))
        
        
        expectation_log_lambda_k=c*T.log(f)+(c-1)*(T.psi(t_c)-T.log(t_f))-f*(t_c/t_f)-T.gammaln(c)
        
        
        expectation_log_varpi=-T.sum(T.gammaln(xi),1)+T.gammaln(T.sum(xi,1))+T.sum((xi-1)*(T.psi(t_xi)-(T.psi(T.sum(t_xi,1))).dimshuffle(0, 'x')),1)
        
        
        expectation_log_skj=e_1*T.log(e_2)-(e_1+1)*(T.log(t_e_2)-T.psi(t_e_1))-e_2*(t_e_1/t_e_2)-T.gammaln(e_1)
        
        
        expectation_log_gdk=-0.5*K*T.log(2*np.pi)+T.sum(0.5*(T.psi(t_c)-T.log(t_f)))-T.sum(0.5*(t_c/t_f).T*(t_mu**2+t_l),1)
        
        
        
        expectation_log_zdk=q_z*T.cumsum(T.psi(h_tau)-T.psi(t_tau+h_tau))+(1.0-q_z)*mult_bound
        
        
        expectation_log_varepsilon=zeta*(T.psi(t_xi)-T.psi(T.sum(t_xi,1)).dimshuffle(0,'x'))
        
        
        expectation_log_y=final_y
        
        
        expectation_log_x=-0.5*D*T.log(2*np.pi)+0.5*D*(T.psi(t_a)-T.log(t_b))+final
         
      
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
        # Maybe should do individual lines again for easier debug.
        # Checked many times but check again
        # Each line is a different entropy term                   
        #=======================================================================
        entropy=t_g_1-T.log(t_g_2)+(1-t_g_1)*T.psi(t_g_1)+T.gammaln(t_g_1) \
                +t_a-T.log(t_b)+(1-t_a)*T.psi(t_a)+T.gammaln(t_a) \
                +T.sum(T.gammaln(t_tau)+T.gammaln(h_tau)-T.gammaln(h_tau+t_tau)-(t_tau-1)*T.psi(t_tau)-(h_tau-1)*T.psi(h_tau)+(t_tau+h_tau-2)*T.psi(t_tau+h_tau)) \
                +T.sum(t_c-T.log(t_f)+T.gammaln(t_c)+(1-t_c)*T.psi(t_c)) \
                +T.sum(T.sum(T.gammaln(t_xi),1)-T.gammaln(T.sum(t_xi,1))-(J-T.sum(t_xi,1))*T.psi(T.sum(t_xi,1))-(T.sum((t_xi-1.0)*T.psi(t_xi),1))) \
                +T.sum(t_e_1+T.log(t_e_2)+T.gammaln(t_e_1)-(1+t_e_1)*T.psi(t_e_1)) \
                +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.sum(T.log(t_l),1)) \
                -T.sum((1.0-q_z)*T.log(1.0-q_z))-T.sum(q_z*T.log(q_z)) \
                +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.sum(T.log(t_s),1)) \
                -T.sum(zeta*T.log(zeta))

        #The evidence lower bound is the likelihood plus the entropy
        lower_bound=likelihood+entropy
        
        
        #=======================================================================
        # set local and global gradient variables
        #=======================================================================
        #to take derivatives wrt to the log
        #Reminder: e.g., t_g_1=T.exp(t_g_1_y)
        print('\t Calculating Derivatives of the lower bound...')
        simpleVariationalVariables=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
        localVar=[t_s_y,t_m,zeta_y]
        batch_grad_vars=[t_b_y,t_e_1_y,t_e_2_y,t_xi_y,t_l_y,t_mu,t_a_y]
        
        
        #=======================================================================
        # calculate the derivatives
        #=======================================================================
        derivatives_local=T.grad(lower_bound,localVar)
        derivatives_batch=T.grad(lower_bound,batch_grad_vars)
        
        print('\t Creating functions...')
        self.localGradientFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,J,D,S],derivatives_local,allow_input_downcast=True,on_unused_input='ignore')
        self.batchGradientFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,J,D,S],derivatives_batch,allow_input_downcast=True,on_unused_input='ignore')
        self.lowerBoundFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,D,J,S],lower_bound,allow_input_downcast=True,on_unused_input='ignore')
    
    
    
    #===========================================================================
    # Save the parameters to examine after experiments
    #===========================================================================
    def save_params(self,iteration,LL):
        with open('params/params'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(self.params, f)
        with open('params/localparams'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(self.local_params, f)
        with open('params/batchparams'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(self.batch_params, f)
        with open('params/bound_iter_'+str(iteration)+'.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(LL, f)
                
                
    #===========================================================================
    # Some naive creation of synthetic data
    #===========================================================================
    def create_synthetic_data(self,N):
        G=np.random.normal(0,1,size=(self.D,self.K))
        y=np.random.normal(0,1,size=(self.K,N))
        return np.dot(G,y).T,G,y
    
#===============================================================================
# Main 
#===============================================================================
    
if __name__ == '__main__':
    
    #some initial variables
    K=5
    N=100
    D=4
    J=8
    S=50
    rho=0
    
    #initialize IBP_ICA object
    z=IBP_ICA(K,D,J,S,N,rho)
    
    #create some synthetic data
    x,G,y=z.create_synthetic_data(N)
    
    #init the posterior parameters
    z.init_params()
    
    #create lower bound and get gradients
    z.createGradientFunctions()
    
   
    iteration=0
    max_iter=10
    
    #keep the lower bound for each iteration
    LL=np.zeros((max_iter,int(N/S)))
    
    #repeat until maxi iterations
    while iteration<max_iter:
        
        iteration+=1
        print("Stochastic IBP-ICA iteration: ",iteration)
        
        #set step size for this iteration (Paisley et al.) 
        z.rho=(iteration+100.0)**(-.6)
        
        #create all the random minibatches for this iteration
        random_indices=np.arange(N)
        np.random.shuffle(random_indices)
        random_indices=random_indices.reshape((int(N/S),S))
        
        #FOR ALL THE MINIBATCHES update the local parameters and the global parameters with SGD
        current_minibatch=0
        
        
        
        for miniBatch_indices in random_indices:
            print('\n')
            print('#########################################')
            print('# Processing miniBatch',current_minibatch+1,'at iteration',iteration,'#')
            print('#########################################')
            current_minibatch+=1
            miniBatch=x[miniBatch_indices,:]
            
            
            #print the lower bound before updating the parameters
            lower_bound=(z.lowerBoundFunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.S,J=z.J))
            
            print('-----------------------------------------------------------------------------------------------------')
            print('The lower bound with the new batch before optimization is',lower_bound)
            print('-----------------------------------------------------------------------------------------------------')

            #perform one iteration of the algorithm 
            z.iterate(miniBatch)
            
            #append lower bound to list
            LL[iteration-1,current_minibatch-1]=(z.lowerBoundFunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.S,J=z.J))
            
            #print the lower bound to see how it goes
            print('------------------------------------------------------------------')
            print("Lower Bound at iteration",iteration," and minibatch ",current_minibatch," is ",LL[iteration-1,current_minibatch-1])
            print('------------------------------------------------------------------')
            
            if (np.isnan(LL[iteration-1,current_minibatch-1])):
                sys.exit('Why is the bound nan? Please Debug.')
            
        #after all the minibatches for this iteration, update the global params with simple VI
        z.global_params_VI()
 
        print(LL[iteration-1,:])
        z.save_params(iteration,LL[iteration-1,:])

    print(LL)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#===============================================================================
# DEPRECATED STUFF. KEEP HERE JUST IN CASE
#===============================================================================


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