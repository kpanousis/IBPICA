#Dear Diary



'''

Created on Jun 29, 2016

@author: kon
'''
import os    
os.environ['THEANO_FLAGS'] = "device=cpu,compute_test_value=ignore"
import theano
import theano.tensor as T
import numpy as np
import time
import pickle
import sys
from scipy import signal
from numpy.linalg import norm
from numpy import newaxis
from theano.ifelse import ifelse
from theano import pp
from theano.gradient import consider_constant
from scipy.special import psi

#from theano import config,sandbox
theano.config.compute_test_value = 'ignore' # Use 'warn' to activate this feature
from theano.compile.nanguardmode import NanGuardMode
theano.config.exception_verbosity='high'
theano.config.optimizer='fast_run'
theano.config.traceback.limit=20
from six.moves import cPickle
from scipy.special import expit
from sklearn import preprocessing
print(theano.config.device)


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
        xi=2*np.ones((self.K,self.J))
        
        #=======================================================================
        # The scalar parameters
        #=======================================================================
        t_a=10.0
        t_b=10.0
        t_g_1=2.0
        t_g_2=2.0
        
        #=======================================================================
        # matrices
        #=======================================================================
        t_e_1=2*np.ones((self.K,self.J),dtype='float32')
        t_e_2=2*np.ones((self.K,self.J),dtype='float32')
        t_xi=5*np.ones((self.K,self.J))
        t_l=np.ones((self.D,self.K),dtype='float32')
        t_mu=np.random.normal(0,1,(self.D,self.K))
        omega=np.random.random((self.D,self.K))
        t_s=np.ones((self.N,self.K),dtype='float32')
        t_m=np.random.normal(0,1,(self.N,self.K))

        #=======================================================================
        # tensor
        #=======================================================================
        #zeta=np.random.random(size=(self.S,self.K,self.J))
        zeta=1.0*np.random.random((self.N,self.K,self.J))
        for s in range(self.N):
            zeta[s,:,:]/=zeta[s,:,:].sum(1).reshape(-1,1)
        
        
        #=======================================================================
        # vectors
        #=======================================================================
        t_c=2.0*np.ones((self.K,1),dtype='float32')
        t_f=2.0*np.ones((self.K,1),dtype='float32')
        t_tau=2.0*np.ones((self.K,1),dtype='float32')
        h_tau=2.0*np.ones((self.K,1),dtype='float32')
        
#         print((t_a/t_b)*(t_mu**2+t_l).sum(0)+(zeta*(t_e_1/t_e_2)).sum(2))
        
        data=preprocessing.scale(data,with_mean=True,with_std=True)

        #=======================================================================
        # The order is very important 
        #=======================================================================
        self.params=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
        self.local_params=[t_s,t_m,zeta]
        self.batch_params=[t_b,t_e_1,t_e_2,t_xi,t_l,t_mu,t_a]
        self.xi=xi
        self.gamma_1=2
        self.gamma_2=2
        self.a=2
        self.b=2
        self.c=2
        self.eta_1=2
        self.eta_2=2
        self.gamma_1=2
        self.gamma_2=2
        self.f=2
        self.l1=2
    
        return data
    
    def mult_bound_calc(self):
        #qk calculation 
        q_k=np.exp(psi(self.params[2])+(np.cumsum(psi(self.params[0]),0)-psi(self.params[0]))-np.cumsum(psi(self.params[0]+self.params[2]),0))
        q_k/=q_k.sum()
        
        #mult bound
        #check this later just to be sure
        second_sum=np.zeros((self.K,1))
        third_sum=np.zeros((self.K,1))
        
        for k in range(self.K):
            for m in range(k):
                temp=q_k[m+1:k+1].sum()
                second_sum[k,0]+=temp*psi(self.params[0][m,0])
            for m in range(k+1):
                temp=q_k[m:k+1].sum()
                third_sum[k,0]+=temp*psi(self.params[0][m,0]+self.params[2][m,0])
             
        mult_bound=np.cumsum(q_k*psi(self.params[2]),0)+second_sum-third_sum-np.cumsum(q_k*np.log(q_k),0)

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
        q_z=expit(self.params[1])
        
        mult_bound,q_k=self.mult_bound_calc()
       
        #tilde tau, that's tricky
        first_sum=np.zeros((self.K,1))
        second_sum=np.zeros((self.K,1))
        for k in range(self.K):
            for m in range(k+1,self.K):
                first_sum[k,0]+=(self.D-q_z[:,m].sum(0))*(q_k[k+1:m+1,0].sum())
            second_sum[k,0]=(q_z[:,k:]).sum()
             
        #here tilde tau
        self.params[0]*=(1.0-self.rho)
        self.params[0]+=self.rho*(self.params[5]/self.params[6]+first_sum+second_sum)
         
        #omega
        self.params[1]*=(1.0-self.rho)
        for k in range(self.K):
            self.params[1][:,k]+=self.rho*((psi(self.params[0][:k+1])-psi(self.params[0][:k+1]+self.params[2][:k+1])).sum()-mult_bound[k]+0.5*(psi(self.params[3][k])-np.log(self.params[4][k]))\
                    -0.5*(self.params[3][k]/self.params[4][k])*(self.batch_params[5][:,k]**2+self.batch_params[4][:,k]))
                    
         
        #hat_tau
        self.params[2]*=(1.0-self.rho)
        for k in range(self.K):
            self.params[2][k]+=self.rho*(1.0+(self.D-q_z[:,k:].sum(0)).sum()*q_k[k])
         
             
        #tilde c
        self.params[3]*=(1.0-self.rho)
        self.params[3][:,0]+=self.rho*(self.c+0.5*q_z.sum(0).T)
         
        #tilde_f
        self.params[4]*=(1.0-self.rho)
        for k in range(self.K):
            self.params[4][k,0]+=self.rho*(self.f+0.5*(self.batch_params[4][:,k].sum()+np.dot(self.batch_params[5][:,k].T,self.batch_params[5][:,k])))
         
        #update t_g_1
        self.params[5]*=(1.0-self.rho)
        self.params[5]+=self.rho*(self.gamma_1+self.K-1)
         
        #update t_g_2
        self.params[6]*=(1.0-self.rho)
        self.params[6]+=self.rho*(self.gamma_2-(psi(self.params[0][:self.K-1])-psi(self.params[0][:self.K-1]+self.params[2][self.K-1])).sum())
        

       
      
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
        
        #Arrange the local parameters as if we have x_s N times
        t_s=self.local_params[0][s,:].reshape(1,-1)       
        t_m=self.local_params[1][s,:].reshape(1,-1)

        
        #stack the same elements to get the N*zeta_{skj}
        zeta=self.local_params[2][s,:,:]
        zeta=zeta[newaxis,:,:]

        #temporary local params for gradient calculation
        local_params=[t_s,t_m,zeta]
         
        #Get observation x_s from the dataset and repeat N times
        x=x[s,:].reshape(1,-1)

        #=======================================================================
        # some gradient ascent thingy here. 
        #=======================================================================

        batch_gradients=self.batchGradientFunction_without(*(self.params+self.batch_params),xi=self.xi,K=self.K,J=self.J)
        dependent=self.batchGradientFunction_dependent(*(local_params+self.params+self.batch_params),x=x,xi=self.xi,K=self.K,D=self.D,S=1,J=self.J)
        

        for i in range(len(self.batch_params)):
            batch_gradients[i]+=self.N*dependent[i]
            
            
        return batch_gradients
    



    #===========================================================================
    # Gradient step for updating the parameters
    #===========================================================================
    def updateParams(self,batch_update):
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
        
        #update the batch global params
        for i in range(len(self.batch_params)):
            #self.batch_params[i]*=(1-self.rho)
            if (i==5):
             self.batch_params[i]+=(self.rho/self.S)*batch_update[i]
             continue
            self.batch_params[i]=self.batch_params[i]*np.exp((self.rho/self.S)*self.batch_params[i]*batch_update[i])
       
        
    #===========================================================================
    # Perform one iteration for the algorithm
    #===========================================================================
    def iterate(self,x,mb_indices):
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
        self.updateLocalParams(x,mb_indices)
        #print(self.local_params[0][mb_indices,:])
        #print(self.local_params[1][mb_indices,:])
        #time.sleep(10)
        #============================================
        # GLOBAL PARAMETERS UPDATE                
        #============================================
        
        #for the batch global parameters get the intermediate values 
        intermediate_values_batch_all=[0]*len(self.batch_params)
        
        #self.l1+=self.rho*self.lagrange1(*(self.local_params+self.params+self.batch_params),x=x,xi=self.xi,K=self.K,D=self.D,S=1,J=self.J)
        
        #for each datapoint calculate gradient and sum over all datapoints
        for s in range(self.S):
           batch_params=self.getBatchGradients(x,mb_indices[s])
           
           for i in range(len(intermediate_values_batch_all)):
               intermediate_values_batch_all[i]+=batch_params[i]
            
        
        #update the parameters
        self.updateParams(intermediate_values_batch_all)
        
        
    
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
        miniBatch=x[mb_indices,:]
        for i in range(10):
            local_params=[self.local_params[0][mb_indices,:],self.local_params[1][mb_indices,:],self.local_params[2][mb_indices,:,:]]
            gradients=self.localGradientFunction(*(local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.S,J=self.J)
            self.local_params[0][mb_indices,:]=self.local_params[0][mb_indices,:]*np.exp(self.rho*self.local_params[0][mb_indices,:]*gradients[0])
            
            local_params=[self.local_params[0][mb_indices,:],self.local_params[1][mb_indices,:],self.local_params[2][mb_indices,:,:]]
            gradients=self.localGradientFunction(*(local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.S,J=self.J)
            self.local_params[1][mb_indices,:]+=self.rho*gradients[1]
            
            local_params=[self.local_params[0][mb_indices,:],self.local_params[1][mb_indices,:],self.local_params[2][mb_indices,:,:]]
            gradients=self.localGradientFunction(*(local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.S,J=self.J)
    
            self.local_params[2][mb_indices,:,:]=self.local_params[2][mb_indices,:,:]*np.exp(self.rho*self.local_params[2][mb_indices,:,:]*gradients[2])
            temp=np.repeat(self.local_params[2][mb_indices,:,:].sum(2)[:,:,newaxis],self.J,2)
            self.local_params[2][mb_indices,:,:]/=temp
        #print(self.local_params[2][mb_indices])
        #time.sleep(10)

             
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
        a=T.constant(10.0,dtype='float32')
        b=T.constant(5.0,dtype='float32')
        c=T.constant(10.0,dtype='float32')
        f=T.constant(5.0,dtype='float32')
        g_1=T.constant(10.0,dtype='float32')
        g_2=T.constant(5.0,dtype='float32')
        e_1=T.constant(10.0,dtype='float32')
        e_2=T.constant(5.0,dtype='float32')

        #Lagrange Multipliers
        #l1=T.fscalar('l1')
        
        #Create some needed scalar variables
        K=T.scalar('K',dtype='int32')
        D=T.scalar('D',dtype='int32')      
        S=T.scalar('S',dtype='int32')
        J=T.scalar('J',dtype='int32')
        xi=T.matrix('xi',dtype='float32')
        
        #The matrix for the data
        x=T.fmatrix('x')
         
        
        #Parameters that do not need the log trick
        omega=T.fmatrix('omega')
        t_mu=T.fmatrix('t_mu')
        t_m=T.fmatrix('t_m')
         
        #need to be positive scalars, so create some y variables so that e.g., t_a=exp(t_a_y)
        t_a=T.scalar('t_a',dtype='float32')
        t_b=T.scalar('t_b',dtype='float32')
        t_g_1=T.scalar('t_g_1',dtype='float32')
        t_g_2=T.scalar('t_g_2',dtype='float32')
         
        #tilde_eta_1,tilde_eta_2,tilde_xi,tilde_l,tilde_mu,omega,tilde_s, tilde_m and zeta are matrices
        t_e_1,t_e_2,t_xi,t_l,t_s=T.fmatrices('t_e_1','t_e_2','t_xi','t_l','t_s')
         
        #the only tensor we got
        zeta=T.ftensor3('zeta')
     
        #the rest are columns
        t_c=T.col('t_c',dtype='float32')
        t_f=T.col('t_f',dtype='float32')
        t_tau=T.col('t_tau',dtype='float32')
        h_tau=T.col('h_tau',dtype='float32')
         
        
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
        q_z=T.nnet.sigmoid(omega)
        
        
        #IS THIS WRONG?
        #THIS AND FINAL SEEM TO BE THE TERMS THAT CREATE PROBLEMS WHEN UPDATING T_A AND T_B

        def deep_loop(k,prev_value,n,xn,ts,tm,tmu,tl):      
            temp=0
            temp+=2*(tm[n,k]*tm[n,k+1:]*T.dot(tmu[:,k].T,tmu[:,k+1:])).sum()
            temp-=2*tm[n,k]*T.dot(tmu[:,k].T,x[n,:].T)
            temp+=(T.dot(tmu[:,k].T,tmu[:,k])+tl[:,k].sum())*(tm[n,k]**2+ts[n,k])
            return prev_value+temp
        
        def inter_loop(s,xn,ts,tm,tmu,tl,ta,tb):
            
            temp,_=theano.scan(fn=deep_loop,
                              sequences=T.arange(K),
                              outputs_info=T.as_tensor_variable(np.asarray(0, x.dtype)),
                              non_sequences=[s,xn,ts,tm,tmu,tl],
                              strict=True
                              )
            return (-0.5*(ta/tb)*((T.dot(xn[s,:],xn[s,:].T))+temp[-1]))

        
        final,_=theano.scan(fn=inter_loop,
                            sequences=T.arange(S),
                            non_sequences=[x,t_s,t_m,t_mu,t_l,t_a,t_b],
                            strict=True
                            )
        
        #=======================================================================
        # calculate all the individual terms for calculating the likelihood
        # this way it's easier to debug 
        # The names are pretty self-explanatory 
        #=======================================================================
        
        expectation_log_p_a=(g_1-1)*(T.psi(t_g_1)-T.log(t_g_2))-g_2*(t_g_1/t_g_2)
        
        
        expectation_log_p_phi=(a-1)*(T.psi(t_a)-T.log(t_b))-b*(t_a/t_b)
        
        
        expectation_log_u_k=T.psi(t_g_1)-T.log(t_g_2)+(t_g_1/t_g_2-1)*(T.psi(t_tau)-T.psi(t_tau+h_tau))
        
        
        expectation_log_lambda_k=(c-1)*(T.psi(t_c)-T.log(t_f))-f*(t_c/t_f)
        
        
        expectation_log_varpi=((xi-1)*(T.psi(t_xi)-T.psi(T.sum(t_xi,1)).dimshuffle(0,'x'))).sum(1)
        
        
        expectation_log_skj=(e_1-1)*(T.psi(t_e_1)-T.log(t_e_2))-e_2*(t_e_1/t_e_2)
        
        
        expectation_log_gdk=+0.5*(T.psi(t_c)-T.log(t_f)).reshape((1,-1))-0.5*((t_c/t_f).T*(t_mu**2+t_l))
        
        
        expectation_log_zdk=q_z*T.cumsum(T.psi(h_tau)-T.psi(t_tau+h_tau))+(1.0-q_z)*mult_bound.T
        
        
        expectation_log_varepsilon=zeta*(T.psi(t_xi)-T.psi(t_xi.sum(1)).dimshuffle(0,'x'))
        
        
        expectation_log_y=0.5*T.sum(zeta*(T.psi(t_e_1)-T.log(t_e_2)),2)-.5*T.sum(zeta*(t_e_1/t_e_2)*(t_s+t_m**2)[:,:,newaxis],2)
        
        
        expectation_log_x=0.5*D*(T.psi(t_a)-T.log(t_b))+final
        

        
      
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
                    +T.sum(expectation_log_zdk)
                                          
        #=======================================================================
        # calculate the entropy term  
        # Maybe should do individual lines again for easier debug.
        # Checked many times but check again
        # Each line is a different entropy term                   
        #=======================================================================
        entropy_q_a=t_g_1-T.log(t_g_2)+(1.0-t_g_1)*T.psi(t_g_1)+T.gammaln(t_g_1)
        
        entropy_q_phi=t_a-T.log(t_b)+(1.0-t_a)*T.psi(t_a)+T.gammaln(t_a)
        
        entropy_q_uk=T.sum(T.gammaln(t_tau)+T.gammaln(h_tau)-T.gammaln(h_tau+t_tau)-(t_tau-1.0)*T.psi(t_tau)-(h_tau-1.0)*T.psi(h_tau)+(t_tau+h_tau-2)*T.psi(t_tau+h_tau))
        
        entropy_q_lambda_k=T.sum(t_c-T.log(t_f)+T.gammaln(t_c)+(1.0-t_c)*T.psi(t_c))
        
        entropy_q_varpi=T.sum(T.gammaln(t_xi).sum(1)-T.gammaln(t_xi.sum(1))-(J-t_xi.sum(1))*T.psi(t_xi.sum(1))-((t_xi-1.0)*T.psi(t_xi)).sum(1))
        
        entropy_q_s=T.sum(t_e_1-T.log(t_e_2)+T.gammaln(t_e_1)+(1.0-t_e_1)*T.psi(t_e_1))
        
        entropy_q_z=T.sum(-(1.0-q_z)*T.log(1.0-q_z)-q_z*T.log(q_z))
        
        entropy=entropy_q_a \
                +entropy_q_phi \
                + entropy_q_uk\
                +entropy_q_lambda_k \
                +entropy_q_varpi \
                +entropy_q_s \
                +entropy_q_z
                 

        terms_dependent_on_N=T.sum(expectation_log_varepsilon)+T.sum(expectation_log_x)+T.sum(expectation_log_y)+0.5*T.sum(T.log(t_l))\
                      +0.5*T.sum(T.log(t_s))-T.sum(zeta*T.log(zeta))

        #The evidence lower bound is the likelihood plus the entropy
        lower_bound_without_N=likelihood+entropy
        lower_bound_with_N=lower_bound_without_N+terms_dependent_on_N
        
        #=======================================================================
        # set local and global gradient variables
        #=======================================================================
        #to take derivatives wrt to the log
        #Reminder: e.g., t_g_1=T.exp(t_g_1_y)
        print('\t Calculating Derivatives of the lower bound...')
        simpleVariationalVariables=[t_tau,omega,h_tau,t_c,t_f,t_g_1,t_g_2]
        localVar=[t_s,t_m,zeta]
        batch_grad_vars=[t_b,t_e_1,t_e_2,t_xi,t_l,t_mu,t_a]
        
        
        #=======================================================================
        # calculate the derivatives
        #=======================================================================
        derivatives_local=T.grad(lower_bound_with_N,localVar)
        derivatives_batch_without=T.grad(lower_bound_without_N,batch_grad_vars)
        derivatives_batch_dependent=T.grad(terms_dependent_on_N,batch_grad_vars)
        derivatives_batch_all=T.grad(lower_bound_with_N,batch_grad_vars)
        #derivatives_l1=T.grad(lower_bound_with_N,lower_bound_with_N)
        
        print('\t Creating functions...')
        #self.check_function=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,J,D,S],final,allow_input_downcast=True,on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
        self.localGradientFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,J,D,S],derivatives_local,allow_input_downcast=True,on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
        self.batchGradientFunction_without=theano.function(simpleVariationalVariables+batch_grad_vars+[xi,K,J],derivatives_batch_without,allow_input_downcast=True,on_unused_input='warn',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))

        
        #for the dependent on data driven sufficient stats
        self.batchGradientFunction_dependent=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,J,D,S],derivatives_batch_dependent,allow_input_downcast=True,on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
        self.lowerBoundFunction=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,D,J,S],lower_bound_without_N,allow_input_downcast=True,on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
    
        #self.lagrange1=theano.function(localVar+simpleVariationalVariables+batch_grad_vars+[xi,x,K,D,J,S,l1],derivatives_l1,allow_input_downcast=True)
    
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
                
                
    #===========================================================================
    # Some naive creation of synthetic data
    #===========================================================================
    def create_synthetic_data(self):
         #x=sio.loadmat(dataset)
        #x=x["X"]
        #_,x=wavfile.read(dataset)
        n_samples = 2000
        time = np.linspace(0, 8, n_samples)

        s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
        s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
        s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

        S = np.c_[s1, s2, s3]
        S += 0.2 * np.random.normal(size=S.shape)  # Add noise
        S /= S.std(axis=0)  # Standardize data
        # Mix data
        A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
        X = np.dot(S, A.T)  # Generate observations
        #x=x.reshape(180,3,3000)[:,0,:]
        #ind=np.arange(500,1000)
        #ind2=np.arange(3000,4500)
        #al=np.append(ind,ind2)
        #ind=np.arange(500,1000,1)
        #ind2=np.arange(3000,4500,1)
        #ind_all=list(ind)
        #ind_all.extend(list(ind2))
        #x=x[:,ind_all]
        return X,0,0
    
#===============================================================================
# Main 
#===============================================================================
    
if __name__ == '__main__':
    
    #some initial variables
    K_init=5
    initN=100
    initD=4
    initJ=3
    initS=50
    
    #initialize IBP_ICA object
    z=IBP_ICA(K_init,initD,initJ,initS,initN)
    
    dataset='test_theano'
    data='test_theano'
    
    #create some synthetic data
    x,_,_=z.create_synthetic_data()
    
    z.N,z.D=x.shape

    #init the posterior parameters
    z.init_params(x)
    
    #create lower bound and get gradients
    z.createGradientFunctions()
    
   
    #init the posterior parameters
    x=z.init_params(x)
    print('N:',z.N,'D:',z.D)
   
    iteration=0
    max_iter=1000
    elbo_tolerance=10**-3
    
    #keep the lower bound for each iteration
    LL=np.empty((max_iter,int(z.N/z.S)))
    LL[:]=np.NaN
    print(LL.shape)
    
    start_time=str(iteration)+(time.strftime("%Y-%m-%d-%H:%M").replace(" ","_")+'_batch_size_'+str(initS)+'_D_'+str(z.D)+'_data_'+data)
    global_min=0
    #repeat until maxi iterations
    st=time.time()
    
    #repeat until maxi iterations
    while iteration<max_iter:
        
        
        if (iteration % 5 ==0):
            print('ELBO at iteration',iteration,":",LL[iteration-1,-1])
        
        iteration+=1

        #set step size for this iteration (Paisley et al.) 
        z.rho=(iteration+1000.0)**(-.6)
        
        #create all the random minibatches for this iteration
        random_indices=np.arange(z.N)
        np.random.shuffle(random_indices)
        random_indices=random_indices.reshape((int(z.N/z.S),z.S))
        
        #FOR ALL THE MINIBATCHES update the local parameters and the global parameters with SGD
        current_minibatch=0
        
        
        for miniBatch_indices in random_indices:
        
            current_minibatch+=1
            miniBatch=x[miniBatch_indices,:]            
            
            
#             final=z.check_function(*(local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.S,J=z.J)
#             print(np.allclose(final,check))
#             
            try:
                z.iterate(x,miniBatch_indices)
                
                #after all the minibatches for this iteration, update the global params with simple VI
                z.global_params_VI()
                
                #perform one iteration of the algorithm
                local_params=[z.local_params[0][miniBatch_indices,:],z.local_params[1][miniBatch_indices,:],z.local_params[2][miniBatch_indices,:,:]]
                
                #append lower bound to list
                LL[iteration-1,current_minibatch-1]=(z.lowerBoundFunction(*(local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.S,J=z.J))
                
                #check convergence
                if (current_minibatch>1):
                    if abs(LL[iteration-1,current_minibatch-1]-LL[iteration-1,current_minibatch-2])<elbo_tolerance:
                        print("Reached ELBO tolerance level..")
                        iteration=max_iter
                        break
                    
            except Exception:
                print("Unexpected error:", sys.exc_info())
                print(z.local_params[2][miniBatch_indices,:])
                print(z.batch_params)
                print(z.params)
                z.save_params('error', start_time, None)
                sys.exit('Why is the bound nan? Please Debug.')
                
            #print the lower bound to see how it goes
            #print('------------------------------------------------------------------')
            #print("Lower Bound at iteration",iteration," and minibatch ",current_minibatch," is ",LL[iteration-1,current_minibatch-1])
            #print('------------------------------------------------------------------')
            #print(z.local_params)
            if (np.isnan(LL[iteration-1,current_minibatch-1])):
                sys.exit('Why is the bound nan? Please Debug.')
        
        
        z.save_params(iteration,start_time,LL[iteration-1,:])
    
    global_min=np.nanmax(LL)
    global_min_ind=np.nanargmax(LL)
    print('------------------------------------------------------------------')
    print('The global max is ',global_min,'found in iteration',global_min_ind)
    print('------------------------------------------------------------------')
    print('Total running time:',time.time()-st)
    z.save_params("final",start_time,LL)

    print(LL)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#===============================================================================
# DEPRECATED STUFF. KEEP HERE JUST IN CASE
#===============================================================================
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
