'''
Created on Jun 29, 2016

@author: kon
'''
import theano
import theano.tensor as T
import numpy as np
import time
from numpy.linalg import norm
from numpy import newaxis
from theano.ifelse import ifelse
from theano import pp
from theano.compile.io import Out
from theano.gradient import consider_constant
theano.config.exception_verbosity='high'
theano.config.optimizer='fast_run'
theano.config.traceback.limit=20
from six.moves import cPickle

class IBP_ICA:
    
    def __init__(self,K,D,J, batch_size,lower_bound):
        self.K=K
        self.D=D
        self.J=J
        self.batch_size=batch_size
        self.lower_bound=lower_bound
        self.likelihood=0
    
    
    #===========================================================================
    # Initialize posterior parameters and xi 
    #===========================================================================
    def init_params(self):
        '''
        Initialize the parameters to pass to the model
        '''
        xi=np.ones((self.K,self.J))
        
        #scalars
        t_a=2.0
        t_b=2.0
        t_g_1=2.0
        t_g_2=2.0
        
        #matrices
        t_e_1=4*np.ones((self.K,self.J))
        t_e_2=4*np.ones((self.K,self.J))
        t_xi=2*np.ones((self.K,self.J))
        #t_l=np.random.gamma(1,1,size=(self.D,self.K))
        #t_mu=np.random.gamma(1,1,size=(self.D,self.K))
        t_l=2*np.ones((self.D,self.K))
        t_mu=np.ones((self.D,self.K))

        omega=np.ones((self.D,self.K))
        
        #more sophisticated initialization
        #t_s=np.random.gamma(1,1,size=(self.batch_size,self.K))
        t_s=2*np.ones((self.batch_size,self.K))
        #here to run PCA
        #t_m=np.random.gamma(1,1,size=(self.batch_size,self.K))
        t_m=np.ones((self.batch_size,self.K))
        #tensor
        #zeta=np.random.random(size=(self.batch_size,self.K,self.J))
        zeta=2*np.ones((self.batch_size,self.K,self.J))
        for s in range(self.batch_size):
            zeta[s,:,:]/=zeta[s,:,:].sum(1).reshape(-1,1)
        
        #tcolumns
        t_c=2*np.ones((self.K,1))
        t_f=2*np.ones((self.K,1))
        t_tau=2*np.ones((self.K,1))
        h_tau=2*np.ones((self.K,1))
        
#         print((t_a/t_b)*(t_mu**2+t_l).sum(0)+(zeta*(t_e_1/t_e_2)).sum(2))
        
        #the first three are the local params
        self.params=[t_tau,omega,h_tau,t_a,t_c,t_f,t_g_1,t_g_2]
        self.local_params=[t_s,t_m,zeta]
        self.batch_params=[t_e_1,t_e_2,t_xi,t_l,t_mu,t_b]
        self.xi=xi
    
    
    #===========================================================================
    # Update the number of features
    #TO BE IMPLEMENTED
    #===========================================================================
    def feature_update(self):
        '''
        Function for using Gibbs sampling to update the number of features.
        To be implemented today or tomorrow (13-14/07)
        '''
        pass
    
    #===========================================================================
    # Get the gradients
    #===========================================================================
    def getGradients(self,miniBatch):
        '''
        Get the gradients for all the parameters of the model
        '''
        print('Getting the gradients...')
        total_gradients=[0]*len(self.params)
        gradients=self.gradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.batch_size)
        
        return gradients
    
    #===========================================================================
    # Get the batch gradients
    #===========================================================================
    def getBatchGradients(self,miniBatch,s):
        
        t_s=self.local_params[0][s,:].reshape(1,-1)
        t_s=np.repeat(t_s,len(miniBatch),axis=0)
        
        t_m=self.local_params[1][s,:].reshape(1,-1)
        t_m=np.repeat(t_m,len(miniBatch),axis=0)
        
        #stack the same elements to get the N*zeta_{skj}
        zeta=self.local_params[2][s,:,:]
        zeta=zeta[newaxis,:,:]
        zeta=np.repeat(zeta, len(miniBatch),axis=0)
        
        
        #get only the elements associated with the sth observation
        local_params=[t_s,t_m,zeta]
        x=miniBatch[s,:].reshape(1,-1)
        x=np.repeat(x,len(miniBatch),axis=0)
        batch_gradients=self.batchGradientFunction(*(local_params+self.params+self.batch_params),x=x,xi=self.xi,K=self.K,D=self.D,S=self.batch_size)
        
        return batch_gradients
        
    #===========================================================================
    # Gradient step for updating the parameters
    #===========================================================================
    def updateParams(self,gradients,batch_gradients,current_batch_size,rho):
        '''
        Update the global parameters with a gradient step
        '''
        print("Updating Global Parameters...")
        for i in range(len(self.params)):
            self.params[i]*=(1.0-rho)
            self.params[i]+=rho*gradients[i]
        for i in range(len(self.batch_params)):
            self.batch_params[i]*=(1.0-rho)
            self.batch_params[i]+=(rho/current_batch_size)*(batch_gradients[i])
        print(self.params)
        
    #===========================================================================
    # Perform one iteration for the algorithm
    #===========================================================================
    def iterate(self,miniBatch,rho):
        '''
        Function that represents one iteration of the SVI IBP ICA algorithm
        '''
        self.updateLocalParams(miniBatch)
        gradients=self.getGradients(miniBatch)
        batch_gradients=[0]*len(self.batch_params)
        print('Batch gradient calculation...')
        for s in range(len(miniBatch)):
            batch_gradients+=self.getBatchGradients(miniBatch, s)
        self.updateParams(gradients, batch_gradients, len(miniBatch), rho)
    
    #===========================================================================
    # Calculate and return the lower bound
    #===========================================================================
    def getLowerBound(self,data):
        lower_bound=self.lowerBoundFunction(*(self.local_params+self.params+self.batch_params),xi=self.xi,K=self.K,x=data,S=self.batch_size)    
        return lower_bound
    
    
    #===========================================================================
    # Get the entropy and the likelihood as separate terms for debugging
    #===========================================================================
    def getLowerBoundterms(self,data):
        likelihood=self.likelihoodFunction(*(self.local_params+self.params+self.batch_params),xi=self.xi,K=self.K,x=data,D=self.D,S=self.batch_size)
        entropy=self.entropyFunction(*(self.local_params+self.params+self.batch_params),K=self.K,D=self.D,S=self.batch_size)

        return likelihood,entropy
    
    #===========================================================================
    # Update the local parameters with gradient steps until convergence
    #===========================================================================
    def updateLocalParams(self,miniBatch):
        '''
        Update the local parameters for the IBP-ICA model until convergence
        May need some modification here (need to update gradients after calculating something)
        Maybe make something like local_gradient_function ?
        '''
        print("Updating local parameters...")
        old_values=[]
        old_values.append(np.array(self.local_params[0],copy=True))
        old_values.append(np.array(self.local_params[1],copy=True))
        old_values.append(np.array(self.local_params[2],copy=True))

        tolerance=10**-3
        while True:
            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.batch_size)
            self.local_params[0]=np.exp(np.log(self.local_params[0])+0.01*gradients[0])

            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.batch_size)
            self.local_params[1]=np.exp(np.log(self.local_params[1])+0.01*gradients[1])
            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D,S=self.batch_size)
            self.local_params[2]=np.exp(np.log(self.local_params[2])+0.01*gradients[2])
            
            print('--------------------------------------------------------------')
            print(abs(norm(old_values[0]).astype(np.float64)-norm(self.local_params[0]).astype(np.float64)))
            print(abs(norm(old_values[1]).astype(np.float64)-norm(self.local_params[1]).astype(np.float64)))
            print(abs(norm(old_values[1]).astype(np.float64)-norm(self.local_params[2]).astype(np.float64)))
            print('--------------------------------------------------------------')


            #check convergence
            if (abs(norm(old_values[0])-norm(self.local_params[0]))<tolerance):
                if (abs(norm(old_values[1])-norm(self.local_params[1]))<tolerance):
                    if (abs(norm(old_values[2])-norm(self.local_params[2]))<tolerance):
                        print("Local params converged...")
                        break
            old_values=[]
            old_values.append(np.array(self.local_params[0],copy=True))
            old_values.append(np.array(self.local_params[1],copy=True))
            old_values.append(np.array(self.local_params[2],copy=True))
        print(self.local_params[0])
            
    #===========================================================================
    # Calculate the lower bound and create gradient functions for all parameters
    #===========================================================================
    def createGradientFunctions(self):
        '''
        Create the lower bound and use T.grad to get the gradients for free
        '''
        
        print("Creating gradient functions...")
        
        print("\t Initializing prior parameters...")
        a=T.constant(2.0*4,dtype='float64')
        b=T.constant(2.0*4,dtype='float64')
        c=T.constant(2.0*4,dtype='float64')
        f=T.constant(2.0*4,dtype='float64')
        g_1=T.constant(2.0*4,dtype='float64')
        g_2=T.constant(2.0*4,dtype='float64')
        e_1=T.constant(2.0*4,dtype='float64')
        e_2=T.constant(2.0*4,dtype='float64')

        K=T.scalar('K',dtype='int64')
        D=T.scalar('D',dtype='int64')      
        S=T.scalar('S',dtype='int64')
        xi=T.matrix('xi',dtype='float64')
        
        #data matrix
        x=T.matrix('x',dtype='float64')
        
        #create the Theano variables
        #tilde_a,tilde_b,tilde_gamma_1 and tilde_gamma_2 are scalars
        t_a,t_b,t_g_1,t_g_2=T.fscalars('t_a','t_b','t_g_1','t_g_2')
        
        #tilde_eta_1,tilde_eta_2,tilde_xi,tilde_l,tilde_mu,omega,tilde_s, tilde_m and zeta are matrices
        t_e_1,t_e_2,t_xi,t_l,t_mu,omega,t_s,t_m=T.fmatrices('t_e_1','t_e_2','t_xi','t_l','t_mu','omega','t_s','t_m')
        
        #the only tensor we got
        zeta=T.ftensor3('zeta')
    
        #the rest are columns
        t_c,t_f,t_tau,h_tau =T.fcols('t_c','t_f','t_tau','h_tau')
        
               
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
        
         #calculate q_k
#         q_k,up=theano.scan(lambda i, et_tau,eh_tau: T.exp(T.psi(eh_tau[i])+T.sum(T.psi(et_tau[:i-1]))-T.sum(T.psi(et_tau[:i]+eh_tau[:i]))),
#                             sequences=T.arange(K),
#                             non_sequences=[t_tau,h_tau])
#         
#         q_k/=T.sum(q_k)
       
        cs=T.cumsum(T.psi(et_tau),0)-T.psi(et_tau)
        q_k=T.exp(T.psi(eh_tau)+cs-T.cumsum(T.psi(et_tau+eh_tau),0))
        q_k/=T.sum(q_k)
       
     
    
        #check these FOR CORRECTNESS
        try_sum,_=theano.scan(lambda i, qk,ttau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i-1])*(T.sum(qk[:i])-T.cumsum(qk[:i-1]))),0.),
                            sequences=T.arange(K),
                            non_sequences=[q_k,et_tau],
                            strict=True
                            )
        
        try_sum2,_=theano.scan(lambda i, qk,ttau,htau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i]+htau[:i])*(T.cumsum(qk[:i])[::-1])),0.),
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
        
        #expectation_log_gdk=-0.5*K*(T.log(2*np.pi))+0.5*K*T.prod(T.psi(t_c)-T.log(t_f))-0.5*(T.nlinalg.trace(T.dot(t_l,t_c/t_f))+T.diag(T.dot(t_mu.T*(t_c/t_f),t_mu)))
        expectation_log_gdk=-0.5*T.log(2*np.pi)+finalg
        
        expectation_log_zdk=q_z*T.cumsum(T.psi(eh_tau)-T.psi(et_tau+eh_tau))+(1.0-q_z)*mult_bound
        
        expectation_log_varepsilon=ezeta*(T.psi(et_xi)-T.psi(T.sum(et_xi,1)).dimshuffle(0,'x'))
        
        #expectation_log_y=0.5*zeta*(-T.log(2*np.pi)+(T.psi(t_e_1)-T.log(t_e_2))-(t_e_1/t_e_2)*(T.nlinalg.trace(t_s)+T.diag(T.dot(t_m.T,t_m)).dimshuffle(0,'x')))
        expectation_log_y=final_y
        
        expectation_log_x=-0.5*T.log(2*np.pi)+0.5*(T.psi(et_a)-T.log(et_b))-0.5*(et_a/et_b)*final
#         
#         expectation_log_x=-0.5*D*T.log(2*np.pi)+0.5*D*(T.psi(et_a)-T.log(t_b))-0.5*(t_a/t_b)*(T.diag(T.dot(x,x.T))-2*T.diag(T.dot(T.dot(t_m,t_mu.T),x.T))\
#             +(T.nlinalg.trace(t_s)+T.diag(T.dot(t_m,t_m.T)))*T.sum(T.nlinalg.trace(t_l)+T.diag(T.dot(t_mu.T,t_mu))))
#         
      
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
        
        
        #=======================================================================
        # set local and global gradient variables
        #=======================================================================
        #to take derivatives wrt to the log
        gradVariables=[lt_tau,omega,lh_tau,lt_a,lt_c,lt_f,lt_g_1,lt_g_2]
        localVar=[lt_s,t_m,lzeta]
        batch_grad_vars=[lt_e_1,lt_e_2,lt_xi,lt_l,t_mu,lt_b]
        
        #=======================================================================
        # calculate the derivatives
        #=======================================================================
        derivatives=T.grad(lower_bound,gradVariables)
        derivatives_local=T.grad(lower_bound,localVar)
        derivatives_batch=T.grad(lower_bound,batch_grad_vars)
        
        #=======================================================================
        # create the derivative functions    
        #=======================================================================
        #BUT THE INPUT MUST BE THE ORIGINAL VARIABLES
        gradVariables=[t_tau,omega,h_tau,t_a,t_c,t_f,t_g_1,t_g_2]
        localVar=[t_s,t_m,zeta]
        batch_grad_vars=[t_e_1,t_e_2,t_xi,t_l,t_mu,t_b]
        
        
        #testing some stuff, here the gradent is the derivative of the entropy term containing t_s wrt to t_s. With the current values, this gives 0.5 because it replaces
        #ts[n,k] with the initial value
#         der=T.grad(expectation_log_y.sum(),t_s)
#         derx=T.grad(expectation_log_x.sum(),t_s)
#         derent=T.grad(0.5*T.sum(K*(T.log(2*np.pi)+1)+T.sum(T.log(t_s),1)),t_s)
#         self.gradyfunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],der,allow_input_downcast=True,on_unused_input='warn')
#         self.gradx=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],derx,allow_input_downcast=True,on_unused_input='warn')        
#         self.gradent=theano.function([t_s,K],derent,allow_input_downcast=True,on_unused_input='warn')
#         
        dera=T.grad(likelihood,lt_a)
        self.gradafunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],dera,allow_input_downcast=True,on_unused_input='warn')
        

        self.gradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],derivatives,allow_input_downcast=True)
        self.localGradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],derivatives_local,allow_input_downcast=True)
        self.batchGradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],derivatives_batch,allow_input_downcast=True)
        self.lowerBoundFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],lower_bound,allow_input_downcast=True)
        self.likelihoodFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D,S],likelihood,allow_input_downcast=True)
        self.entropyFunction=theano.function(localVar+gradVariables+batch_grad_vars+[K,D,S],entropy,allow_input_downcast=True,on_unused_input='warn')
        
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
    
    K=5
    N=10000
    D=4
    J=8
    S=50
    lower_bound=0
    
    z=IBP_ICA(K,D,J,S,lower_bound)
    
    x,G,y=z.create_synthetic_data(N)
        #sample the data 
    random_indices=np.random.randint(0,len(x),S)
    miniBatch=x[random_indices,:]
    
    z.init_params()
    z.createGradientFunctions()
    LL=[]
    print(z.lowerBoundFunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))
    #print(z.gradyfunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))
    #print(z.gradx(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))
    #print(z.gradent(K=z.K,t_s=z.local_params[0]))
    print('sup',z.gradafunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))

    iteration=1
    while True:
        print("Stochastic IBP-ICA iteration: ",iteration)
        rho=(iteration+1.0)**(-.75)
        iteration+=1
        
        z.lower_bound=0
        z.iterate(miniBatch,rho)
        LL.append(z.lower_bound)
        print(z.lowerBoundFunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch,S=z.batch_size))
        print("Lower Bound at iteration",iteration,"is ",z.lower_bound)
        if (iteration>1):
            break
        
    
#===============================================================================
# DEPRECATED STUFF. KEEP HERE JUST IN CASE
#===============================================================================

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