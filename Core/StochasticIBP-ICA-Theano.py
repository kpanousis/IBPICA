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
theano.config.exception_verbosity='high'
#theano.config.optimizer='None'
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
        t_e_1=2*np.ones((self.K,self.J))
        t_e_2=2*np.ones((self.K,self.J))
        t_xi=np.ones((self.K,self.J))
        t_l=np.random.gamma(1,1,size=(self.D,self.K))
        t_mu=np.random.gamma(1,1,size=(self.D,self.K))
        omega=np.random.random(size=(self.D,self.K))
        
        #more sophisticated initialization
        t_s=np.random.gamma(1,1,size=(self.batch_size,self.K))
        
        #here to run PCA
        t_m=np.random.gamma(1,1,size=(self.batch_size,self.K))
        
        #tensor
        zeta=np.random.random(size=(self.batch_size,self.K,self.J))
        for s in range(self.batch_size):
            zeta[s,:,:]/=zeta[s,:,:].sum(1).reshape(-1,1)
        
        #tcolumns
        t_c=2*np.ones((self.K,1))
        t_f=2*np.ones((self.K,1))
        t_tau=2*np.ones((self.K,1))
        h_tau=2*np.ones((self.K,1))
        
        #the first three are the local params
        self.params=[t_tau,omega,h_tau,t_a,t_c,t_f,t_g_1,t_g_2]
        self.local_params=[t_s,t_m,zeta]
        self.batch_params=[t_e_1,t_e_2,t_xi,t_l,t_mu,t_b]
        self.xi=xi
    
    def feature_update(self):
        '''
        Function for using Gibbs sampling to update the number of features.
        To be implemented today or tomorrow (13-14/07)
        '''
        pass
    
    def getGradients(self,miniBatch):
        '''
        Get the gradients for all the parameters of the model
        '''
        total_gradients=[0]*len(self.params)
        gradients=self.gradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K)
        
        return gradients
    
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
        batch_gradients=self.batchGradientFunction(*(local_params+self.params+self.batch_params),x=miniBatch[s,:].reshape(1,-1),xi=self.xi,K=self.K)
        
        return batch_gradients
        
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
    
    def iterate(self,miniBatch,rho):
        '''
        Function that represents one iteration of the SVI IBP ICA algorithm
        '''
        self.updateLocalParams(miniBatch)
        gradients=self.getGradients(miniBatch)
        batch_gradients=[0]*len(self.batch_params)
        for s in range(len(miniBatch)):
            batch_gradients+=self.getBatchGradients(miniBatch, s)
        self.updateParams(gradients, batch_gradients, len(miniBatch), rho)
    

    def getLowerBound(self,data):
        lower_bound=self.lowerBoundFunction(*(self.local_params+self.params+self.batch_params),xi=self.xi,K=self.K,x=data)    
        return lower_bound
    
    
    def getLowerBoundterms(self,data):
        likelihood=self.likelihoodFunction(*(self.local_params+self.params+self.batch_params),xi=self.xi,K=self.K,x=data)
        entropy=self.entropyFunction(*(self.local_params+self.params+self.batch_params),K=self.K,D=self.D)

        return likelihood,entropy
    
    def updateLocalParams(self,miniBatch):
        '''
        Update the local parameters for the IBP-ICA model until convergence
        May need some modification here (need to update gradients after calculating something)
        Maybe make something like local_gradient_function ?
        '''
        print("Updating local parameters...")
        old_values=self.local_params[:]

        tolerance=10**-5
        while True:
            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D)
            #print("grad[0]:",gradients[0])
            #time.sleep(3)
            self.local_params[0]=gradients[0]
            print("where negative:",np.where(self.local_params[0]<0))
            print("where nan:",np.isnan(self.local_params[0]))
            time.sleep(3)

            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D)
            self.local_params[1]=gradients[1]
            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K,D=self.D)
            self.local_params[2]=gradients[2]
            
          
            #check convergence
            if (abs(norm(old_values[0])-norm(self.local_params[0]))<tolerance):
                if (abs(norm(old_values[1])-norm(self.local_params[1]))<tolerance):
                    if (abs(norm(old_values[2])-norm(self.local_params[2]))<tolerance):
                        print("Local params converged...")
                        break
            old_values=self.local_params[:]

            
    def createGradientFunctions(self):
        '''
        Create the lower bound and use T.grad to get the gradients for free
        '''
        
        print("Creating gradient functions...")
        
        print("Initializing prior parameters...")
        a=T.constant(2.0)
        b=T.constant(2.0)
        c=T.constant(2.0)
        f=T.constant(2.0)
        g_1=T.constant(2.0)
        g_2=T.constant(2.0)
        e_1=T.constant(2.0)
        e_2=T.constant(2.0)

        K=T.iscalar('K')
        D=T.iscalar('D')      
          
        xi=T.fmatrix('xi')
        
        x=T.matrix('x')
        #create the Theano variables
        #tilde_a,tilde_b,tilde_gamma_1 and tilde_gamma_2 are scalars
        t_a,t_b,t_g_1,t_g_2=T.fscalars('t_a','t_b','t_g_1','t_g_2')
        
        #tilde_eta_1,tilde_eta_2,tilde_xi,tilde_l,tilde_mu,omega,tilde_s, tilde_m and zeta are matrices
        t_e_1,t_e_2,t_xi,t_l,t_mu,omega,t_s,t_m=T.fmatrices('t_e_1','t_e_2','t_xi','t_l','t_mu','omega','t_s','t_m')
        
        zeta=T.ftensor3('zeta')
    
        #the rest are columns
        t_c,t_f,t_tau,h_tau =T.fcols('t_c','t_f','t_tau','h_tau')
        
               
        
#         et_g_1=T.exp(t_g_1)
#         et_g_2=T.exp(t_g_2)
#         et_e_1=T.exp(t_e_1)
#         et_e_2=T.exp(t_e_2)
#         et_a=T.exp(t_a)
#         et_b=T.exp(t_b)
#         et_c=T.exp(t_c)
#         et_f=T.exp(t_f)
#         et_l=T.exp(t_l)
#         ezeta=T.exp(zeta)
#         et_xi=T.exp(t_xi)
#         eh_tau=T.exp(h_tau)
#         et_tau=T.exp(t_tau)
#         et_s=T.exp(t_s)
        
         #calculate q_k
#         q_k,up=theano.scan(lambda i, et_tau,eh_tau: T.exp(T.psi(eh_tau[i])+T.sum(T.psi(et_tau[:i-1]))-T.sum(T.psi(et_tau[:i]+eh_tau[:i]))),
#                             sequences=T.arange(K),
#                             non_sequences=[t_tau,h_tau])
#         
#         q_k/=T.sum(q_k)
       
        cs=T.cumsum(T.psi(t_tau),0)-T.psi(t_tau)
        q_k=T.exp(T.psi(h_tau)+cs-T.cumsum(T.psi(t_tau+h_tau),0))
        q_k/=T.sum(q_k)
       
        #=======================================================================
        # Nested scan for the calculation of the multinomial bound
        #=======================================================================
        def oneStep(s_index,curr,tau,q_k,k,add_index):
            return ifelse(T.gt(k,0),curr+T.psi(tau[s_index,0])*T.sum(q_k[s_index+add_index:k+1]),curr)
         
        def inner_loop(k,tau,q_k,add_index):
           
            zero=T.constant(0).astype('int64')
 
            s_indices=ifelse(T.gt(k,zero),T.arange(k),T.arange(0,1))
 
            n_steps=ifelse(T.gt(k-add_index,zero),k-add_index,T.constant(1).astype('int64'))
 
            outputs,_=theano.scan(fn=oneStep,
                                        sequences=s_indices,
                                        outputs_info=T.constant(.0),
                                        non_sequences=[tau,q_k,k,add_index],
                                        )
            #print("outputs",outputs.type)
            outputs=outputs[-1]
            return outputs
         
        add_index=T.constant(1).astype(('int64'))
        sec_sum,updates=theano.scan(fn=inner_loop,
                                      sequences=T.arange(K),
                                      non_sequences=[t_tau,q_k,add_index],
                                      )
         
        #all I need to change for the second sum are the indices
        add_index_zero=T.constant(0).astype('int64')
        third_sum,up=theano.scan(fn=inner_loop,
                                      sequences=T.arange(K),
                                      non_sequences=[t_tau,q_k,add_index_zero],
                                      )
    
        #check these FOR CORRECTNESS
        try_sum,_=theano.scan(lambda i, qk,ttau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i-1])*(T.sum(qk[:i])-T.cumsum(qk[:i-1]))),0.),
                            sequences=T.arange(K),
                            non_sequences=[q_k,t_tau]
                            )
        
        try_sum2,_=theano.scan(lambda i, qk,ttau: ifelse(T.gt(i,0),T.sum(T.psi(ttau[:i]+h_tau[:i])*(T.cumsum(qk[:i])[::-1])),0.),
                            sequences=T.arange(K),
                            non_sequences=[q_k,t_tau]
                            )
      
        print((try_sum.type))
        #the multinomial bound for calculating p(z_{dk}|\upsilon_k)
        mult_bound=T.cumsum(q_k*T.psi(h_tau))+try_sum-try_sum2-T.cumsum(q_k*T.log(q_k))

        #calculate q(z_{dk}=1)
        q_z=1.0/(1.0+T.exp(-omega))
        
       

        #=======================================================================
        # calculation of the weird last term
        #=======================================================================

        
        def normalCalc(d,n,xn,ts,tm,tmu,tl):
            return xn[n,d]**2-2*xn[n,d]*T.sum(tmu[d,:]*tm[n,:])\
                    +T.sum(tmu[d,:]*tm[n,:])**2\
                    +T.sum((tmu[d,:]**2+tl[d,:])*(tm[n,:]**2+ts[n,:])-tmu[d,:]**2*tm[n,:]**2)
             
        def inter_loop(n,xn,ts,tm,tmu,tl):
            
            outputs,_=theano.scan(fn=normalCalc,
                                sequences=T.arange(D),
                                non_sequences=[n,xn,ts,tm,tmu,tl]
                                )
            return outputs
        
        final,_=theano.scan(fn=inter_loop,
                            sequences=T.arange(S),
                            non_sequences=[x,t_s,t_m,t_mu,t_l]
                            )
        
        print("final",final.type)
        
        #=======================================================================
        # Calculate Gdk expectation
        #=======================================================================
        def gcalc(k,d,tc,tf,tmu,tl):
            return +0.5*(T.psi(tc[k,0])-T.log(tf[k,0]))-0.5*(tc[k,0]/tf[k,0])*(tmu[d,k]**2+tl[d,k])
        
        def inter_g_loop(n,tc,tf,tmu,tl):
            
            outputs,_=theano.scan(fn=gcalc,
                                  sequences=T.arange(K),
                                  non_sequences=[n,tc,tf,tmu,tl]
                                  )
            return outputs
        
        finalg,_=theano.scan(fn=inter_g_loop,
                                 sequences=T.arange(D),
                                 non_sequences=[t_c,t_f,t_mu,t_l]
                                 )
        
        #=======================================================================
        # Calculate the expectation of logy
        #=======================================================================
        def y_calc(j,k,n,zt,te1,te2,ts,tm):
            return zt[n,k,j]*(-0.5*T.log(2*np.pi)+0.5*(T.psi(te1[k,j])-T.log(te2[k,j]))-0.5*(te1[k,j]/te2[k,j])*(ts[n,k]+tm[n,k]**2))
        
        def deepest_loop(k,n,zt,te1,te2,ts,tm):
            out,_=theano.scan(fn=y_calc,
                              sequences=T.arange(J),
                              non_sequences=[k,n,zt,te1,te2,ts,tm]
                              )
            return out 
        
        def not_so_deep_loop(n,zt,te1,te2,ts,tm):
            out,_=theano.scan(fn=deepest_loop,
                              sequences=T.arange(K),
                              non_sequences=[n,zt,te1,te2,ts,tm]
                              )
            return out
        
        final_y,_=theano.scan(fn=not_so_deep_loop,
                              sequences=T.arange(S),
                              non_sequences=[zeta,t_e_1,t_e_2,t_s,t_m]
                              )
        print(final_y.type)
#      
        #=======================================================================
        # calculate the likelihood term 
        #=======================================================================
        expectation_log_p_a=g_1*T.log(g_2)+(g_1-1)*(T.psi(t_g_1)-T.log(t_g_2))-g_2*(t_g_1/t_g_2)-T.log(T.gamma(g_1))
        
        expectation_log_p_phi=a*T.log(b)+(a-1)*(T.psi(t_a)-T.log(t_b))-b*(t_a/t_b)-T.log(T.gamma(a))
        
        expectation_log_u_k=T.psi(t_g_1)-T.log(t_g_2)+(t_g_1/t_g_2-1)*(T.psi(t_tau)-T.psi(t_tau+h_tau))
        
        expectation_log_lambda_k=c*T.log(f)+(c-1)*(T.psi(t_c)-T.log(t_f))-f*(t_c/t_f)-T.log(T.gamma(c))
        
        expectation_log_varpi=-T.log(T.prod(T.gamma(xi),1))/T.gamma(T.sum(xi,1))+T.sum((xi-1)*(T.psi(t_xi)-(T.psi(T.sum(t_xi,1))).dimshuffle(0, 'x')),1)
        
        expectation_log_skj=e_1*T.log(e_2)-(e_1+1)*(T.log(t_e_2)-T.psi(t_e_1))-e_2*(t_e_1/t_e_2)-T.log(T.gamma(e_1))
        
        #expectation_log_gdk=-0.5*K*(T.log(2*np.pi))+0.5*K*T.prod(T.psi(t_c)-T.log(t_f))-0.5*(T.nlinalg.trace(T.dot(t_l,t_c/t_f))+T.diag(T.dot(t_mu.T*(t_c/t_f),t_mu)))
        expectation_log_gdk=-0.5*T.log(2*np.pi)+finalg
        
        expectation_log_zdk=q_z*T.cumsum(T.psi(h_tau)-T.psi(t_tau+h_tau))+(1.0-q_z)*mult_bound
        
        expectation_log_varepsilon=zeta*(T.psi(t_xi)-T.psi(T.sum(t_xi,1)).dimshuffle(0,'x'))
        
        #expectation_log_y=0.5*zeta*(-T.log(2*np.pi)+(T.psi(t_e_1)-T.log(t_e_2))-(t_e_1/t_e_2)*(T.nlinalg.trace(t_s)+T.diag(T.dot(t_m.T,t_m)).dimshuffle(0,'x')))
        expectation_log_y=final_y
        
        expectation_log_x=-0.5*T.log(2*np.pi)+0.5*(T.psi(t_a)-T.log(t_b))-0.5*(t_a/t_b)*final
#         
#         expectation_log_x=-0.5*D*T.log(2*np.pi)+0.5*D*(T.psi(t_a)-T.log(t_b))-0.5*(t_a/t_b)*(T.diag(T.dot(x,x.T))-2*T.diag(T.dot(T.dot(t_m,t_mu.T),x.T))\
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
        entropy=t_g_1-T.log(t_g_2)+(1-t_g_1)*T.psi(t_g_1)+T.log(T.gamma(t_g_1)) \
                +t_a-T.log(t_b)+(1-t_a)*T.psi(t_a)+T.log(T.gamma(t_a)) \
                +T.sum(T.log(T.gamma(t_tau)*T.gamma(h_tau)/T.gamma(h_tau+t_tau))-(t_tau-1)*T.psi(t_tau)-(h_tau-1)*T.psi(h_tau)+(t_tau+h_tau-2)*T.psi(t_tau+h_tau)) \
                +T.sum(t_c-T.log(t_f)+T.log(T.gamma(t_c))+(1-t_c)*T.psi(t_c)) \
                +T.sum(T.prod(T.gamma(t_xi),1)/T.gamma(T.sum(t_xi,1))-(J-T.sum(t_xi,1))*T.psi(T.sum(t_xi,1))-(T.sum((t_xi-1.0)*T.psi(t_xi),1))) \
                +T.sum(t_e_1+T.log(t_e_2*T.gamma(t_e_1))-(1+t_e_1)*T.psi(t_e_1)) \
                +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.log(T.prod(t_l,1))) \
                -T.sum((1.0-q_z)*T.log(1.0-q_z))-T.sum(q_z*T.log(q_z)) \
                +0.5*T.sum(K*(T.log(2*np.pi)+1)+T.log(T.prod(t_s,1))) \
                -T.sum(zeta*T.log(zeta))

        lower_bound=likelihood+entropy
        
        #print("LL",likelihood.type)
        
        #=======================================================================
        # set local and global gradient variables
        #=======================================================================
        gradVariables=[t_tau,omega,h_tau,t_a,t_c,t_f,t_g_1,t_g_2]
        localVar=[t_s,t_m,zeta]
        batch_grad_vars=[t_e_1,t_e_2,t_xi,t_l,t_mu,t_b]
        
        #=======================================================================
        # calculate the derivatives
        #=======================================================================
        derivatives=T.grad(lower_bound,gradVariables)
        derivatives_local=T.grad(lower_bound,localVar)
        derivatives_batch=T.grad(lower_bound,batch_grad_vars)
        print(pp(derivatives_local[0]))
        
        t_s_d=T.grad(lower_bound,[zeta])
        t_s_d_f=theano.function_dump('func_ts',localVar+gradVariables+batch_grad_vars+[xi,x,K,D],t_s_d)
        #print('t_s',(t_s_d[0]))
        #print('t_s',pp(t_s_d[0][1]))

                
        self.gradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D],derivatives,allow_input_downcast=True)
        self.localGradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D],derivatives_local,allow_input_downcast=True)
        self.batchGradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D],derivatives_batch,allow_input_downcast=True)
        self.lowerBoundFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D],lower_bound,allow_input_downcast=True)
        self.likelihoodFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K,D],likelihood,allow_input_downcast=True)
        self.entropyFunction=theano.function(localVar+gradVariables+batch_grad_vars+[K,D],entropy,allow_input_downcast=True,on_unused_input='warn')
        print(pp(t_s_d_f.maker.fgraph.outputs[0]))
        
    def create_synthetic_data(self,N):
        G=np.random.random(size=(self.D,self.K))
        y=np.random.random(size=(self.K,N))
        return np.dot(G,y).T,G,y
    
if __name__ == '__main__':
    
    K=5
    N=10000
    D=4
    J=8
    S=500
    lower_bound=0
    
    z=IBP_ICA(K,D,J,S,lower_bound)
    
    x,G,y=z.create_synthetic_data(N)
        #sample the data 
    random_indices=np.random.randint(0,len(x),S)
    miniBatch=x[random_indices,:]
    
    z.init_params()
    z.createGradientFunctions()
    LL=[]
    print(z.lowerBoundFunction(*(z.local_params+z.params+z.batch_params),xi=z.xi,K=z.K,D=z.D,x=miniBatch))

    print(np.where(z.local_params[0]<0))

    print(np.where(z.local_params[1]<0))
    time.sleep(4)
    
    print(z.params[-3])
    print(z.params[0])
    print(z.params[1])

    iteration=1
    while True:
        print("Stochastic IBP-ICA iteration: ",iteration)
        rho=(iteration+1.0)**(-.75)
        iteration+=1
        
        z.lower_bound=0
        z.iterate(miniBatch,rho)
        print(z.local_params)
        print("params",z.params)
        print("batch",z.batch_params)
        LL.append(z.lower_bound)
        print(z.getLowerBoundterms(miniBatch))
        print("Lower Bound at iteration",iteration,"is ",z.lower_bound)
        if (iteration>1):
            break
        
    
#===============================================================================
# DEPRECATED STUFF. KEEP HERE JUST IN CASE
#===============================================================================
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