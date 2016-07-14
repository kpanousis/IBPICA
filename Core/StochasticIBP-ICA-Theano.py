'''
Created on Jun 29, 2016

@author: kon
'''
import theano
import theano.tensor as T
import numpy as np
import time
from theano.ifelse import ifelse
theano.config.exception_verbosity='high'

class IBP_ICA:
    
    def __init__(self,K,D,J, batch_size,lower_bound):
        self.K=K
        self.D=D
        self.J=J
        self.batch_size=batch_size
        self.lower_bound=lower_bound
    
    def init_params(self):
        '''
        Initialize the parameters to pass to the model
        '''
        xi=np.ones((self.K,self.J))*(1.0/self.J)
        
        #scalars
        t_a=1.0
        t_b=1.0
        t_g_1=1.0
        t_g_2=1.0
        
        #matrices
        t_e_1=np.ones((self.K,self.J))
        t_e_2=np.ones((self.K,self.J))
        t_xi=np.ones((self.K,self.J))*(1.0/J)
        t_l=np.random.gamma(1,1,size=(self.D,self.K))
        t_mu=np.random.normal(0,1,size=(self.D,self.K))
        omega=np.random.random(size=(self.D,self.K))
        #more sophisticated initialization
        t_s=np.random.gamma(1,1,size=(self.batch_size,self.K))
        
        #here to run PCA
        t_m=np.random.normal(0,1,size=(self.batch_size,self.K))
        
        #tensor
        zeta=np.random.random(size=(self.batch_size,self.K,self.J))
        for s in range(self.batch_size):
            zeta[s,:,:]/=zeta[s,:,:].sum(1).reshape(-1,1)
        
        #tcolumns
        t_c=np.ones((self.K,1))
        t_f=np.ones((self.K,1))
        t_tau=np.ones((self.K,1))
        h_tau=np.ones((self.K,1))
        
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
        self.lower_bound+=gradients[-1]
        
#         for i in range(len(self.params)):
#             total_gradients[i]+=gradients[i]
            
        return gradients
    
    def getBatchGradients(self,miniBatch,s):
        
        t_s=self.params[0][s,:].reshape(1,-1)
        t_s=np.repeat(t_s,s,axis=0)
        
        t_m=self.params[1][s,:].reshape(1,-1)
        t_m=np.repeat(t_m,s,axis=0)
        
        #stack the same elements to get the N*zeta_{skj}
        zeta=self.local_params[2][s,:,:]
        zeta=zeta[newaxis,:,:]
        zeta=np.repeat(zeta, len(miniBatch),axis=0)
        
        #get only the elements associated with the sth observation
        local_params=[t_s,t_m,zeta]
        batch_gradients=self.batchGradientFunction(*(local_params+self.params+self.batch_params),x=minibatch[s,:],xi=self.xi,K=self.K)
        
        return batch_gradients
        
    def updateParams(self,gradients,batch_gradients,current_batch_size,rho):
        '''
        Update the global parameters with a gradient step
        '''
        
        for i in range(len(self.params)):
            self.params[i]*=(1.0-rho)
            self.params[i]+=rho*gradients[i]
        for i in range(len(self.batch_params)):
            self.batch_params[i]*=(1-rho)
            self.batch_params[i]+=(rho/current_batch_size)*(batch_gradients[i])
    
    def iterate(self,miniBatch,rho):
        '''
        Function that represents one iteration of the SVI IBP ICA algorithm
        '''
        self.updateLocalParams(miniBatch)
        _,gradients=self.getGradients(miniBatch)
        batch_gradients=[0]*len(self.batch_params)
        for s in range(len(miniBatch)):
            batch_gradients+=self.getBatchGradients(miniBatch, s)
        self.updateParams(gradients, batch_gradients, len(miniBatch), rho)
    

    def getLowerBound(self,data):
        lower_bound=self.lowerBoundFunction(*(self.local_params+self.params+self.batch_params),xi=self.xi,K=self.K,x=data)    
        return lower_bound
            
    def updateLocalParams(self,miniBatch):
        '''
        Update the local parameters for the IBP-ICA model until convergence
        May need some modification here (need to update gradients after calculating something)
        Maybe make something like local_gradient_function ?
        '''
        old_values=self.local_params
        tolerance=10**-5
        while True:
            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K)
            self.local_params[0]=gradients[0]
            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K)
            self.local_params[1]=gradients[1]
            gradients=self.localGradientFunction(*(self.local_params+self.params+self.batch_params),x=miniBatch,xi=self.xi,K=self.K)
            self.local_params[2]=gradients[2]
            
            #check convergence
            if (abs(norm(old_values[0])-norm(self.local_params[0]))<tolerance):
                if (abs(norm(old_values[1])-norm(self.local_params[1]))<tolerance):
                    if (abs(norm(old_values[2])-norm(self.local_params[2]))<tolerance):
                        break
            
    def createGradientFunctions(self):
        '''
        Create the lower bound and use T.grad to get the gradients for free
        '''
        
        print("Creating gradient functions...")
        
        print("Initializing prior parameters...")
        a=T.constant(1.0)
        b=T.constant(1.0)
        c=T.constant(1.0)
        f=T.constant(1.0)
        g_1=T.constant(1.0)
        g_2=T.constant(1.0)
        e_1=T.constant(1.0)
        e_2=T.constant(1.0)

        K=T.iscalar('K')
        
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
        
        #calculate q_k
        #scan this shit
        def qkStep(s_index,t_tau,h_tau):
            return T.exp(T.psi(h_tau[s_index])+T.sum(T.psi(t_tau[:s_index]))-T.sum(T.psi(t_tau[:s_index+1]+h_tau[:s_index+1])))
        
        indices=T.arange(K)
        q_k,up=theano.scan(fn=qkStep,
                           sequences=indices,
                           non_sequences=[t_tau,h_tau])
        
        q_k/=T.sum(q_k)
       
                
        #some terms for the multinomial bound
        #entropy of q_k        
        q_k_entropy=T.sum(q_k*T.log(q_k))
        f_sum=T.cumsum(q_k*T.psi(h_tau))
        sum_m_plus_one_k=T.cumsum(q_k)
        
        
        #=======================================================================
        # Nested scan for the calculation of the multinomial bound
        #=======================================================================
        def oneStep(s_index,curr,tau,q_k,k,add_index):
            zero=T.constant(0).astype('int64')
            return ifelse(T.gt(k,zero),curr+T.psi(tau[s_index,0])*T.sum(q_k[s_index+add_index:k+1]),curr+T.constant(0).astype('float32'))
        
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
        
        start_slices=T.arange(K)
        add_index=T.constant(1).astype(('int64'))
        sec_sum,updates=theano.scan(fn=inner_loop,
                                      sequences=start_slices,
                                      non_sequences=[t_tau,q_k,add_index],
                                      )
        
        #all I need to change for the second sum are the indices
        add_index_zero=T.constant(0).astype('int64')
        third_sum,up=theano.scan(fn=inner_loop,
                                      sequences=[start_slices],
                                      non_sequences=[t_tau,q_k,add_index_zero],
                                      )
        #the multinomial bound for calculating p(z_{dk}|\upsilon_k)
        #WTF
        #print("sec",sec_sum.type)
        #print("thrd",third_sum.type)
        mult_bound=T.cumsum(q_k*T.psi(h_tau))+sec_sum+third_sum-T.cumsum(q_k*T.log(q_k))
        #print("Mult_bound",mult_bound.type)
        #calculate q(z_{dk}=1)
        q_z=1.0/(1.0+T.exp(-omega))
        
        #calculate likelihood
        #missing term for zeta
    
        #=======================================================================
        # modified this a little bit cause we need N times some stuff and not sum over N
        #=======================================================================

        likelihood=g_1*T.log(g_2)+(g_1-1)*(T.psi(t_g_1)-T.log(t_g_2))-g_2*(t_g_1/t_g_2)-T.log(T.gamma(g_1)) \
                    +a*T.log(b)+(a-1)*(T.psi(t_a)-T.log(t_b))-b*(t_a/t_b)-T.log(T.gamma(a)) \
                    +T.sum(-T.psi(t_g_1)+T.log(t_g_2)+(t_g_1/t_g_2-1)*(T.psi(t_tau)-T.psi(t_tau+h_tau))) \
                    +T.sum(c*T.log(f)+(c-1)*(T.psi(t_c)-T.log(t_f))-f*(t_c/t_f)-T.log(T.gamma(c))) \
                    +T.sum(T.sum((xi-1)*(T.psi(t_xi)-T.psi(T.sum(t_xi,1))),1)) \
                    +T.sum(e_1*T.log(e_2)-(e_1+1)*(T.log(t_e_2)-T.psi(t_e_1))-e_2*(t_e_1/t_e_2))-T.gamma(e_1) \
                    +T.sum(0.5*T.log(2*np.pi)+0.5*(T.psi(t_c)-T.log(t_f))-0.5*(t_mu**2+t_l)*(t_c/t_f)) \
                    +T.sum(q_z*T.cumsum(T.psi(h_tau)-T.psi(t_tau+h_tau))+(1.0-q_z)*mult_bound)\
                    +T.sum(zeta*(T.psi(t_xi)-T.psi(T.sum(t_xi,1))))\
                    +T.sum(0.5*T.log(2*np.pi)+0.5*zeta*(T.psi(t_e_1)-T.log(t_e_2)-(t_m**2+t_s)*(t_e_1/t_e_2))) \
                    +T.sum(-0.5*T.log(2*np.pi)-0.5*(T.psi(t_a)-T.log(t_b)) \
                           -0.5*(T.dot(T.transpose(x), x)-T.dot(T.dot(x,t_mu),t_m.T)\
                                 -T.dot(T.dot(t_m,T.transpose(t_mu)),x.T)+T.dot(T.dot(t_m,T.nlinalg.trace(t_l)+T.dot(T.transpose(t_mu),t_mu)),(t_m**2+t_s   )))*(t_b/(t_a-1)))
                    
        entropy=T.sum(t_g_1-T.log(t_g_2)+(1-t_g_1)*T.psi(t_g_1)) \
                +T.sum(t_a-T.log(t_b)+(1-t_a)*T.psi(t_a)) \
                -T.sum((t_tau-1)*(T.psi(t_tau))-(h_tau-1)*T.psi(h_tau)+(t_tau+h_tau-2)*T.psi(t_tau+h_tau)) \
                +T.sum(t_c-T.log(t_f)+T.log(t_c)+(1-t_c)*T.psi(t_c)) \
                -T.sum((T.sum(1-t_xi,1))*T.psi(T.sum(t_xi,1))-T.sum((t_xi-1)*T.psi(t_xi),1)) \
                +T.sum(t_e_1+T.log(t_e_2*T.gamma(t_e_1))-(1+t_e_1)*T.psi(t_e_1)) \
                +T.sum(0.5*(T.log(2*np.pi*t_l))+1) \
                -T.sum((1.0-q_z)*T.log(1.0-q_z)+q_z*T.log(q_z)) \
                +T.sum(0.5*T.log(2*np.pi*t_s)+1) \
                +T.sum(zeta)
                        
        lower_bound=likelihood+entropy
        
        #print("LL",likelihood.type)
        #print("E",entropy.type)
        #print("lowerbound",lower_bound.type)
        #gradVariables=t_e_1
        
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
       
        
        derivatives.append(lower_bound)
        
        self.gradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K],derivatives,allow_input_downcast=True)
        self.localGradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K],derivatives_local,allow_input_downcast=True)
        self.batchGradientFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K],derivatives_batch,allow_input_downcast=True)
        self.lowerBoundFunction=theano.function(localVar+gradVariables+batch_grad_vars+[xi,x,K],lower_bound)
    
    def create_synthetic_data(self,N):
        G=np.random.normal(size=(self.D,self.K))
        y=np.random.normal(size=(self.K,N))
        return np.dot(G,y).T,G,y
    
if __name__ == '__main__':
    
    K=5
    N=10000
    D=10
    J=10
    S=2000
    lower_bound=0
    
    z=IBP_ICA(K,D,J,S,lower_bound)
    
    x,G,y=z.create_synthetic_data(N)
    
    z.init_params()
    z.createGradientFunctions()
    i=1
    LL=[]
    
    #sample the data 
    random_indices=np.random.randint(0,len(x),S)
    miniBatch=x[random_indices,:]
    
    while True:
        
        rho=(i+1.0)**(-.75)
        i+=1
        
        z.lower_bound=0
        z.iterate(miniBatch,rho)
        break
        
    
