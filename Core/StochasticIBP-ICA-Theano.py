'''
Created on Jun 29, 2016

@author: kon
'''
import theano
import theano.tensor as T
import numpy as np
import time
from theano.ifelse import ifelse

class IBP_ICA:
    
    def __init__(self,K,N,D,J, batch_size,lower_bound):
        self.K=K
        self.N=N
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
        omega=np.random.random(size=(D,K))
        #more sophisticated initialization
        t_s=np.random.gamma(1,1,size=(N,K))
        
        #here to run PCA
        t_m=np.random.normal(0,1,size=(N,K))
        
        #tensor
        zeta=np.random.random(size=(N,K,J))
        for i in range(self.N):
            zeta[i,:,:]/=zeta[i,:,:].sum(1).reshape(-1,1)
        
        #tcolumns
        t_c=np.ones((self.K))
        t_f=np.ones((self.K))
        t_tau=np.ones((self.K))
        h_tau=np.ones((self.K))
        
        #the first three are the local params
        self.params=[t_s,t_m,zeta,t_tau,omega,h_tau,t_a,t_c,t_f,t_g_1,t_g_2,t_e_1,t_e_2,t_xi,t_mu,t_l,t_b]
        self.xi=xi
    
    def getGradients(self,miniBatch):
        '''
        Get the gradients for all the parameters of the model
        '''
        total_gradients=[0]*len(self.params)
        gradients=self.gradientFunction(*(self.params),x=miniBatch,xi=self.xi,K=self.K)
        self.lower_bound+=gradients[-1]
        
        for i in range(len(self.params)):
            total_gradients[i]+=gradients[i]
            
        return total_gradients
    
    def updateParams(self,total_gradients, N,current_batch_size,rho):
        '''
        Update the global parameters with a gradient step
        '''
        
        for i in range(3,len(self.params)):
            self.params[i]*=(1.0-rho)
            if i>10:
                self.params[i]+=(rho/current_batch_size)*(total_gradients[i].sum(0))
            else:
                self.params[i]+=rho*total_gradients[i]
    
    
    def updateLocalParams(self,miniBatch):
        '''
        Update the local parameters for the IBP-ICA model until convergence
        May need some modification here (need to update gradients after calculating something)
        Maybe make something like local_gradient_function ?
        '''
        old_values=self.params[:3]
        tolerance=10**-5
        while True:
            gradients=self.gradientFunction(*(self.params),x=miniBatch,xi=self.xi,K=self.K)
            self.params[0]=gradients[0]
            self.params[1]=gradient[1]
            self.params[2]=gradient[2]
            if (abs(norm(old_values[0])-norm(self.params[0]))<tolerance):
                break
            if (abs(norm(old_values[1])-norm(self.params[1]))<tolerance):
                break
            if (abs(norm(old_values[2])-norm(self.params[2]))<tolerance):
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
       
        
        #set the gradient variables
        gradVariables=[t_s,t_m,zeta,t_tau,omega,h_tau,t_a,t_c,t_f,t_g_1,t_g_2,t_e_1,t_e_2,t_xi,t_mu,t_l,t_b]
        
        #some terms for the multinomial bound
        #entropy of q_k        
        q_k_entropy=T.sum(q_k*T.log(q_k))
        print("entropy", q_k_entropy.type)
        f_sum=T.cumsum(q_k*T.psi(h_tau))
        print("f_sum",f_sum.type)
        sum_m_plus_one_k=T.cumsum(q_k)
        print("sum_plus_one_type",sum_m_plus_one_k.type)
        
        print("q_k",q_k.type)
        #=======================================================================
        # Nested scan for the calculation of the multinomial bound
        #=======================================================================
        def oneStep(s_index,curr,tau,q_k,k,add_index):
            print("s_index",s_index.type)
            print("tau",tau.type)
            print("tau[s]",tau[s_index,1].type)
            print("curr",curr.type)
            print("qk",q_k.type)
            print((curr+T.psi(tau[s_index,1])*T.sum(q_k[s_index+add_index:k+1])).type)
            return ifelse(T.gt(k,0),curr+T.psi(tau[s_index,1])*T.sum(q_k[s_index+add_index:k+1]),curr+T.constant(0).astype('float32'))
        
        def inner_loop(k,tau,q_k,add_index):
            #my_tau=tau[:k+add_index]
            #my_q=q_k[:k+1]
            zero=T.constant(0).astype('int64')

            s_indices=ifelse(T.gt(k,zero),T.arange(k),T.arange(-1,0))

            n_steps=ifelse(T.gt(k-add_index,zero),k-add_index,T.constant(1).astype('int64'))

            outputs,_=theano.scan(fn=oneStep,
                                        sequences=s_indices,
                                        outputs_info=T.constant(0.),
                                        non_sequences=[tau,q_k,k,add_index],
                                        n_steps=n_steps)
            print("outputs",outputs.type)
            outputs=outputs[-1]
            return outputs

        start_slices=T.arange(K)
        add_index=T.constant(1).astype(('int64'))
        sec_sum,updates=theano.scan(fn=inner_loop,
                                      sequences=start_slices,
                                      non_sequences=[t_tau,q_k,add_index],
                                      n_steps=K)
        
        #all I need to change for the second sum are the indices
        add_index_zero=T.constant(0).astype('int64')
        third_sum,up=theano.scan(fn=inner_loop,
                                      sequences=[start_slices],
                                      non_sequences=[t_tau,q_k,add_index_zero],
                                      n_steps=K)
        #the multinomial bound for calculating p(z_{dk}|\upsilon_k)
        #WTF
        print("sec",sec_sum.type)
        print("thrd",third_sum.type)
        mult_bound=T.cumsum(q_k*T.psi(h_tau))+sec_sum+third_sum-T.cumsum(q_k*T.log(q_k))
        print("Mult_bound",mult_bound.type)
        #calculate q(z_{dk}=1)
        q_z=1.0/(1.0+T.exp(-omega))
        
        #calculate likelihood
        #missing term for zeta
    

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
                           -0.5*(T.dot(T.transpose(x), x)-T.dot(T.dot(T.transpose(x),T.transpose(t_mu)),t_m)-T.dot(T.dot(T.transpose(t_m),T.transpose(t_mu)),x)+T.dot(T.dot(T.transpose(t_m),T.nlinalg.trace(t_l)+T.dot(T.transpose(t_mu),t_mu)),t_m))*(t_b/(t_a-1)))
                    
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
        
        print("LL",likelihood.type)
        print("E",entropy.type)
        print("lowerbound",lower_bound.type)
        #gradVariables=t_e_1
        derivatives=T.grad(lower_bound,gradVariables)
        #from theano import pp
        #print(pp(derivatives))
        
        derivatives.append(lower_bound)
        
        self.gradientFunction=theano.function(gradVariables+[xi,x,K],derivatives,allow_input_downcast=True)
        self.lowerBoundFunction=theano.function(gradVariables+[xi,x,K],lower_bound)
    
if __name__ == '__main__':
    K=5
    N=100
    D=10
    J=10
    S=2000
    lower_bound=0
    z=IBP_ICA(K,N,D,J,S,lower_bound)
    #D=10
    #K=10
    #J=7
    z.init_params()
    z.createGradientFunctions()
