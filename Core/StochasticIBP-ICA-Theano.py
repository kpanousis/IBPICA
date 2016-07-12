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
    
    def __init__(self,dimX, batch_size,lower_bound):
        self.dimX=dimX
        self.batch_size=batch_size
        self.lower_bound=lower_bound
    
    
    def createGradientFunctions(self,D,K,J,x):
        
        print("Creating gradient functions...")
        
        print("Initializing prior parameters...")
        a,b,c,f,g_1,g_2,e_1,e_2=(1,1,1,1,1,1,1,1)
        xi=T.ones((K,J))*(1.0/J)
        
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
        gradVariables=[omega,h_tau,t_a,t_b,t_c,t_f,t_g_1,t_g_2,t_e_1,t_e_2,t_m,t_mu,t_l,t_s,t_tau,t_xi,zeta]
        
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
                    +T.sum(T.psi(t_g_1)-T.log(t_g_2)+(t_g_1/t_g_2-1)*(T.psi(t_tau)-T.psi(t_tau+h_tau))) \
                    +T.sum(c*T.log(f)+(c-1)*(T.psi(t_c)-T.log(t_f))-f*(t_c/t_f)-T.log(T.gamma(c))) \
                    +T.sum(T.sum((xi-1)*(T.psi(t_xi)-T.psi(T.sum(t_xi,1))),1)) \
                    +T.sum(e_1*T.log(e_2)-(e_1+1)*(T.log(t_e_2)-T.psi(t_e_1))-e_2*(t_e_1/t_e_2)) \
                    +T.sum(0.5*(T.psi(t_c)-T.log(t_f))-0.5*(t_mu**2+t_l)*(t_c/t_f)) \
                    +T.sum(q_z*T.cumsum(T.psi(h_tau)-T.psi(t_tau+h_tau))+(1.0-q_z)*mult_bound)\
                    +T.sum(0.5*zeta*(T.psi(t_e_1)-T.log(t_e_2)-(t_m**2+t_s)*(t_e_1/t_e_2))) \
                    +T.sum(0.5*(T.psi(t_a)-T.log(t_b)) \
                           -0.5*(T.dot(T.transpose(x), x)-T.dot(T.transpose(x),T.dot(T.transpose(t_mu),t_m))-T.dot(T.dot(T.transpose(t_m),T.transpose(t_mu)),x)+T.dot(T.dot(T.transpose(t_m),T.nlinalg.trace(t_l)+T.dot(T.transpose(t_mu),t_mu)),t_m))*(-t_a/t_b))
                    
        entropy=0.5*(K+T.sum(T.log(t_s))) \
                -T.sum((t_tau-1)*(T.psi(t_tau))-(h_tau-1)*T.psi(h_tau)+(t_tau+h_tau-2)*T.psi(t_tau+h_tau)) \
                -T.sum((1.0-q_z)*T.log(1.0-q_z)-q_z*T.log(q_z)) \
                +T.sum(t_g_1-T.log(t_g_2)+(1-t_g_1)*T.psi(t_g_1)) \
                + T.sum(t_c-T.log(t_f)+(1-t_c)*T.psi(t_c)) \
                +T.sum(t_a-T.log(t_b)+(1-t_a)*T.psi(t_a)) \
                -T.sum((J-T.sum(t_xi,1))*T.psi(T.sum(t_xi,1))-T.sum((t_xi-1)*T.psi(t_xi),1)) \
                +T.sum(t_e_1-T.log(t_e_2)+(1-t_e_1)*T.psi(t_e_1)) \
                +0.5*D+T.sum(T.log(T.nlinalg.det(2*np.pi*t_l)))
        
        lower_bound=likelihood+entropy
        
        print("LL",likelihood.type)
        print("E",entropy.type)
        print("lowerbound",lower_bound.type)
        #gradVariables=t_e_1
        derivatives=T.grad(lower_bound,gradVariables)
        #from theano import pp
        #print(pp(derivatives))
        
        derivatives.append(lower_bound)
        
        self.gradientFunction=theano.function(gradVariables+[x],derivatives,allow_input_downcast=True)
        self.lowerBoundFunction=theano.function(gradVariables+[x],lower_bound)
    
if __name__ == '__main__':
    z=IBP_ICA(1,10,0)
    D=10
    K=10
    J=7
    z.createGradientFunctions(D,K,J,T.ones((10,10)))
