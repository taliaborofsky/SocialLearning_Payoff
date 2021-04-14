
import numpy as np
from numpy import linalg as LA
from helperfuns import *
import scipy.stats as scs
import matplotlib.pyplot as plt

# find K given s and the normal curve
def Kfun(s, norm = scs.norm(0.2,1)):
    # Finds K, the probability of socially learning
    #input: d real positive numbe or -1, and a normmal curve
    #output K
    K = norm.cdf(s) - norm.cdf(-s)
    return(K)
# find pc given s and the normal curve
def pcfun(s, norm = scs.norm(0.2,1)):
    # Finds pc, the probability of individual learning correctly
    #input: d real positive numbe or -1, and a normmal curve
    #output pc
    pc = 1 - norm.cdf(s)
    return(pc)

# find pw given s and the normal curve
def pwfun(s,norm = scs.norm(0.2,1)):
    # Finds pw, the probability of individual learning incorrectly
    #input: d real positive numbe or -1, and a normmal curve
    #output pw
    pw = norm.cdf(-s)
    return(pw)


# Find the right side of the recursion shown in Eqs 4a, 4c, or 4e (in the text)
def Wv_fun(psi_i,v,r_i,K,pc):
    # This is the recursion for u_i, x_i, y_i, times W... It's the left side of the recursion equations
    #v is a stand in for u, x, or y
    soc_learn = K*psi_i
    to_return = v*(1+r_i)*(soc_learn + pc)
    return(to_return)

# Find the right side of the recursion shown in Eqs 4b, 4d, or 4f (in the text)
def Wvbar_fun(v,K,pc):
    # This is the left side of the recursion equations for \bar{u}, \bar{x}, and \bar{y}
    pw = 1 - K - pc
    return(v*pw)

# The resource recursion
def ri_fun(r_i,p_i,beta, eta = 1):
    # The recursion for resource $i$
    ri_next = r_i*(1 + eta - beta*p_i)/(1+eta*r_i)
    return(ri_next)

# Iterates the system one time step.
def NextGen(uvec,xvec,rvec,K,pc,beta,deltas = [0, 0], eta=1):
    # Uses the recursion to get the frequencies in the next generation (u_1', u_2', ...)
    #@inputs: curr is current values of u_1,u_2,bu,...
    #         The other inputs are the parameters
    #         deltas are [delta_K, delta_pc]
    
    K_x = K + deltas[0]; pc_x = pc + deltas[1];
    u = sum(uvec)
    x = sum(xvec)

    p1 = uvec[0] + xvec[0] 
    p2 = uvec[1] + xvec[1] 
    
    # get psi_1 and psi_2, and note that if the denominator is 0, then psi_1 = psi_2 = 0
    mask_p1p2pos = p1 + p2 > 0
    psi_1 = p1
    psi_1[mask_p1p2pos] = p1[mask_p1p2pos]/(p1[mask_p1p2pos] + p2[mask_p1p2pos])
    psi_2 = p2
    psi_2[mask_p1p2pos] = p2[mask_p1p2pos]/(p1[mask_p1p2pos] + p2[mask_p1p2pos])

    
    Wu1 = Wv_fun(psi_1,u,rvec[0],K,pc)
    Wu2 = Wv_fun(psi_2,u,rvec[1],K,pc)
    Wbu = Wvbar_fun(u,K,pc)
    
    Wx1 = Wv_fun(p1,x,rvec[0],K_x,pc_x)
    Wx2 = Wv_fun(p2,x,rvec[1],K_x,pc_x)
    Wbx = Wvbar_fun(x,K_x,pc_x)

    
    W = Wu1 + Wu2 + Wbu + Wx1 + Wx2 + Wbx 
    freqs = (1/W)*np.array([Wu1, Wu2, Wbu, Wx1, Wx2, Wbx])
    uvec = freqs[0:3]; xvec = freqs[3:6]
    
    rvec = [ri_fun(rvec[0], p1, beta,eta), ri_fun(rvec[1],p2,beta, eta)]
    return(uvec, xvec, rvec,W)


# Check equilibrium is internally stable with local stability analysis

def Jac_UR(uvec, rvec, K,  W, beta):
    return(1)

# Check equilibrium is externally stable with local stability analysis
def Grad_s(uvec, rvec, W, s, beta):
    return 1
# solve predicted equilibrium
def PredictEquilibrium_NoPref(K,pc,beta):
    def Equilibrium_beta0(L,pc):
        return(2*L/(1+pc+2*L))
    
    def Equilibrium_betapos(L,pc,beta):
        a = 2*beta*L
        b = -(1 + pc + L*(2+beta))
        c = 2*L
        u1_minus = (-b - np.sqrt(b**2 - 4*a*c))/(2*a) # the equilibrium must be the smaller root
        return(u1_minus)
    L = K*0.5 + pc
    if type(K) != int:
        ans = np.zeros(len(K))
        ans[beta>0]= Equilibrium_betapos(L[beta>0],pc[beta>0],beta[beta>0])
        ans[beta==0]=Equilibrium_beta0(L[beta==0],pc[beta==0])
    else:
        ans = Equilibrium_betapos(L,pc,beta) if beta > 0 else Equilibrium_beta0
   
    
    return(ans)