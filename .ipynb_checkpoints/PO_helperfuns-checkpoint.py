
import numpy as np
from numpy import linalg as LA
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
    if np.array(p1).size==1: # p1 is a scalar
        denom = p1 + p2
        psi_1 = p1/denom if denom>0 else 0
        psi_2 = p2/denom if denom>0 else 0
    else:
        # get psi_1 and psi_2, and note that if the denominator is 0, then psi_1 = psi_2 = 0
        mask_p1p2pos = p1 + p2 > 0
        psi_1 = np.zeros(np.array(p1).shape)
        denom = p1[mask_p1p2pos] + p2[mask_p1p2pos]
        psi_1[mask_p1p2pos] = p1[mask_p1p2pos]/denom
        psi_2 = np.zeros(np.array(p1).shape)
        psi_2[mask_p1p2pos] = 1 - psi_1[mask_p1p2pos]
                     
    r1 = rvec[0]
    r2 = rvec[1]
    
    Wu1 = u*(1+r1)*(K*psi_1 + pc)
    Wu2 = u*(1+r2)*(K*psi_2 + pc)
    Wbu = u*(1-pc-K)
    
    Wx1 = x*(1+r1)*(K_x*psi_1 + pc_x)
    Wx2 = x*(1+r2)*(K_x*psi_2 + pc_x)
    Wbx = x*(1-pc_x-K_x)

    
    W = Wu1 + Wu2 + Wbu + Wx1 + Wx2 + Wbx 
    freqs = [item/W for item in [Wu1, Wu2, Wbu, Wx1, Wx2, Wbx]]
    uvec = freqs[0:3]; xvec = freqs[3:6]
    
    rvec = [ri_fun(rvec[0], p1, beta,eta), ri_fun(rvec[1],p2,beta, eta)]
    return(uvec, xvec, rvec,W)


# Check equilibrium is internally stable with local stability analysis
""" Test Internal Stability of an equilibrium
Jac_UR_evals finds the Jacobian and then solves for the eigenvalues.
I compare the outputs of Jac_UR_evals and lambdastar to check that my equations in 
the appendix are correct
"""

def Jac_UR_evals(u1, r1,K, pc, beta):
    
    # constants
    L = K/2 + pc
    xi = K*(1+r1)/(4*u1**2)
    W = 1 + pc + 2*r1*(K/2 + pc)
    row1 = np.array([L*(1-u1)/W, -u1*L/W, u1*xi/W, -u1*xi/W])
    row2 = np.array([-u1*L/W, L*(1-u1)/W, -u1*xi/W, u1*xi/W])
    row3 = np.array([-beta*r1/(1+r1), 0, 1/(1+r1),0])
    row4 = np.array([0, -beta*r1/(1+r1), 0, 1/(1+r1)])
    Jac = np.array([row1, row2, row3, row4])
    return(Jac)
""" Test Internal Stability of an equilibrium
lambdastar finds the eigenvalues using my equation (in the appendix)
I compare the outputs of Jac_UR_evals and lambdastar to check that my equations in 
the appendix are correct
"""


def lambdastar(u1,r1, K,pc,beta):
    R = 1 + r1
    L = K/2+pc
    W = 1 + pc +2*r1*L
    a1 = -W*R
    b1 = W
    c1 = 2*L*r1*u1*beta - L*r1*beta
    lamda1 = (-b1 + np.lib.scimath.sqrt(b1**2 - 4*a1*c1))/(2*a1)
    lamda2= (-b1 - np.lib.scimath.sqrt(b1**2 - 4*a1*c1))/(2*a1)
    
    a2 = -2*R*W*u1
    b2 = 2*W*u1 + K*R**2
    c2 = -K*R - 2*L*r1*u1*beta
    discrim2 = b2**2 - 4*a2*c2
    lamda3 = (-b2 + np.lib.scimath.sqrt(discrim2))/(2*a2) # this is the smaller one because a1 < 0
    lamda4= (-b2 - np.lib.scimath.sqrt(discrim2))/(2*a2)
    
    return([lamda1,lamda2,lamda3,lamda4])

# Check equilibrium is externally stable with local stability analysis
# Finds C_s for external stability analysis
def Grad_s(uvec, rvec, W, s, beta, mu):
    norm = scs.norm(mu,1)
    u1 = uvec[0]
    u2 = uvec[1]
    # get psi_1 and psi_2, and note that if the denominator is 0, then psi_1 = psi_2 = 0
    mask_p1p2pos = u1 + u2 > 0
    psi_1 = np.zeros(np.array(u1).shape)
    denom = u1[mask_p1p2pos] + u2[mask_p1p2pos]
    psi_1[mask_p1p2pos] = u1[mask_p1p2pos]/denom
    psi_2 = np.zeros(np.array(u1).shape)
    psi_2[mask_p1p2pos] = 1 - psi_1[mask_p1p2pos]
    r1 = rvec[0]
    r2 = rvec[1]
    fs = norm.pdf(s)
    fminuss = norm.pdf(-s)
    C_s = (1/W)*(-fs + r1*((fminuss +fs)*psi_1 - fs) + r2*((fminuss + fs)*psi_2 -fs))
    return(C_s)

# solve predicted equilibrium
# input: parameters K, pc, beta. For this to work properly, they all need to be the same size
# output: u1 value at equilibrium
def PredictEquilibrium_NoPref(K,pc,beta, AllowWarning = 0):
    if AllowWarning==1:
        if (np.array(K).shape != np.array(pc).shape) or (np.array(K).shape != np.array(beta).shape):
            print('K, pc, beta are not all the same shape in PredictEquilibrium_NoPref. This may cause problems. Their shapes are ' + str(np.array(K).shape) +","+ str(np.array(pc).shape) +","+ str(np.array(beta).shape))
        

    def Equilibrium_beta0(L,pc):
        return(2*L/(1+pc+2*L))
    
    def Equilibrium_betapos(L,pc,beta):
        a = 2*beta*L
        b = -(1 + pc + L*(2+beta))
        c = 2*L
        u1_minus = (-b - np.sqrt(b**2 - 4*a*c))/(2*a) # the equilibrium must be the smaller root
        return(u1_minus)
    
    L = K*0.5 + pc
    if np.array(beta).size >1:
        ans = np.zeros(beta.shape)
        if np.array(L).size>1:
            ans[beta>0]= Equilibrium_betapos(L[beta>0],pc[beta>0],beta[beta>0])
            ans[beta==0]=Equilibrium_beta0(L[beta==0],pc[beta==0])
        else:
            ans[beta>0]= Equilibrium_betapos(L,pc,beta[beta>0])
            ans[beta==0]=Equilibrium_beta0(L,pc)
    else:  
        ans = Equilibrium_betapos(L,pc,beta) if beta > 0 else Equilibrium_beta0(L,pc)
    
    return(ans)