
import numpy as np
from numpy import linalg as LA
import scipy.stats as scs
import matplotlib.pyplot as plt

# find K given s and the normal curve
def LearningCurve(ri, b,c):
    return(1 - exp(-(ri/b)^c))


# The resource recursion
def ri_fun(r_i,p_i,beta, eta = 1):
    # The recursion for resource $i$
    ri_next = r_i*(1 + eta - beta*p_i)/(1+eta*r_i)
    return(ri_next)

# Iterates the system one time step.
def NextGen(uvec,xvec,rvec,K,pc,beta,b,c, dk, eta=1):
    # Uses the recursion to get the frequencies in the next generation (u_1', u_2', ...)
    #@inputs: curr is current values of u_1,u_2,bu,...
    #         The other inputs are the parameters
    #         deltas are [delta_K, delta_pc]
    
    K_x = K + dk; pc_x = pc - dk;
    u = sum(uvec)
    x = sum(xvec)

    p1 = uvec[0] + xvec[0] 
    p2 = uvec[1] + xvec[1] 
    
    r1 = rvec[0]
    r2 = rvec[1]
    
    Wu1 = u*(1+r1)*(K*p1 + pc*LearningCurve(r1,b,c))
    Wu2 = u*(1+r2)*(K*p2 + pc*LearningCurve(r2,b,c))
    Wbu = u*(K*(1-p1-p2) + pc*(1-LearningCurve(r1,b,c))(1-LearningCurve(r2,b,c)))
    
    Wx1 = x*(1+r1)*(K_x*p1 + pc_x*LearningCurve(r1,b,c))
    Wx2 = x*(1+r2)*(K_x*p2 + pc_x*LearningCurve(r2,b,c))
    Wbx = x*(K_x*(1-p1-p2) + pc_x*(1-LearningCurve(r1,b,c))(1-LearningCurve(r2,b,c)))

    
    W = Wu1 + Wu2 + Wbu + Wx1 + Wx2 + Wbx 
    uvec = [item/W for item in [Wu1, Wu2, Wbu]]
    xvec = [item/W for item in [Wx1, Wx2, Wbx]]
    
    rvec = [ri_fun(r1, p1, beta,eta), ri_fun(r2,p2,beta, eta)]
    return(uvec, xvec, rvec,W)


# Check equilibrium is internally stable with local stability analysis

def Jac_UR(uvec, rvec, K,  W, beta):
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    Jac = np.array([])
    return(1)

# Check equilibrium is externally stable with local stability analysis
# Finds C_s for external stability analysis
def Grad_s(uvec, rvec, W, b, c):
    u1 = uvec[0]; u2 = uvec[1]; r1 = rvec[0]; r2 = rvec[1];
    sigma1 = LearningCurve(r1,b,c); sigma2 = LearningCurve(r2,b,c)
    C_s = (1/W)*(r1*(u1 - sigma1) + r2*(u2 - sigma2) - sigma1*sigma2)
    return(C_s)

