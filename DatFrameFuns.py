import numpy as np
from numpy import linalg as LA
from helperfunsPayOff import *
import scipy.stats as scs
import pandas as pd
#for parallelizing:
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
# for solving when iterations take too long
from scipy.optimize import fsolve

# Make the parameter mesh
def get_param_grid(svals,muvals, betavals):
    # gives parameter grids, with u1/u2 initials for the u2 side of the simplex (need to be reflected later)
    u1init = [0.05,0.48, 0.08]
    u2init = [0.1,0.4,0.9]
    which_uinit = [0,1,2]
    r1init = [0.05, 0.5, 0.85, 0.2, 0.9] #each point's reflection accross the 45˚ line is included
    r2init = [0.1,  0.3,  0.9,  0.89, 0.2]
    which_rinit = [0,1,2,3,4]

    # make mesh
    smesh, betamesh, which_uinit_mesh, which_rinit_mesh, mumesh = np.meshgrid(svals,betavals,which_uinit, which_rinit,muvals)
    # flatten
    [svec,betavec,which_uinit_vec, which_rinit_vec, muvec] = [np.ndarray.flatten(item) for item in [smesh, betamesh, which_uinit_mesh, which_rinit_mesh, mumesh]]
    # get rid of invalid 'rows' where bu is < 0
    u1vec = np.array([u1init[i] for i in which_uinit_vec])
    u2vec = np.array([u2init[i] for i in which_uinit_vec])
    buvec = 1 - u1vec-u2vec
    r1vec = np.array([r1init[i] for i in which_rinit_vec])
    r2vec = np.array([r2init[i] for i in which_rinit_vec])
    whichOK = buvec >=0
    [svec,betavec,u1vec,u2vec, r1vec, r2vec, muvec] = [item[whichOK] for item in [svec,betavec,u1vec,u2vec,r1vec,r2vec, muvec]]
    buvec = 1 - u1vec-u2vec


    norms = scs.norm(muvec,1)
    Kvec = Kfun(svec,norms)
    pcvec = pcfun(svec,norms)
    pwvec = 1 - Kvec - pcvec
    n = len(Kvec)

    data = {'mu':muvec,'K': Kvec, 'pc': pcvec,'s':svec, 'beta': betavec, 'u1init': u1vec, 'u2init':u2vec, 
            'buinit':buvec, 'r1init':r1vec, 'r2init':r2vec, 'u1eq': np.zeros(n), 'u2eq': np.zeros(n), 'bueq':np.zeros(n),
            'r1eq':np.zeros(n), 'r2eq':np.zeros(n), 'Weq': np.zeros(n), 'time': np.zeros(n), 'reached_eq': np.zeros(n),
           'URstable': np.zeros(n)}
    df = pd.DataFrame(data = data)
    df = df.sort_values(by=['mu','beta','s'])
    df = df.reindex(np.arange(len(df.index)))
    
    
    
    return(df)

# Iterate each initial point and parameter combination 50000 times.
def get_1side_df(df):
    #make sure indexed correctly
    tsteps = 500000
    
    
    result = GetXsteps(df, tsteps = tsteps)
    umat, xmat, rmat, W, reached_eq = result
    
    #check_eq 
    
    
    df.u1eq = umat[0]
    df.u2eq = umat[1]
    df.bueq = umat[2]
    df.r1eq = rmat[0]
    df.r2eq = rmat[1]
    df.Weq = W
    df.time = tsteps
    df.reached_eq = reached_eq
    return(df)
    
# Given a parameter mesh (param_grid) and tsteps, iterates each row for "tsteps" time stpes
def GetXsteps(param_grid, tsteps):
        
    u1init = param_grid.u1init.values
    u2init = param_grid.u2init.values
    buinit = param_grid.buinit.values
    r1init = param_grid.r1init.values
    r2init = param_grid.r2init.values
    Kvec = param_grid.K.values
    pcvec = param_grid.pc.values
    betavec = param_grid.beta.values
    
    n = len(u2init)
    uvec = [u1init, u2init, buinit]
    xvec = [np.zeros(n),np.zeros(n), np.zeros(n)]
    rvec = [r1init, r2init]
    for i in range(0,tsteps):
        result = NextGen(uvec,xvec,rvec, Kvec,pcvec ,betavec)
        uvec, xvec, rvec, W = result
    # check that reached equilibrium by checking that the previous value = the current value
    
    next_step = NextGen(uvec,xvec,rvec, Kvec,pcvec ,betavec)
    uvec2, xvec2, rvec2, W2 = next_step
    
    # for the allclose to work, they all need to be arrays
    #uvec, xvec, rvec, W = [np.array(item) for item in [uvec, xvec, rvec, W]]
    
                           
    reached_eq = np.allclose(np.array([*uvec[0],*uvec[1],*rvec[0],*rvec[1],*W]), 
                            np.array([*uvec2[0],*uvec2[1],*rvec2[0],*rvec2[1],*W2]),atol=1e-10, rtol = 1e-10)
    
    result = uvec, xvec, rvec, W, reached_eq
    return(result)

# Uses fsolve from scipy.optimize on rows that are taking more than 50000 iterations to reach an equilibrium.
def fsolve_failed_eq(df_fail):
    # take rows that failed to reach equilibrium, use their final states and plug in as initials to fsolve
    def EqSystem(freqs, params):
        [u1,u2,bu,r1,r2, W] = freqs
        K,pc,beta = params
        uvec = [u1,u2,bu]; 
        xvec = [0,0,0]; 
        rvec = [r1,r2]
        uvec, xvec, rvec,W = NextGen(uvec,xvec,rvec, K,pc,beta = beta)
        return_vec = np.array(freqs) - np.array([*uvec,*rvec,W])
        # return_vec of [Wu1 - Wu1func, ... r1 - r1fun, r2 - r2fun]
        return(return_vec)
    def fsolve_rows(row):
        freqs = [row.u1eq, row.u2eq, row.bueq, row.r1eq, row.r2eq, row.Weq]
        params = [row.K, row.pc, row.beta]
        res, infodict, ier,mesg = fsolve(EqSystem, freqs, args=params, full_output=True, xtol = 1e-10)
        row.u1eq, row.u2eq, row.bueq, row.r1eq, row.r2eq, row.Weq = res
        row.reached_eq = ier
        row.time = -1
        return(row)
    new_eq = df_fail.apply(lambda row: fsolve_rows(row), axis = 1)
    # now check these eq... FILL IN
    return(new_eq)

# The system is symmetric over the u1 = u2 line. 
# This function reflects over that line as explained in Numerical Analysis and Eqs. 24-25
def reflect_df(df):
    df2 = df.copy()
    df2.u2init = df.u1init
    df2.u1init = df.u2init
    df2.r1init = df.r2init
    df2.r2init = df.r1init
    df2.u1eq = df.u2eq
    df2.u2eq = df.u1eq
    df2.r1eq = df.r2eq
    df2.r2eq = df.r1eq
    
    df = df.append(df2)
    return(df)

# extract the unique equilibria, since manyy initial points will have iterated to the same equilibrium.
def get_UniqueEquilibria(df,if_save=False):
    df_eq = df.round(6)[(df.reached_eq==1)&(df.iterated==1)].groupby(['K','pc','s','mu','beta','u1eq','u2eq','bueq',
                                                     'r1eq','r2eq','Weq','URstable'], as_index = False)
    df_eq = df_eq['u2init'].count()
    df_eq.rename(columns={'u2init':'NumInitials'}, inplace=True)
    # df_eq.reset_index(inplace=True, drop=True)
    
    df_eq = get_gradients(df_eq)
    if if_save:
        df_eq.to_csv('UniqueEquilibria.csv', index = False)
    return(df_eq)

# Check internal stability for each row
def JstarStable(row):
    # Checks Jstar stability... 1 if stable, 0 if not, -1 if leading eval is 1 (or -1)
    # adds in the absolute value of leading eigenvalue
    
    uvec = [row.u1eq, row.u2eq, row.bueq]
    rvec = [row.r1eq, row.r2eq]
    K = row.K
    W = row.Weq
    beta = row.beta
    Jstar = Jac_UR(uvec, rvec, K,  W, beta)
    evals = np.linalg.eigvals(Jstar).real
    abs_evals = np.abs(evals)
    maxval = max(abs_evals)
    leadval = evals[np.where(np.abs(evals) == maxval)[0][0]]
    row.lambdastar = leadval
    if maxval<1:
        row.URstable = 1.0

    elif maxval ==1:
        row.URstable = -1.0
    else:
        row.URstable = 0.0
    return(row)

# Check external stability by finding C_s
def get_gradients(df):
    df_use = df.copy()
    u1vec = df_use.u1eq
    u2vec = df_use.u2eq
    r1vec = df_use.r1eq
    r2vec = df_use.r2eq
    Wvec = df_use.Weq
    Kvec = df_use.K
    svec = df_use.s
    muvec = df_use.mu
    betavec = df_use.beta
    Csvec = [Grad_s([u1,u2], [r1,r2], W, s, beta,mu) for u1,
             u2,r1,r2,W,s,beta,mu in zip(u1vec,u2vec,r1vec,r2vec,Wvec,svec,betavec, muvec)]
    df_use['C_s'] = Csvec
    return(df_use)




def df_ext_stability_iterate(df):
    # Check stability to allele a (with delta_s > 0 and delta_s < 0)
    
    x_pos_invades = df.apply(lambda row_eq:Check_ext_stability_iterate(row_eq, 0.01, 0), axis = 1) # ds > 0
    x_neg_invades = df.apply(lambda row_eq:Check_ext_stability_iterate(row_eq, -0.01, 0), axis = 1) # ds < 0
    df['x_pos_invades'] = x_pos_invades
    df['x_neg_invades'] = x_neg_invades
    return(df)
    
def Check_ext_stability_iterate(row_eq, ds):
    # for a row of df, perturb and iterate 1000 steps
    # ds tells us if we're perturbing in the x  direction
    # if z(final) - z(initial) > dz or then unstable
    # return 1 if unstable, 0 otherwise
    
    # get parameters
    s, mu, beta, K, pc = row_eq[['s','mu','beta','K','pc']].values.flatten().tolist()
    
    # edge cases 
    
    if s == 0 and ds < 0:
        return(False)
    
    # vectors of values
    dx = 0.01
    uvec,xvec,rvec = Perturb(row_eq) 
 
    
    # calculate dk and d_(pi_C)
    norm = scs.norm(mu,1)
    dk = Kfun(s + ds, norm) - K
    dpc = pcfun(s + ds, norm) - pc
    
    tsteps = 10000
    for i in range(0, tsteps):
        result = NextGen(uvec,xvec,rvec, K,pc ,beta, deltas = [ dk, dpc])
        uvec, xvec, rvec, W = result
    x = sum(xvec)
    return(sum(x>dx)>0)
    
def Perturb(row):
    # After the [u1,u2,bu,r1,r2] eq is perturbed with the addition of the a allele, get new frequencies
    # perturb by a magnitude of 0.01... so |dr1| = |dr2| = |du| = 0.01, and  |dx|  = 0.01
    u1eq, u2eq, bueq, r1eq, r2eq = row[['u1eq','u2eq','bueq', 'r1eq','r2eq']].values.flatten().tolist()
    
    uvec = [u1eq, u2eq, bueq]
    rvec = [r1eq, r2eq]
    #overall change magnitude
    dz = 0.01
    
    # change u
    du = -dz
    du1vec = -du*np.array([0.015, 0.29, 0.49, 0.9])
    du2vec = -du* np.array([0.11, 0.49, 0.51, 0.05])
    dbuvec = du - du1vec - du2vec
    # get new post-perturb vectors
    for i in range(0,len(du1vec)):
        duvec = [du1vec[i],du2vec[i],dbuvec[i]]
        [du1vec[i],du2vec[i],dbuvec[i]] = Perturb_EdgeCase(uvec,duvec)
    
    # change r
    
    dr = 0.01
    dr1vec = dr*np.array([1,0.5,-1,0.2])
    check_r1_0 = rvec[0] + dr1vec < 0 
    check_r1_1 = rvec[0] + dr1vec> 1
    check_r1 = check_r1_0 + check_r1_1
    dr1vec[check_r1] = - dr1vec[check_r1]
    
    dr2vec = dr*np.array([0.9,-0.3,0.5,-0.8])
    check_r2_0 = rvec[1] + dr2vec < 0 
    check_r2_1 = rvec[1] + dr2vec> 1
    check_r2 = check_r2_0 + check_r2_1
    dr2vec[check_r2] = - dr2vec[check_r2]
    
    
    # no need to check if valid
    
    
    dz1vec = np.array([0.05, 0.48, 0.9, 0.3])*dz
    dz2vec = np.array([0.1, 0.5, 0.04, 0.6])*dz
    dbzvec = dz - dz1vec - dz2vec

    u1 = uvec[0] + du1vec
    u2 = uvec[1] + du2vec
    bu = uvec[2] + dbuvec
    r1 = rvec[0] + dr1vec
    r2 = rvec[1] + dr2vec
    
    
    return([u1,u2,bu],[dz1vec,dz2vec,dbzvec],[r1,r2])

def Perturb_EdgeCase(uvec,duvec):
    #recursively checks for edge cases so i don't get an invalid frequency. Adjusts duvec if needed
    
    # make sure using numpy arrays
    du = sum(duvec)
    uvec = np.array(uvec); duvec = np.array(duvec);
    # find locations of edge cases
    edge_bool = uvec + duvec <= 0
    
    n = sum(edge_bool)
    if n>0:
        duvec[edge_bool] = -uvec[edge_bool] +0.00001 # so not at exactly 0
        du_remain = du - sum(duvec)
        duvec[~edge_bool] = duvec[~edge_bool] + (1/np.float64(3-n))*du_remain
        
        # make sure that we didn't cause a different frequency to be negative:
        return(Perturb_EdgeCase(uvec,duvec))

    else:
        return(duvec)



