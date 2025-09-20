import cvxpy as cp
import numpy as np

def sum_neigh(x, x_old, x_neig):
    result=0
    for i in range(x_neig.shape[1]):
        result = result + cp.norm(x-(x_old+x_neig[:, i])/2.0, 2)**2
    return result

def loss_beta(ei, xi, betai, rho, vi):
    return rho/2.0*cp.norm(ei-betai-cp.multiply(vi, xi), 2)**2

def loss_e(A, Yi, e):
    return cp.norm(A@e-Yi, 2)**2

def regularizer(x):
    return cp.norm(x, 1)

###############################################################
def objective_e(A, Yi, ei, xi, betai, rho, gamma3, taoi, vi):
    return loss_e(A, Yi, ei) + rho/2.0*cp.norm(ei-betai-cp.multiply(vi, xi), 2)**2 +taoi.T@(ei-cp.multiply(vi, xi)-betai)+gamma3*regularizer(ei)

def objective_beta(ei, xi, betai, rho, taoi, gamma2, vi):
    return loss_beta(ei, xi, betai, rho, vi) + taoi.T@(ei-cp.multiply(vi, xi)-betai) +gamma2*regularizer(betai)

def objective_x(ei, xx, xi_old, x_neig, betai, pi, c, gamma1, taoi, rho, vi):
    return rho/2.0*cp.norm(ei-betai-cp.multiply(vi, xx), 2)**2+ xx.T@(pi-np.multiply(taoi, vi.T)) + c * sum_neigh(xx, xi_old, x_neig) + gamma1*regularizer(xx)


##############################################################
def update_e_JSM1(A, Yi, xi, betai, rho, gamma3, taoi, vi):
    e = cp.Variable(xi.shape)
    problem = cp.Problem(cp.Minimize(objective_e(A, Yi, e, xi, betai, rho, gamma3, taoi, vi)))
    problem.solve(solver='ECOS') #verbose=True
    e_value = e.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return e_value


##############################################################
def update_beta_JSM1(ei, xi, rho, taoi, gamma2, vi):
    beta = cp.Variable(xi.shape)
    problem = cp.Problem(cp.Minimize(objective_beta(ei, xi, beta, rho, taoi, gamma2, vi)))
    problem.solve(solver='ECOS') #verbose=True
    beta_value = beta.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return beta_value


##############################################################
def update_x_JSM1(ei, xi_old, x_neig, betai, pi, c, gamma1, taoi, rho, vi):
    xx = cp.Variable(xi_old.shape)
    problem = cp.Problem(cp.Minimize(objective_x(ei, xx, xi_old, x_neig, betai, pi, c, gamma1, taoi, rho, vi)))
    problem.solve(solver='ECOS') #verbose=True
    xx_value = xx.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return xx_value



