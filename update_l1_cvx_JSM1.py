import cvxpy as cp
import numpy as np

def sum_neigh(x, x_old, x_neig):
    result=0
    for i in range(x_neig.shape[1]):
        result = result + cp.norm(x-(x_old+x_neig[:, i])/2.0, 2)**2
    return result

def loss_beta(ei, xi, betai, rho):
    return rho/2.0*cp.norm(ei-betai-xi, 2)**2

def loss_e(A, Yi, e):
    return cp.norm(A@e-Yi, 2)**2

def regularizer(x):
    return cp.norm(x, 1)

###############################################################
def objective_e(A, Yi, ei, xi, betai, rho, gamma3, taoi):
    return loss_e(A, Yi, ei) + rho/2.0*cp.norm(ei-betai-xi, 2)**2 +taoi.T@(ei-xi-betai)+gamma3*regularizer(ei)

def objective_beta(ei, xi, betai, rho, taoi, gamma2):
    return loss_beta(ei, xi, betai, rho) + taoi.T@(ei-xi-betai) +gamma2*regularizer(betai)

def objective_x(ei, xx, xi_old, x_neig, betai, pi, c, gamma1, taoi, rho):
    return rho/2.0*cp.norm(ei-betai-xx, 2)**2+ xx.T@(pi-taoi) + c * sum_neigh(xx, xi_old, x_neig) + gamma1*regularizer(xx)


##############################################################
def update_e_JSM1(A, Yi, xi, betai, rho, gamma3, taoi):
    e = cp.Variable(xi.shape)
    problem = cp.Problem(cp.Minimize(objective_e(A, Yi, e, xi, betai, rho, gamma3, taoi)))
    problem.solve(solver='ECOS') #verbose=True
    e_value = e.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return e_value


##############################################################
def update_beta_JSM1(ei, xi, rho, taoi, gamma2):
    beta = cp.Variable(xi.shape)
    problem = cp.Problem(cp.Minimize(objective_beta(ei, xi, beta, rho, taoi, gamma2)))
    problem.solve(solver='ECOS') #verbose=True
    beta_value = beta.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return beta_value


##############################################################
def update_x_JSM1(ei, xi_old, x_neig, betai, pi, c, gamma1, taoi, rho):
    xx = cp.Variable(xi_old.shape)
    problem = cp.Problem(cp.Minimize(objective_x(ei, xx, xi_old, x_neig, betai, pi, c, gamma1, taoi, rho)))
    problem.solve(solver='ECOS') #verbose=True
    xx_value = xx.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return xx_value



