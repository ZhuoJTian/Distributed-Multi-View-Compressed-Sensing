import cvxpy as cp
import numpy as np

def sum_neigh(x, x_old, x_neig):
    result=0
    for i in range(x_neig.shape[1]):
        result = result + cp.norm(x-(x_old+x_neig[:, i])/2.0, 2)**2
    return result


def loss_fn(A, Yi, x):
    return cp.norm(A@x-Yi, 2)**2


def loss_fn_beta(A, Yi, x, beta):
    return cp.norm(A@x-(Yi-beta), 2)**2


def penalty_fn(M, x, T):
    a=x-0.5*np.ones(T)
    return cp.norm(M@(x-0.5*np.ones(T)), 2)**2


def regularizer(x):
    return cp.norm(x, 1)

###############################################################
def objective_fn_admm(A, Y, x, gamma, p, c, x_old, x_neig):
    return loss_fn(A, Y, x)+ x.T@p + c * sum_neigh(x, x_old, x_neig) + gamma*regularizer(x)


def objective_fn_admm_beta(A, Y, x, beta, gamma, p, c, x_old, x_neig):
    return loss_fn_beta(A, Y, x, beta)+ x.T@p + c * sum_neigh(x, x_old, x_neig) + gamma*regularizer(x)


def objective_beta(A, Y, x, beta, gamma):
    return loss_fn_beta(A, Y, x, beta) + gamma*regularizer(beta)


def objective_fn_admm_new(A, Y, x, gamma, p, c, x_old, x_neig, eita_k):
    return loss_fn(A, Y, x)+ x.T@p + c * sum_neigh(x, x_old, x_neig) + \
           cp.norm(x-x_old)**2/eita_k+ gamma*regularizer(x)


def objective_fn(A, Y, x, gamma):
    return loss_fn(A, Y, x)+ gamma*regularizer(x)


def objective_fn_v(B, Y, x, alpha, T):
    return cp.norm(B@x-Y, 2)**2-alpha*regularizer(x-0.5*np.ones((T, 1)))


##############################################################

def mse(A, Y, x):
    return (1.0/A.shape[0])*loss_fn(A, Y, x).value
################################################################

def update_x_ADMM(Yi, Ai, vi, p, c, x_old, x_neig, gamma):
    Ai_trans=np.dot(Ai, np.diag(vi))
    beta = cp.Variable(x_old.shape)
    problem = cp.Problem(cp.Minimize(objective_fn_admm(Ai_trans, Yi, beta, gamma, p, c, x_old, x_neig)))
    problem.solve(solver='ECOS') #verbose=True
    mse_opt = mse(Ai_trans, Yi, beta)
    beta_value = beta.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return beta_value, mse_opt


def update_x_ADMM_beta(Yi, Ai, betai, p, c, x_old, x_neig, gamma1):
    xx = cp.Variable(x_old.shape)
    problem = cp.Problem(cp.Minimize(objective_fn_admm_beta(Ai, Yi, xx, betai, gamma1, p, c, x_old, x_neig)))
    problem.solve(solver='ECOS') #verbose=True
    xx_value = xx.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return xx_value


def update_x_ADMM_new(Yi, Ai, vi, p, c, x_old, x_neig, gamma, eita_k):
    Ai_trans=np.dot(Ai, np.diag(vi))
    beta = cp.Variable(x_old.shape)
    problem = cp.Problem(cp.Minimize(objective_fn_admm_new(Ai_trans, Yi, beta, gamma, p, c, x_old, x_neig, eita_k)))
    problem.solve(solver='ECOS') #verbose=True
    mse_opt = mse(Ai_trans, Yi, beta)
    beta_value = beta.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return beta_value, mse_opt


def update_x(Yi, Ai, gamma):
    T=Ai.shape[1]
    beta = cp.Variable(T)
    problem = cp.Problem(cp.Minimize(objective_fn(Ai, Yi, beta, gamma)))
    problem.solve(solver='ECOS')
    beta_value = beta.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return beta_value

def update_x_standard(Yi, Ai, p, c, x_old, x_neig, gamma):
    T=Ai.shape[1]
    beta = cp.Variable(T)
    problem = cp.Problem(cp.Minimize(objective_fn_admm(Ai, Yi, beta, gamma, p, c, x_old, x_neig)))
    problem.solve(solver='ECOS')
    beta_value = beta.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return beta_value

##############################################################
def update_beta(Yi, Ai, beta_old, x_e, gamma2):
    xx = cp.Variable(beta_old.shape)
    problem = cp.Problem(cp.Minimize(objective_beta(Ai, Yi, x_e, xx, gamma2)))
    problem.solve(solver='ECOS') #verbose=True
    xx_value = xx.value
    # train_errors = 1.0 * np.linalg.norm((beta_value - x), 2) / np.linalg.norm((x), 2)
    return xx_value

#################################################################
def update_v_l1(Yi, Bi, alpha):
    T=Bi.shape[1]
    beta = cp.Variable((T,1))
    problem = cp.Problem(cp.Minimize(objective_fn_v(Bi, np.reshape(Yi, (Bi.shape[0], 1)), beta, alpha, T)))
    problem.solve(solver='ECOS')
    # mse_opt = mse_v(Bi, np.reshape(Yi, (Bi.shape[0], 1)), beta, alpha, gamma2, T)
    beta_value = beta.value
    v_temp = np.reshape(beta_value, (T,))
    return v_temp
