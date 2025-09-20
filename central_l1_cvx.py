import cvxpy as cp
import numpy as np

def loss_fn(A, Y, x):
    return cp.norm2(A@x-Y)**2

def regularizer(x):
    return cp.norm1(x)

def objective_fn(A, Y, x, lam):
    return loss_fn(A, Y, x)+lam*regularizer(x)

def mse(A, Y, x):
    return (1.0/A.shape[0])*loss_fn(A, Y, x).value

def central_cvx(A, Y, x, lam):
    beta = cp.Variable(A.shape[1])
    problem = cp.Problem(cp.Minimize(objective_fn(A, Y, beta, lam)))
    problem.solve()
    obj_opt=objective_fn(A, Y, beta, lam).value
    beta_value=beta.value
    train_errors=1.0*np.linalg.norm((beta_value-x), 2)/np.linalg.norm((x), 2)
    return beta_value, obj_opt, train_errors
