import numpy as np
from scipy import optimize
import scipy

# def ele_multi(a,b):
#     result=np.zeros((len(a),1))
#     for i in range(len(a)):
#         result[i]=a[i]*b[i]
#     return result

def sigmoid(inX):
    result=np.ones((np.size(inX)))
    for i in range(len(inX)):
        if inX[i]>=0:
            result[i]=1.0/(1+np.exp(-inX[i]))
        else:
            result[i]=np.exp(inX[i])/(1+np.exp(inX[i]))
        if result[i]==0:
            result[i]=np.exp(-700)
    return result


def neig_sum_squre(x, x_old, X_neig, prob, num_neig):
    sum=0
    if prob==0:
        return 0
    elif num_neig==1:
        sum = sum + np.dot((x - ((x_old + X_neig) / 2)).T, (x - ((x_old + X_neig) / 2))) / prob
    else:
        # print np.dot((x - ((x_old + X_neig[:, 0]) / 2)).T, (x - ((x_old + X_neig[:, 0]) / 2)))
        for i in range(num_neig):
            sum = sum + np.dot((x - ((x_old + X_neig[:, i]) / 2)).T, (x - ((x_old + X_neig[:, i]) / 2)))
    return sum


def neig_sum(x, x_old, X_neig, prob, num_neig):
    sum=np.zeros((len(x),))
    if prob==0:
        return 0
    elif num_neig==1:
        sum = sum + (x - ((x_old + X_neig) / 2)) / prob
    else:
        for i in range(num_neig):
            sum = sum + x - ((x_old + X_neig[:, i]) / 2)
    return sum


def l2_log(x, C, b, p, X_neig, rho, x_old, co, prob, N, num_neig):
    # expo=np.exp(np.dot(C, x))
    # sum(np.log(1+np.exp(np.dot(C, x)))) +
    # result=sum(np.log(1+np.exp(np.dot(C, x)))) - np.dot(b.T, np.dot(C, x)) + rho*np.dot((x - z + u).T, (x - z + u))/2
    result = sum(-1.0*np.log(sigmoid(-1.0*np.dot(C, x)))) - np.dot(b.T, np.dot(C, x)) + rho * np.dot(x.T,x)/N + \
             np.dot(x.T, p) + co*neig_sum_squre(x, x_old, X_neig, prob, num_neig)
    # print np.dot(b.T, np.dot(C, x)).shape
    return result


def l2_log_grad(x, C, b, p, X_neig, rho, x_old, co, prob, N, num_neig):
    # e2CX = np.exp(np.dot(C, x))
    # e2CXr = e2CX/(e2CX+1)
    e2CXr = sigmoid(np.dot(C, x))
    # grad = np.dot(C.T, e2CXr) - np.squeeze(np.dot(C.T, b)) + rho * (x - z + u)
    grad =np.dot(C.T, e2CXr)- np.squeeze(np.dot(C.T, b)) + 2* rho * x/N + p + 2*co*neig_sum(x, x_old, X_neig, prob, num_neig)
    # print np.squeeze(np.dot(C.T, b)).shape
    return grad


def bfgs_update(x, C, b, p, X_neig, rho, x_old, co, prob, N, num_neig):
    # solve the x update
    # minimize [ -logistic(x_i) + (rho/2)||x_i - z^k + u^k||^2 ]
    # via L-BFGS
    m = np.size(C, 0)
    n = np.size(C, 1)
    x_s, f_s, d_s = optimize.fmin_l_bfgs_b(func=l2_log, x0=x, fprime=l2_log_grad, args=(C, b, p, X_neig, rho, x_old, co, prob, N, num_neig), iprint=0)
    # print (l2_log_grad(x, C, u, z, N, rho)).shape
    num_iters = d_s['nit']
    # print x_s.shape
    return x_s