import numpy as np
from scipy import optimize
import scipy
import copy

# def ele_multi(a,b):
#     result=np.zeros((len(a),1))
#     for i in range(len(a)):
#         result[i]=a[i]*b[i]
#     return result


def vec_max(a, vector):
    result=np.zeros((len(vector)))
    for i in range(len(vector)):
        result[i]=max(a, vector[i])
    return result


def vec_min(a, vector):
    result=np.zeros((len(vector)))
    for i in range(len(vector)):
        result[i]=min(a, vector[i])
    return result


def shrinkage(a,kappa):
    result = max(0, a - kappa) - max(0, -a - kappa)
    return result


def vect_shrinkage(a, kappa):
    num=len(a)
    result = np.zeros(num)
    for i in range(num):
        result[i] = max(0, a[i] - kappa) - max(0, -a[i] - kappa)
    return result


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


def f_grad(x, C, b):
    # e2CX = np.exp(np.dot(C, x))
    # e2CXr = e2CX/(e2CX+1)
    grad = 2*np.dot(C.T, (np.squeeze(np.dot(C,x))-b))
    # print np.squeeze(np.dot(C.T, b)).shape
    return grad


def neig_sum(x, x_old, X_neig, num_neig):
    sum = np.zeros((len(x),))
    for i in range(num_neig):
        sum = sum + (x - ((x_old + X_neig[:, i]) / 2))
    sum=1.0 * sum/num_neig
    return sum

'''
def in_neig_sum(x, X_neig):
    sum=np.zeros((len(x),))
    num_neig=np.size(X_neig,1)
    for i in range(num_neig):
        sum = sum + (x + X_neig[:, i])
    return sum
'''

def objective(A,b,z):
    m=np.size(A, 0)
    # print A.shape, b.shape, z
    # np.dot(-A, z[1: len(z)]).astype('float32')
    dot1 = np.dot(A, z).astype('float32')
    # dot2 = np.dot(-b, z[0]).astype('float32')
    return sum(np.log(np.exp(dot1)+1))-np.dot(b.T, dot1)+1.0*np.linalg.norm(z, ord=1)
    # return mu*m*np.linalg.norm(z[1: len(z)], ord=1)


def FISTA_update(x_oldd, C, b, p, c, K, X_neig, N, jud):
    x=np.copy(x_oldd)
    max_iter=100000
    rho=0.01
    a=1
    lam=0.1
    #pgr=1.0
    z = np.zeros((np.size(x)))
    pgr=1.0*np.linalg.norm(z-x, 2)/(rho*np.sqrt(K+1))
    num_neig=np.size(X_neig, 1)

    for l in range(max_iter):
        l=l+1
        z_old=copy.deepcopy(z)
        x_old = copy.deepcopy(x)
        #print neig_sum(z_old, x, X_neig).shape
        temp= f_grad(z_old, C, b) + p + 2 * c * neig_sum(z_old, x_old, X_neig, num_neig) # + 2*(z_old-x_old)/eita
        #print temp.shape
        for i in range(len(x)):
            x[i] = shrinkage(z_old[i] - rho * temp[i], lam * rho / N)
                # max(-0.9999, min(0.9999, shrinkage(z_old[i] - rho*temp[i], lam*rho/N)))
            # x[i] = shrinkage(z_old[i] - rho * temp[i], lam * rho / N)
        z = x + 1.0*(l-1)/(l+2)*(x-x_old)
        pgr = 1.0 * np.linalg.norm(z_old - x, 2) / (rho * np.sqrt(K))
        if pgr < jud:
            break
    return x


def FISTA_update2(x, C, b, p, c, K, X_neig, N, jud, prob, num_neig):
    max_iter=100000
    rho=0.01
    a=1
    lam=0.1
    #pgr=1.0
    z = np.zeros((np.size(x)))
    pgr=1.0*np.linalg.norm(z-x, 2)/(rho*np.sqrt(K+1))

    for l in range(max_iter):
        l=l+1
        z_old=copy.deepcopy(z)
        x_old = copy.deepcopy(x)
        #print neig_sum(z_old, x, X_neig).shape
        temp= l2_log_grad(z_old, C, b) + p + 2 * c * neig_sum(z_old, x_old, X_neig, prob, num_neig)
        #print temp.shape
        for i in range(len(x)):
            x[i] = max(-0.9999, min(0.9999, shrinkage(z_old[i] - rho*temp[i], lam*rho/N)))
            # x[i] = shrinkage(z_old[i] - rho * temp[i], lam * rho / N)
        z = x + 1.0*(l-1)/(l+2)*(x-x_old)
        pgr = 1.0 * np.linalg.norm(z_old - x, 2) / (rho * np.sqrt(K))
        if pgr < jud:
            break
    return x



def update_2(x, C, b, p, c, X_neig, eita, prob, num_neig):
    result=np.zeros(np.size(x))
    grad_h=l2_log_grad(x, C, b) + p + 2*c*neig_sum(x, x, X_neig, prob, num_neig)
    for i in range(len(x)):
        result[i]=max(-0.9999, min(0.9999, shrinkage(x[i]-eita*grad_h[i]/2.0, 2.0/eita)))
    return result

'''
def inexact_FISTA_update(x, C, b, p, c, X_neig, N):
    lam=0.1
    beta=1.2
    num_neig = np.size(X_neig, 1)
    gamma = beta+2*c*num_neig
    x_old=copy.deepcopy(x)

    operator = 1.0/gamma * (beta*x_old - l2_log_grad(x_old, C, b) - p + c*in_neig_sum(x_old, X_neig))
    # return vec_max(0, vec_min(1, vect_shrinkage(operator, 1.0/gamma* lam/N)))
    return vect_shrinkage(operator, 1.0/gamma)
'''