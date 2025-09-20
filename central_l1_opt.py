import numpy as np
import copy
def shrinkage(x, a):
    result = max(0, x - a) - max(0, -x - a)
    return result


def vect_shrinkage(x,a ):
    num=len(x)
    result = np.zeros(num)
    for i in range(num):
        result[i] = max(0, x[i] - a) - max(0, -x[i] - a)
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


def objective(A, Y, X, rho):
    m=np.size(A, 0)
    # print A.shape, b.shape, z
    # np.dot(-A, z[1: len(z)]).astype('float32')
    dot1 = np.dot(A, X).astype('float32')
    # dot2 = np.dot(-b, z[0]).astype('float32')
    return 1.0 * np.linalg.norm(Y-dot1, ord=2)+1.0 * rho * np.linalg.norm(X, ord=1)
    # return mu*m*np.linalg.norm(z[1: len(z)], ord=1)


def f_grad(z_old, A, Y):
    # print np.size(np.dot(A, z_old)), np.size((np.dot(A, z_old) - Y))
    grad = 2*np.dot(A.T, (np.squeeze(np.dot(A,z_old))-Y))
    return grad


def central_l1_FISTA(A, Y, X, jud):
    max_iter=100000
    rho=1
    t=1
    eta=1.2
    lam=100
    #pgr=1.0
    z = np.zeros((np.size(X),))
    pgr=1.0*np.linalg.norm(z-X, 2)/(rho*np.sqrt(len(X)))

    for l in range(max_iter):
        l=l+1
        z_old=copy.deepcopy(z)
        x_old=copy.deepcopy(X)
        #print neig_sum(z_old, x, X_neig).shape
        temp= f_grad(z_old, A, Y)
        # print np.size(temp)
        # print temp.shape
        for i in range(len(X)):
            # print z_old[i], rho * temp[i], lam * rho
            X[i]= shrinkage((z_old[i] - 1.0/t * temp[i]), lam*t)
                # max(-0.9999, min(0.9999, shrinkage(z_old[i]- rho*temp[i], lam*rho)))
            # x[i] = shrinkage(z_old[i] - rho * temp[i], lam * rho)
        t_new=1.0*(1+np.sqrt(1+4*(t**2)))/2
        z = X + 1.0*(t-1)/(t_new)*(X-x_old)
        t=copy.deepcopy(t_new)
        pgr = 1.0 * np.linalg.norm(z_old - X, 2) / (rho * np.sqrt(len(X)))
        print("%10.20f\t" % pgr)
        if pgr < jud:
            break
    obj = objective(A, Y, X)
    return X, obj

