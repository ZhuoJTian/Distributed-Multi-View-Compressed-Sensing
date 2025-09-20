import numpy as np

def sigmoid(inX):
    M = inX.shape[0]
    T = inX.shape[1]
    result=np.ones((M, T))
    for i in range(M):
        for j in range(T):
            if inX[i,j]>=0:
                result[i,j]=1.0/(1+np.exp(-1.0*inX[i,j]))
            else:
                result[i,j]=np.exp(inX[i,j])/(1+np.exp(inX[i,j]))
    return result

def sigmoid_vector(inX):
    result=np.ones(inX.shape[0])
    for i in range(inX.shape[0]):
        if inX[i]>=0:
            result[i]=1.0/(1+np.exp(-1.0*inX[i]))
        else:
            result[i]=np.exp(inX[i])/(1+np.exp(inX[i]))
    return result


def log_likelyhood(inX):
    M = inX.shape[0]
    T = inX.shape[1]
    result = np.ones((M, T))
    for i in range(M):
        for j in range(T):
            result[i,j]=-1.0*np.log(1.0/inX[i,j]-1.0)
    return result


def PDF_gaussian(x, u, v):
    result=(1.0/np.sqrt(2*np.pi*v))*np.exp(-(x-u)**2/(2*v))
    return result


def MP_inference_v_iter(Y, B, x, sigma, Max_iter, lamb):  # the variance of noise is known
    M=B.shape[0]
    T=B.shape[1]
    index=list(np.where(abs(x)>=0.00001)[0])
    L_km=np.zeros((M, T))
    L_mk = np.zeros((M, T))
    P_mk = np.zeros((M, T))
    P_km=sigmoid(L_km)
    U=np.zeros((M, T))
    V=np.zeros((M, T))
    iter=0
    lk=1.0/(1+np.exp(-1.0*lamb))
    lk_hat=np.zeros(T)
    p_hat = np.zeros(T)
    v_est=np.zeros(T)
    while iter<Max_iter:
        # update of u and v
        for m in range(M):
            for k in index:
                sum1=0
                sum2=0
                # sum1=np.sum(np.multiply(B[m, :], P_km[m, :]))-B[m, k]*P_km[m,k]
                for j in index:
                    if j!=k:
                        sum1=sum1+B[m, j]*P_km[m,j]
                        sum2=sum2+(B[m, j]**2)*P_km[m, j]*(1-P_km[m, j])
                sum2=sum2+sigma
                U[m, k]=sum1
                V[m, k]=sum2

        # update of pmk and lmk
        for m in range(M):
            for k in index:
                P_mk[m, k]=1.0/(1+PDF_gaussian(Y[m] ,U[m,k], V[m,k])
                                    /PDF_gaussian(Y[m], U[m,k]+B[m, k], V[m,k]))
        L_mk=log_likelyhood(P_mk)

        # update of lkm
        for k in index:
            for m in range(M):
                sum=lk
                for j in range(M):
                    if j!=m:
                        sum=sum+L_mk[j, k]
                L_km[m, k]=sum
        P_km = sigmoid(L_km)

        # update of v
        for k in index:
            sum=lk
            for m in range(M):
                sum=sum+L_mk[m, k]
            lk_hat[k] = sum
            if lk_hat[k]>=0:
                v_est[k]=1
            else:
                v_est[k] = 0
        p_hat=sigmoid_vector(lk_hat)

        # compute the mse convergence2
        mse= np.linalg.norm(np.multiply(p_hat, 1 - p_hat), 2)/T
        if mse<=0.2:
            return v_est
        else:
            iter=iter+1

    return v_est


def AMP_inference_v_iter(Y, B, sigma, Max_iter, lamb):
    M = B.shape[0]
    T = B.shape[1]



