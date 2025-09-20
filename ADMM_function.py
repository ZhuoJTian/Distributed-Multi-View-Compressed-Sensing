# coding=utf-8
import numpy as np
import update_l1_cvx as upd
import update_l1_cvx_JSM1 as updJSM1
import update_l1_cvx_ErrCor as updEC
import inference_v as iv
import copy
import matplotlib.pyplot as plt

def sum_neigh(x_old, x_neig):
    result=0
    for i in range(x_neig.shape[1]):
        result = result + (x_old - x_neig[:, i])
    return result

def sigmoid(inX, tau):
    result=np.ones((np.size(inX)))
    for i in range(len(inX)):
        if inX[i]>=0:
            result[i]=1.0/(1+np.exp(-1.0*tau*inX[i]))
        else:
            result[i]=np.exp(tau*inX[i])/(1+np.exp(tau*inX[i]))
    return result


def objective(An, Yn, Xn_e):
    m=np.size(An, 0)
    dot1 = np.dot(An, Xn_e)
    return 1.0 * np.linalg.norm(Yn-dot1, ord=2)**2 + 1.0 * np.linalg.norm(Xn_e, ord=1)


def loss_fn(A, Yi, x):
    dot1 = np.dot(A, x)
    return 1.0 * np.linalg.norm(Yi-dot1, ord=2)**2


def mean_objective(A, Y, X, N, M):
    obj=np.zeros(N)
    for i in range(N):
        An = A[i * M: (i + 1) * M, :]
        # print CK.shape
        Yn = Y[i * M: (i + 1) * M]
        dot= np.dot(An, X[:, i])
        obj[i] = 1.0/N * np.linalg.norm(Yn-dot, ord=2)**2 + 1.0/N * np.linalg.norm(X[:, i], ord=1)
    #print obj
    return sum(obj)


def shrinkage(x, a):
    result=np.zeros(len(x))
    for i in range(len(x)):
        result[i]=max(0, x[i]-a)-max(0, -x[i]-a)
    return result


def map_v_soft(v, x_neigh, x, v_neig):
    T=v.shape[0]
    result = np.zeros(T)
    P=np.ones(T)
    KK=np.where(abs(x)>=0.001)[0].shape[0]
    xx = np.hstack((x_neigh, np.reshape(x, [v.shape[0], 1])))
    '''
    major_value = np.zeros(v.shape)
    num_vote = xx.shape[1]
    v_index=np.where(v_real==1)[0]
    for i in range(result.shape[0]):
        num_major0 = np.shape(np.where(xx[i, :]==0)[0])[0]
        num_major1 = np.shape(np.where(xx[i, :]==1)[0])[0]
        if num_major0==num_major1:
            major_value[i] = 2
        elif num_major0>num_major1:
            major_value[i] = 0
        elif num_major0<num_major1:
            major_value[i] = 1
    
    list_uncertain = []
    for i in range(T):
        if v[i] <= 0.3:
            result[i] = 0
        elif v[i] >= 0.7:
            result[i] = 1
        else:
            list_uncertain.append(i)
            major=np.where(v_neig[i, :]==1)[0]
            if major.shape[0]==0:
                if abs(np.average(x_neigh[i, :]))<abs(x[i]):
                    result[i]=1
                else:
                    result[i]=0
            else:
                thres=(np.average(x_neigh[i, major])-x[i])**2
                if (thres>1.0/KK*(r**2)) & (abs(np.average(x_neigh[i, major]))>abs(x[i])):
                    result[i] = 0
                else:
                    result[i] = 1
    '''
    list_uncertain=[]
    num_one=0
    for i in range(T):
        if v[i] <= 0.2:
            result[i] = 0
        elif v[i] >= 0.8:
            result[i] = 1
            num_one=num_one+1
        else:
            '''
            list_uncertain.append(i)
            major = np.where(v_neig[i, :] == 1)[0]
            if major.shape[0] == 0:
                if abs(np.average(x_neigh[i, :])) < abs(x[i]):
                    result[i] = 1
                else:
                    result[i] = 0
            else:
                thres = (np.average(x_neigh[i, major]) - x[i]) ** 2
                if (thres > 1.0 / KK * (r ** 2)) & (abs(np.average(x_neigh[i, major])) > abs(x[i])):
                    result[i] = 0
                else:
                    result[i] = 1
            # result[i]=np.random.choice([0, 1], size=1, p=[1-v[i], v[i]])
            '''
            list_uncertain.append(i)
            major=np.where(v_neig[i,:]==1)[0]
            if major.shape[0]==0:
                result[i]=0  # v[i]*0.1
            else:
                '''
                if abs(x[i])<abs(np.average(x_neigh[i, major])):
                    thres=((np.average(x_neigh[i, major])-x[i]) / (1.0*x[i]))**2
                # tt=c*(x_old[i]-x[i])
                # P[i]=v[i]*(r**2)/(thres*KK)
                    P[i] = v[i] / thres
                else:
                    thres = ((np.average(x_neigh[i, major]) - x[i]) / (1.0 * np.average(x_neigh[i, major]))) ** 2
                    # tt=c*(x_old[i]-x[i])
                    # P[i]=v[i]*(r**2)/(thres*KK)
                    P[i] = v[i] * thres'''
                # print(i, v[i], thres, r, KK, tt**2, P[i])
                P[i] = v[i]* abs(x[i]) / abs(np.average(x_neigh[i, major]))
                # if abs(x[i])<abs(np.average(x_neigh[i, major])):
                #    P[i]=0
                # else:
                #    P[i]=1
            if P[i]<=0:
                result[i]=0
            elif P[i]>=1:
                result[i]=1
            else:
                result[i]=np.random.choice([0, 1], size=1, p=[1-P[i], P[i]])
    '''
    rr=np.ones(T)
    for i in range(T):
        rr[i]=((np.average(x_neigh[i, :])-x[i])/x[i])**2
    rr_s=sorted(rr)
    thres=rr_s
    ll=[]
    for i in range(T):
        if v[i] <= 0.3:
            result[i] = 0
        elif v[i] >= 0.7:
            result[i] = 1
        else:
            result[i] = np.random.choice([0, 1], size=1, p=[1 - v[i], v[i]])'''
    print(num_one, np.where(result==1)[0].shape[0])  #list_uncertain,
    return result, len(list_uncertain), num_one


def map_v_hard1(v, ii):  #, x_neigh, x, k, com_mse, v_real
    T=v.shape[0]
    result = np.zeros(T)
    if ii==0:
        for i in range(T):
            if v[i]<=0:
                result[i]=0
            elif v[i]>=1:
                result[i]=1
            else:
                result[i] = v[i]
    else:
        for i in range(T):
            if v[i]<=0.5:
                result[i]=0
            elif v[i]>0.5:
                result[i]=1
    return result


def map_v_hard2(v):  #, x_neigh, x, k, com_mse, v_real
    T=v.shape[0]
    result = np.zeros(T)
    for i in range(T):
        if v[i]<=0:
            result[i]=0
        elif v[i]>=1:
            result[i]=1
        else:
            result[i] = v[i]
    return result


def map_v_hard3(v, ii, x):  #, x_neigh, x, k, com_mse, v_real
    T=v.shape[0]
    result = np.zeros(T)
    if ii==0:
        for i in range(T):
            if (v[i]<=0)|(abs(x[i])<=0.0001):
                result[i]=0
            elif v[i]>=1:
                result[i]=1
            else:
                result[i] = v[i]
    else:
        for i in range(T):
            if v[i]<=0.5:
                result[i]=0
            elif v[i]>0.5:
                result[i]=1
    return result


def grad_w(Y_i, B_i, tau, w_old):
    result = np.zeros(len(w_old))
    v_old = sigmoid(w_old, tau)
    temp = 2.0*np.dot(B_i.T, (np.dot(B_i, v_old)-Y_i))
    for i in range(len(w_old)):
        if w_old[i]>=0:
            result[i] = temp[i] * tau * np.exp(-1.0 * tau*w_old[i])/((1.0 + np.exp(-1.0*tau*w_old[i]))**2)
        else:
            result[i] = temp[i] * tau * np.exp(tau * w_old[i]) / ((1.0 + np.exp(tau * w_old[i])) ** 2)
    return result


######################################################################################################
def update_v(alpha, Y_i, B_i, tau, w_old, v_old):
    gradient = grad_w(Y_i, B_i, tau, w_old)
    w_oo = copy.deepcopy(w_old)
    iternum=0
    while not all(abs(gradient)<=1e-3):
        w_oo = w_oo - alpha * gradient
        gradient = grad_w(Y_i, B_i, tau, w_oo)
        v_oo = sigmoid(w_oo, tau)
        f=1.0 * np.linalg.norm(Y_i-np.dot(B_i, v_oo), ord=2)
        iternum=iternum+1
        if iternum==30:
            break
    return w_oo


def baseline_int(A, Y, X, V, N, Adjacent_Matrix, gamma):
    m = np.size(A, 0)
    T = np.size(A, 1)
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    C = A
    com_mse=np.zeros(N)
    loc_mse=np.zeros(N)
    loss=np.zeros(N)
    supp=np.ones((T, N))
    step=0
    ind=1
    for i in range(N):
        C_i = C[i * M: (i + 1) * M, :]
        Y_i = Y[i * M: (i + 1) * M]
        V_i = V[i * T: (i + 1) * T]
        x_e[:, i]= upd.update_x(Y_i, C_i, gamma)

    for i in range(N):
        for j in range(T):
            if abs(x_e[j, i])<=0.0001:
                supp[j, i]=0
                v_e[j, i]=0

    for i in range(N):
        neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
        for j in neig_location:
            if (supp[:, i]!=supp[:, j]).any():
                ind=0
                break
        if ind==0:
                break

    while ind==0:
        step=step+1
        x_old=copy.deepcopy(x_e)
        for i in range(N):
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            num_neig=x_neig.shape[1]
            for j in range(T):
                neigh = []
                if abs(x_old[j, i])<=0.0001:
                    for mm in range(num_neig):
                        if abs(x_neig[j, mm])>0.0001:
                            neigh.append(x_neig[j, mm])
                    if len(neigh):
                        x_e[j, i] = np.average(np.array(neigh))
                else:
                    neigh.append(x_old[j, i])
                    for mm in range(num_neig):
                        if abs(x_neig[j, mm])>0.0001:
                            neigh.append(x_neig[j, mm])
                    if len(neigh):
                        x_e[j, i] = np.average(np.array(neigh))
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))

        ind=1
        '''
        #判断是否support set都相同
        supp = np.ones((T, N))
        for i in range(N):
            for j in range(T):
                if abs(x_e[j, i]) <= 0.0001:
                    supp[j, i] = 0

        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            for j in neig_location:
                if (supp[:, i] != supp[:, j]).any():
                    ind = 0
                    break
            if ind == 0:
                break

        if step>=10:
            break'''

    num_correct = np.zeros(N)
    ratio_correct = np.zeros(N)
    for i in range(N):
        for t in range(T):
            if v_e[t, i] == V_array[t, i]:
                num_correct[i] = num_correct[i] + 1
        ratio_correct[i] = 1.0 * num_correct[i] / T

    x_tilde = np.mean(x_e, axis=1)
    avr_com_mse = np.average(com_mse)
    avr_loc_mse = np.average(loc_mse)
    avr_loss = np.average(loss)
    print("avr_com_mse: %10.4f, avr_loc_mse: %10.4f, avr_loss:  %10.4f, step: %d\n" % (avr_com_mse, avr_loc_mse, avr_loss, step))

    mse_total = 1.0 /T* np.linalg.norm(x_tilde - X, 2)**2
    cserr = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
    print("mse_total: %10.4f, cserr: %10.4f\n" % (mse_total, cserr))
    return avr_com_mse, avr_loc_mse, mse_total, cserr, np.average(ratio_correct)


def baseline_Cint(A, Y, X, V, N, gamma):
    m = np.size(A, 0)
    T = np.size(A, 1)
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    xx=np.zeros(T)
    C = A
    loc_mse=np.zeros(N)
    loss=np.zeros(N)
    supp=np.ones((T, N))
    step=0
    ind=1
    for i in range(N):
        C_i = C[i * M: (i + 1) * M, :]
        Y_i = Y[i * M: (i + 1) * M]
        V_i = V[i * T: (i + 1) * T]
        x_e[:, i]= upd.update_x(Y_i, C_i, gamma)

    for t in range(T):
        sum_t=0
        sum_n=0
        for i in range(N):
            if abs(x_e[t, i])>0.0001:
                sum_n=sum_n+1
                sum_t=sum_t+x_e[t, i]
        if sum_n!=0:
            xx[t]=1.0*sum_t/sum_n

    for i in range(N):
        loc_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - np.multiply(V_array[:, i], X), 2) ** 2

    avr_com_mse = 1.0 / T * np.linalg.norm(xx - X, 2) ** 2
    avr_loc_mse = np.average(loc_mse)
    avr_loss = np.average(loss)
    print("avr_com_mse: %10.4f, avr_loc_mse: %10.4f\n" % (avr_com_mse, avr_loc_mse))

    return avr_com_mse, avr_loc_mse


def baseline_JSM1(A, Y, X, V, N, Adjacent_Matrix, c, rho, gamma1, gamma2, gamma3, MAX_ITER):
    # corrupted Lasso
    QUIET = 0
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    e_e = np.zeros((T, N))
    beta_e = np.zeros((T, N))
    x_e = np.zeros((T, N))
    p_e = np.zeros((T, N))
    tao_e = np.zeros((T, N))
    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        tao_old = copy.deepcopy(tao_e)
        e_old = copy.deepcopy(e_e)
        x_old = copy.deepcopy(x_e)
        beta_old = copy.deepcopy(beta_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of dual variables
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            tao_e[:, i]=tao_old[:, i]+rho*(e_old[:, i]-x_old[:, i]-beta_old[:, i])
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            # e_e[:, i]=np.dot(temp1, temp2)
            e_e[:, i]=updJSM1.update_e_JSM1(C_i, Y_i, x_old[:, i], beta_old[:, i], rho, gamma3, tao_e[:, i])
            x_e[:, i] = updJSM1.update_x_JSM1(e_e[:, i], x_old[:, i], x_neig, beta_old[:, i],
                                              p_e[:, i], c, gamma1, tao_e[:, i], rho)
            beta_e[:, i] = updJSM1.update_beta_JSM1(e_e[:, i], x_e[:, i], rho, tao_e[:, i], gamma2)

            # 得到了全局的稀疏矢量xe之后很重点的一步是如何得到对应的本地xlocal 可以考虑利用xe的support set降低本地xlocal估计的误差
            # 具体的。对应x_e为0的index，A的对应数列置为0
            x_local = e_e[:, i]
            loc_mse[i] = 1.0 / T * np.linalg.norm(x_local - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            # loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
        #     % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))


        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
                  % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse[MAX_ITER-1], avr_loc_mse[MAX_ITER-1]  #[MAX_ITER-1] #, mse_total, cserr, maxiter


def decentral_l1_VR_penalty_c_ind_hard(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, eta, USTp, Umin):
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    v_temp = np.ones((T, N))
    p_e = np.zeros((T, N))

    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))

    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = np.ones(N)
    ind3 = np.zeros(N)
    ind5 = np.zeros(N)
    ii = np.zeros(N)
    KK = MAX_ITER*np.ones(N)
    K_end = MAX_ITER * np.ones(N)
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            if k>=1:
                mean_indv[k, i]= indd_Vd[k-1, i]   #np.average(np.column_stack(indv_neig, x_old[:, i]))
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c*np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)

            if (ind5[i] == 0) &  (k >= KK[i]+ 1) & (
                    (r[k, i] ** 2 > eta * s[k, i] ** 2) | (eta * r[k, i] ** 2 < s[k, i] ** 2) | (k>=40)):
                K_end[i]=k

            if (ind[i] == 1) & (ind3[i] == 0) & (k >= 2):
                if mean_indv[k, i] <= mean_indv[k - 1, i]:
                    ii[i] = 0
                else:
                    ii[i] = 1

            if (ind3[i] == 0) & (k>=Umin) & ((ii[i] == 1) | (k >= USTp)):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                # print("Discretization for node %d"%i)
                KK[i] = k + 1
                ind3[i] = 1


            if (ind3[i]==0) & (ind[i]==0):
                ind3[i] = 1
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                v_e[:, i] = map_v_hard1(v_temp[:, i], 1)


            if (ind[i] == 1):  # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = 1.0/(1.0-alpha)*np.linalg.pinv(B_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                if ind3[i] == 0:
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 0)
                else:
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 1)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            indd_Vd[k, i]=(np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2

            if k == K_end[i] :
                ind[i] = 0
                ind5[i] = 1
                #print("****************the update of v %d ends*********************"%i)

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        a=v_e-np.ones((T, N))
        b=np.multiply(v_e, a)
        indd_V[k] = ((np.linalg.norm(b, 'fro')) ** 2) / N
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, indd_V:  %10.4f, cserr:  %10.4f\n"
        #    % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k], cserr[k]))

        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, inddV:  %10.4f\n"
                 % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse[MAX_ITER-1], avr_loc_mse[MAX_ITER-1]  # [MAX_ITER-1]


def ADMM_VPD_JSM1_ind(A, Y, X, V, N, Adjacent_Matrix, alpha, c,
                  rho, gamma1, gamma2, gamma3, c_new, gamma_new, MAX_ITER, eta, USTp, Umin):
    # Combination of VPD and JSM1 algorithm
    QUIET=np.ones(N)
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    e_e = np.zeros((T, N))
    beta_e = np.zeros((T, N))
    tao_e = np.zeros((T, N))
    r = np.zeros((MAX_ITER,N))
    s = np.zeros((MAX_ITER, N))

    C=A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = np.ones(N)
    ind3 = np.zeros(N)
    ind5 = np.zeros(N)
    ii = np.zeros(N)
    KK = MAX_ITER * np.ones(N)
    K_end = MAX_ITER*np.ones(N)
    stop_trans=0
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        tao_old = copy.deepcopy(tao_e)
        e_old = copy.deepcopy(e_e)
        beta_old = copy.deepcopy(beta_e)
        for i in range(N):
            if QUIET[i]==1:
                neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
                x_neig = x_old[:, neig_location]
                # update of p
                p_e[:, i] = p_old[:, i] + c_new * sum_neigh(x_old[:, i], x_neig)
                if k >= 1:
                    mean_indv[k, i] = indd_Vd[k - 1, i]  # np.average(np.column_stack(indv_neig, x_old[:, i]))

                # the local measurement matrix and the measurement result
                C_i = C[i * M: (i + 1) * M, :]
                Y_i = Y[i * M: (i + 1) * M]
                V_i = V[i * T: (i + 1) * T]

                x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c_new, x_old[:, i], x_neig,
                                                   gamma_new)
                r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
                if k > 0:
                    s[k, i] = c_new * np.linalg.norm(np.average(x_neig, axis=1) - \
                                                     np.average(x_oldd[:, neig_location], axis=1), ord=2)

                if (ind5[i] == 0) & (k >= KK[i] + 1) & (
                        ((r[k, i] ** 2 > eta * s[k, i] ** 2)) | (
                        eta * r[k, i] ** 2 < s[k, i] ** 2) | k>=40):  # (k >= max(KK) + 1) &
                    K_end[i] = k

                if (ind[i] == 1) & (ind3[i] == 0) & (k >= 2):
                    if mean_indv[k, i] <= mean_indv[k - 1, i]:
                        ii[i] = 0
                    else:
                        ii[i] = 1

                if (ind3[i] == 0) & (k >= Umin) & ((ii[i] == 1) | (k >= USTp)):
                    # print("Discretization for node %d" % i)
                    KK[i] = k + 1
                    ind3[i] = 1

                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = np.linalg.pinv(B_i_d - M_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                if ind3[i] == 0:
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 0)
                else:
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 1)

                loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
                com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
                loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
                indd_Vd[k, i] = (np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2

                if k == K_end[i]:
                    ind[i] = 0
                    ind5[i] = 1
                    QUIET[i] = 0
                    # print("end of node %d"%i)
                    C_i = C[i * M: (i + 1) * M, :]
                    Y_i = Y[i * M: (i + 1) * M]
                    V_i = V[i * T: (i + 1) * T]
                    if (ind3[i] == 0) & (ind[i] == 0):
                        ind3[i] = 1
                        B_i = np.dot(C_i, np.diag(x_e[:, i]))
                        B_i_d = np.dot(B_i.T, B_i)
                        v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                        v_e[:, i] = map_v_hard1(v_temp[:, i], 1)
                        e_e[:, i] = np.multiply(v_e[:, i], x_e[:, i]) + beta_e[:, i]
                    else:
                        e_e[:, i] = np.multiply(v_e[:, i], x_e[:, i]) + beta_e[:, i]

            elif QUIET[i] == 0:
                neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
                x_neig = x_old[:, neig_location]

                # the local measurement matrix and the measurement result
                C_i = C[i * M: (i + 1) * M, :]
                Y_i = Y[i * M: (i + 1) * M]
                V_i = V[i * T: (i + 1) * T]

                # update of dual variables
                p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
                tao_e[:, i] = tao_old[:, i] + rho * (e_old[:, i] - np.multiply(x_old[:, i], v_e[:, i]) - beta_old[:, i])

                e_e[:, i] = updEC.update_e_JSM1(C_i, Y_i, x_old[:, i], beta_old[:, i], rho, gamma3, tao_e[:, i],
                                                v_e[:, i])
                beta_e[:, i] = updEC.update_beta_JSM1(e_e[:, i], x_old[:, i], rho, tao_e[:, i], gamma2, v_e[:, i])
                x_e[:, i] = updEC.update_x_JSM1(e_e[:, i], x_old[:, i], x_neig, beta_e[:, i],
                                                p_e[:, i], c, gamma1, tao_e[:, i], rho, v_e[:, i])

                # 得到了全局的稀疏矢量xe之后很重点的一步是如何得到对应的本地xlocal 可以考虑利用xe的support set降低本地xlocal估计的误差
                # 具体的。对应x_e为0的index，A的对应数列置为0
                x_local = e_e[:, i]
                loc_mse[i] = 1.0 / T * np.linalg.norm(x_local - np.multiply(V_i, X), 2) ** 2
                com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, inddv:  %10.4f\n"
        #   % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))

        if (k==MAX_ITER-1): # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
                  % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))
            break

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse[MAX_ITER-1], avr_loc_mse[MAX_ITER-1]  # [MAX_ITER-1]  #, mse_total, cserr, maxiter


def observe_v(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, eta, USTp, Umin):
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    v_temp = np.ones((T, N))
    p_e = np.zeros((T, N))

    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))

    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = np.ones(N)
    ind3 = np.zeros(N)
    ind5 = np.zeros(N)
    ii = np.zeros(N)
    KK = MAX_ITER*np.ones(N)
    K_end = MAX_ITER * np.ones(N)
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            if k>=1:
                mean_indv[k, i]= indd_Vd[k-1, i]   #np.average(np.column_stack(indv_neig, x_old[:, i]))
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c*np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)

            if (ind5[i] == 0) &  (k >= KK[i]+ 1) & (
                    (r[k, i] ** 2 > eta * s[k, i] ** 2) | (eta * r[k, i] ** 2 < s[k, i] ** 2) | (k>=75)):
                K_end[i]=k

            if (ind[i] == 1) & (ind3[i] == 0) & (k >= 2):
                if mean_indv[k, i] <= mean_indv[k - 1, i]:
                    ii[i] = 0
                else:
                    ii[i] = 1

            if (ind3[i] == 0) & (k>=Umin) & ((ii[i] == 1) | (k >= USTp)):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                # print("Discretization for node %d"%i)
                KK[i] = k + 1
                ind3[i] = 1


            if (ind3[i]==0) & (ind[i]==0):
                ind3[i] = 1
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                v_e[:, i] = map_v_hard1(v_temp[:, i], 1)


            if (ind[i] == 1):  # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = 1.0/(1.0-alpha)*np.linalg.pinv(B_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                if ind3[i] == 0:
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 0)
                else:
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 1)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            indd_Vd[k, i]=(np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2

            if k == K_end[i] :
                ind[i] = 0
                ind5[i] = 1
                #print("****************the update of v %d ends*********************"%i)

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        a=v_e-np.ones((T, N))
        b=np.multiply(v_e, a)
        indd_V[k] = ((np.linalg.norm(b, 'fro')) ** 2) / N
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, indd_V:  %10.4f, cserr:  %10.4f\n"
        #    % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k], cserr[k]))

        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, inddV:  %10.4f\n"
                 % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse[MAX_ITER-1], avr_loc_mse[MAX_ITER-1]


# add one more comparison of the standard method
def stand_DLasso(A, Y, X, V, N, Adjacent_Matrix, c, gamma, MAX_ITER):
    # corrupted Lasso
    QUIET = 0
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    x_e = np.zeros((T, N))
    p_e = np.zeros((T, N))
    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of dual variables
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i] = upd.update_x_standard(Y_i, C_i, p_e[:, i], c, x_old[:, i], x_neig, gamma)

            x_local = x_e[:, i]
            loc_mse[i] = 1.0 / T * np.linalg.norm(x_local - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            # loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
        #    % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))


        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
                  % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse[MAX_ITER-1], avr_loc_mse[MAX_ITER-1]  #[MAX_ITER-1] #, mse_total, cserr, maxiter


# plot the curves of the masking vector
def curve_v(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, eta, USTp, Umin):
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    v_temp = np.ones((T, N))
    p_e = np.zeros((T, N))

    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))

    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    avr_v = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = np.ones(N)
    ind3 = np.zeros(N)
    ind5 = np.zeros(N)
    ii = np.zeros(N)
    KK = MAX_ITER*np.ones(N)
    K_end = MAX_ITER * np.ones(N)
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        v_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            if k>=1:
                mean_indv[k, i]= indd_Vd[k-1, i]   #np.average(np.column_stack(indv_neig, x_old[:, i]))
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c*np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)

            if (ind5[i] == 0) &  (k >= KK[i]+ 1) & (
                    (r[k, i] ** 2 > eta * s[k, i] ** 2) | (eta * r[k, i] ** 2 < s[k, i] ** 2) | (k>=75)):
                K_end[i]=k

            if (ind[i] == 1) & (ind3[i] == 0) & (k >= 2):
                if mean_indv[k, i] <= mean_indv[k - 1, i]:
                    ii[i] = 0
                else:
                    ii[i] = 1

            if (ind3[i] == 0) & (k>=Umin) & ((ii[i] == 1) | (k >= USTp)):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                print("Discretization for node %d"%i)
                KK[i] = k + 1
                ind3[i] = 1


            if (ind3[i]==0) & (ind[i]==0):
                ind3[i] = 1
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                v_e[:, i] = map_v_hard1(v_temp[:, i], 1)


            if (ind[i] == 1):  # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = 1.0/(1.0-alpha)*np.linalg.pinv(B_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                if ind3[i] == 0:
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 0)
                else:
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 1)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            indd_Vd[k, i]=(np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2
            v_mse[i] = 1.0 / T * np.linalg.norm(v_e[:, i] - V_i, 2) ** 2

            if k == K_end[i] :
                ind[i] = 0
                ind5[i] = 1
                print("****************the update of v %d ends*********************"%i)

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        avr_v[k] = np.average(v_mse)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        a=v_e-np.ones((T, N))
        b=np.multiply(v_e, a)
        indd_V[k] = ((np.linalg.norm(b, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, avr_v_mse:  %10.4f, v_mse:  %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], avr_v[k], v_mse[5]))

        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, avr_v_mse:  %10.4f\n"
                 % (k, avr_com_mse[k], avr_loc_mse[k], avr_v[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]
    avr_v[maxiter: MAX_ITER] = avr_v[maxiter - 1]

    return avr_com_mse, avr_loc_mse, avr_v


###############################################################################################################
def decentral_l1_VR_penalty_c_1hard(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, mu, eta, USTp):
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    v_temp = np.ones((T, N))
    p_e = np.zeros((T, N))

    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))

    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 1
    indd = 1
    ind3 = 0
    iid = 0
    ind5 = 0
    ii = np.zeros(N)
    ii_all = 0
    KK = MAX_ITER
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        v_temp_old=copy.deepcopy(v_temp)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            if k>=1:
                indv_neig=indd_Vd[k-1, neig_location]
                mean_indv[k, i]=np.average(indv_neig)
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c*np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)

            if (ind5 == 0) & (iid == 1) & (k >= KK + 1) & (
                    ((r[k, i] ** 2 > eta * s[k, i] ** 2)) | (eta * r[k, i] ** 2 < s[k, i] ** 2)):
                ind = 0
                ind5 = 1
                K_end = k + 1
                print("****************the update of v ends********************")
            '''
            if (ind == 1) & (ind3 == 0) & (k >= 2):
                # print(np.linalg.norm(x_old[:, i] - x_oldd[:, i], ord=2))
                # print(np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2))
                ii = (np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2) <= mu)'''

            if (ind == 1) & (ind3 == 0) & (k >= 2):
                if mean_indv[k, i] <= mean_indv[k - 1, i]:
                    ii[i] = 0
                else:
                    ii[i] = 1

            if (ind3 == 0) & ((ii_all == 1) | (k >= USTp)):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                print("Discretization")
                KK = k + 1
                ind3 = 1

            if (indd == 1):  # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i =  B_i
                M_i_d = alpha *np.dot(M_i.T, M_i)
                part1 = np.linalg.pinv(B_i_d - M_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                # v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                if iid==0:
                    v_temp[:, i] = np.dot(part1, part2)
                    # v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i) - alpha * (v_old[:, i] - 0.5 * np.ones(T)))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)  #iid
                else:
                    # alpha=0.01
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            indd_Vd[k, i]=(np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2
        if ind == 1:
            indd = 1
        elif k==K_end-1:
            indd = 0
            # gamma=gamma/5

        if (ind3 == 1) & (k==KK-1):
            iid = 1
        if np.sum(ii) == N:
            ii_all = 1
        else:
            ii_all = 0

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        a=v_e-np.ones((T, N))
        b=np.multiply(v_e, a)
        indd_V[k] = ((np.linalg.norm(b, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, indd_V:  %10.4f, cserr:  %10.4f\n"
             % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k], cserr[k]))

        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, inddV:  %10.4f\n"
                 % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter



def ADMM_VPD_JSM1(A, Y, X, V, N, Adjacent_Matrix, alpha, c,
                  rho, gamma1, gamma2, gamma3, c_new, gamma_new, MAX_ITER, mu, eta, USTp):
    # Combination of VPD and JSM1 algorithm
    QUIET=1
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    e_e = np.zeros((T, N))
    beta_e = np.zeros((T, N))
    tao_e = np.zeros((T, N))
    r = np.zeros((MAX_ITER,N))
    s = np.zeros((MAX_ITER, N))

    C=A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = np.ones(N)
    ind3 = 0
    iid=0
    ind5 = 0
    ii = np.zeros(N)
    ii_all = 0
    KK=MAX_ITER
    stop_trans=0
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        if QUIET==1:
            # print("VPD: iteration %d\n" % k)
            loss = np.zeros(N)
            loc_mse = np.zeros(N)
            com_mse = np.zeros(N)
            p_old = copy.deepcopy(p_e)
            x_old = copy.deepcopy(x_e)
            v_old = copy.deepcopy(v_e)
            for i in range(N):
                neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
                x_neig = x_old[:, neig_location]
                v_neig = v_old[:, neig_location]
                # update of p
                p_e[:, i] = p_old[:, i]+ c_new*sum_neigh(x_old[:,i], x_neig)
                #print p[:,i].shape
                if k >= 1:
                    indv_neig = indd_Vd[k - 1, neig_location]
                    mean_indv[k, i] = np.average(indv_neig)

                # the local measurement matrix and the measurement result
                C_i = C[i * M: (i + 1) * M, :]
                Y_i = Y[i * M: (i + 1) * M]
                V_i = V[i * T: (i + 1) * T]

                x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c_new, x_old[:, i], x_neig, gamma_new)
                r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
                if k>0:
                    s[k, i] =  c_new*np.linalg.norm(np.average(x_neig, axis=1) - \
                                            np.average(x_oldd[:, neig_location], axis=1), ord=2)

                if (ind5[i]==0) & (ind3==1) & (k>=KK+1) &\
                        (((r[k, i] ** 2 > eta * s[k, i] ** 2))|( eta * r[k, i] ** 2 < s[k, i] ** 2)| (k>75)):
                    ind[i] = 0
                    ind5[i] = 1
                    K_end=k+1
                    print("****************the update of v ends********************")

                if (ind == 1) & (ind3 == 0) & (k >= 2):
                    if mean_indv[k, i] <= mean_indv[k - 1, i]:  # any(indd_Vd[k-1, neig_location]<=indd_Vd[k-2, neig_location]):
                        ii[i] = 0
                    else:
                        ii[i] = 1

                if (ind3 == 0) & ((ii_all == 1) | (k >= USTp)):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                    print("Discretization")
                    KK = k + 1
                    ind3 = 1

                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha*np.dot(M_i.T, M_i)
                part1 = np.linalg.pinv(B_i_d - M_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                if iid==0:
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] =map_v_hard1(v_temp[:, i], iid)
                else:
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)

                loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
                com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
                loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
                indd_Vd[k, i] = (np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2
            if ind3==1:
                iid=1
            if np.sum(ii) == N:
                ii_all = 1
            else:
                ii_all = 0
        else:
            # print("JSM1: iteration %d\n" % k)
            loss = np.zeros(N)
            loc_mse = np.zeros(N)
            com_mse = np.zeros(N)
            p_old = copy.deepcopy(p_e)
            tao_old = copy.deepcopy(tao_e)
            e_old = copy.deepcopy(e_e)
            x_old = copy.deepcopy(x_e)
            beta_old = copy.deepcopy(beta_e)
            for i in range(N):
                neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
                x_neig = x_old[:, neig_location]

                # the local measurement matrix and the measurement result
                C_i = C[i * M: (i + 1) * M, :]
                Y_i = Y[i * M: (i + 1) * M]
                V_i = V[i * T: (i + 1) * T]

                # update of dual variables
                p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
                tao_e[:, i] = tao_old[:, i] + rho * (e_old[:, i] - np.multiply(x_old[:, i], v_e[:, i]) - beta_old[:, i])

                e_e[:, i] = updEC.update_e_JSM1(C_i, Y_i, x_old[:, i], beta_old[:, i], rho, gamma3, tao_e[:, i],
                                                v_e[:, i])
                beta_e[:, i] = updEC.update_beta_JSM1(e_e[:, i], x_old[:, i], rho, tao_e[:, i], gamma2, v_e[:, i])
                x_e[:, i] = updEC.update_x_JSM1(e_e[:, i], x_old[:, i], x_neig, beta_e[:, i],
                                                p_e[:, i], c, gamma1, tao_e[:, i], rho, v_e[:, i])

                # 得到了全局的稀疏矢量xe之后很重点的一步是如何得到对应的本地xlocal 可以考虑利用xe的support set降低本地xlocal估计的误差
                # 具体的。对应x_e为0的index，A的对应数列置为0
                x_local = e_e[:, i]
                loc_mse[i] = 1.0 / T * np.linalg.norm(x_local - np.multiply(V_i, X), 2) ** 2
                com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2

        # decide whether go to JSM1 in the next iteration
        if ind == 1:
            QUIET = 1
        elif (ind==0) & (stop_trans==0):
            QUIET = 0
            stop_trans=1
            # p_e = np.zeros((T, N))
            # initialization of the JSM1 parameters
            for i in range(N):
                # beta_e[:, i] = np.multiply(v_e[:, i], x_e[:, i])-x_e[:, i]
                e_e[:, i] = np.multiply(v_e[:, i], x_e[:, i])+beta_e[:, i]  # beta_e[:, i]+x_e[:, i]#'''

        X_array = np.array([X, X, X, X, X, X]).T
        beta_best = np.multiply((V_array - v_e), X_array)
        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, inddv:  %10.4f\n"
        #       % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))

        if (k==MAX_ITER-1): # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
                  % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))
            break

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter


def decentral_l1_VR_penalty_c_ind_hard_glo(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, eta, USTp, Umin):
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    v_temp = np.ones((T, N))
    p_e = np.zeros((T, N))

    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))

    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 1
    ind3 = np.zeros(N)
    ind5 = 0
    ii = np.zeros(N)
    KK = MAX_ITER*np.ones(N)
    K_end = MAX_ITER
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            if k>=1:
                mean_indv[k, i]= indd_Vd[k-1, i]   #np.average(np.column_stack(indv_neig, x_old[:, i]))
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c*np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)

            if (ind5== 0) &  (k >= KK[i]+ 1) & (
                    ((r[k, i] ** 2 > eta * s[k, i] ** 2)) | (eta * r[k, i] ** 2 < s[k, i] ** 2) | k>=75):
                K_end=k

            if (ind== 1) & (ind3[i] == 0) & (k >= 2):
                if mean_indv[k, i] <= mean_indv[k - 1, i]:
                    ii[i] = 0
                else:
                    ii[i] = 1

            if (ind3[i] == 0) & (k>=Umin) & ((ii[i] == 1) | (k >= USTp)):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                # print("Discretization for node %d"%i)
                KK[i] = k + 1
                ind3[i] = 1


            if (ind3[i]==0) & (ind==0):
                ind3[i] = 1
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                v_e[:, i] = map_v_hard1(v_temp[:, i], 1)


            if (ind == 1):  # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = 1.0/(1.0-alpha)*np.linalg.pinv(B_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                if ind3[i] == 0:
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 0)
                else:
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 1)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            indd_Vd[k, i]=(np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2

        if k == K_end :
            ind = 0
            ind5 = 1
            print("****************the update of v ends*********************")

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        a=v_e-np.ones((T, N))
        b=np.multiply(v_e, a)
        indd_V[k] = ((np.linalg.norm(b, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, indd_V:  %10.4f, cserr:  %10.4f\n"
            % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k], cserr[k]))

        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, inddV:  %10.4f\n"
                 % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse[MAX_ITER-1], avr_loc_mse[MAX_ITER-1]


def ADMM_VPD_JSM1_ind_glo(A, Y, X, V, N, Adjacent_Matrix, alpha, c,
                  rho, gamma1, gamma2, gamma3, c_new, gamma_new, MAX_ITER, eta, USTp, Umin):
    # Combination of VPD and JSM1 algorithm
    QUIET=1
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    e_e = np.zeros((T, N))
    beta_e = np.zeros((T, N))
    tao_e = np.zeros((T, N))
    r = np.zeros((MAX_ITER,N))
    s = np.zeros((MAX_ITER, N))

    C=A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = np.ones(N)
    ind3 = np.zeros(N)
    ind5 = np.zeros(N)
    ii = np.zeros(N)
    KK = MAX_ITER * np.ones(N)
    K_end = MAX_ITER*np.ones(N)
    stop_trans=0
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        if QUIET==1:
            #print("VPD: iteration %d\n" % k)
            loss = np.zeros(N)
            loc_mse = np.zeros(N)
            com_mse = np.zeros(N)
            p_old = copy.deepcopy(p_e)
            x_old = copy.deepcopy(x_e)
            v_old = copy.deepcopy(v_e)
            for i in range(N):
                x_inv=np.ones(T)
                neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
                x_neig = x_old[:, neig_location]
                # update of p
                p_e[:, i] = p_old[:, i] + c_new *sum_neigh(x_old[:, i], x_neig)
                if k >= 1:
                    mean_indv[k, i] = indd_Vd[k - 1, i]  # np.average(np.column_stack(indv_neig, x_old[:, i]))

                # the local measurement matrix and the measurement result
                C_i = C[i * M: (i + 1) * M, :]
                Y_i = Y[i * M: (i + 1) * M]
                V_i = V[i * T: (i + 1) * T]

                x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c_new, x_old[:, i], x_neig, gamma_new)
                r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
                if k > 0:
                    s[k, i] = c_new * np.linalg.norm(np.average(x_neig, axis=1) - \
                                                 np.average(x_oldd[:, neig_location], axis=1), ord=2)

                if (ind5[i] == 0) & (k>=KK[i]+1)  &(
                        ((r[k, i] ** 2 > eta * s[k, i] ** 2)) | (eta * r[k, i] ** 2 < s[k, i] ** 2)): # (k >= max(KK) + 1) &
                    K_end[i] = k

                if (ind == 1) & (ind3[i] == 0) & (k >= 2):
                    if mean_indv[k, i] <= mean_indv[k - 1, i]:
                        ii[i] = 0
                    else:
                        ii[i] = 1

                if (ind3[i] == 0) & (k >= Umin) & ((ii[i] == 1) | (k >= USTp)):
                    # print("Discretization for node %d" % i)
                    KK[i] = k + 1
                    ind3[i] = 1

                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = np.linalg.pinv(B_i_d - M_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                if ind3[i] == 0:
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 0)
                else:
                    v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], 1)

                loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
                com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
                loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
                indd_Vd[k, i] = (np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2

                if k == K_end[i]:
                    ind[i] = 0
                    ind5[i] = 1
                    QUIET = 0
                    # print("****************the update of v ends for node*********************")
                    for i in range(N):
                        C_i = C[i * M: (i + 1) * M, :]
                        Y_i = Y[i * M: (i + 1) * M]
                        V_i = V[i * T: (i + 1) * T]
                        if (ind3[i] == 0) & (ind == 0):
                            ind3[i] = 1
                            B_i = np.dot(C_i, np.diag(x_e[:, i]))
                            B_i_d = np.dot(B_i.T, B_i)
                            v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                            v_e[:, i] = map_v_hard1(v_temp[:, i], 1)
                            e_e[:, i] = np.multiply(v_e[:, i], x_e[:, i]) + beta_e[:, i]
                        else:
                            e_e[:, i] = np.multiply(v_e[:, i], x_e[:, i]) + beta_e[:, i]

        else:
            # print("JSM")
            loss = np.zeros(N)
            loc_mse = np.zeros(N)
            com_mse = np.zeros(N)
            p_old = copy.deepcopy(p_e)
            tao_old = copy.deepcopy(tao_e)
            e_old = copy.deepcopy(e_e)
            x_old = copy.deepcopy(x_e)
            beta_old = copy.deepcopy(beta_e)
            for i in range(N):
                neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
                x_neig = x_old[:, neig_location]

                # the local measurement matrix and the measurement result
                C_i = C[i * M: (i + 1) * M, :]
                Y_i = Y[i * M: (i + 1) * M]
                V_i = V[i * T: (i + 1) * T]

                # update of dual variables
                p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
                tao_e[:, i] = tao_old[:, i] + rho * (e_old[:, i] - np.multiply(x_old[:, i], v_e[:, i]) - beta_old[:, i])

                e_e[:, i] = updEC.update_e_JSM1(C_i, Y_i, x_old[:, i], beta_old[:, i], rho, gamma3, tao_e[:, i],
                                                v_e[:, i])
                beta_e[:, i] = updEC.update_beta_JSM1(e_e[:, i], x_old[:, i], rho, tao_e[:, i], gamma2, v_e[:, i])
                x_e[:, i] = updEC.update_x_JSM1(e_e[:, i], x_old[:, i], x_neig, beta_e[:, i],
                                                p_e[:, i], c, gamma1, tao_e[:, i], rho, v_e[:, i])

                # 得到了全局的稀疏矢量xe之后很重点的一步是如何得到对应的本地xlocal 可以考虑利用xe的support set降低本地xlocal估计的误差
                # 具体的。对应x_e为0的index，A的对应数列置为0
                x_local = e_e[:, i]
                loc_mse[i] = 1.0 / T * np.linalg.norm(x_local - np.multiply(V_i, X), 2) ** 2
                com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, inddv:  %10.4f\n"
        #    % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))

        if (k==MAX_ITER-1): # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
                  % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))
            break

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse[MAX_ITER-1], avr_loc_mse[MAX_ITER-1]  #, mse_total, cserr, maxiter


#####################################################################################################3
def decentral_l1_VR_penalty_c_1hard2(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, mu, eta):
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    v_temp = np.ones((T, N))
    p_e = np.zeros((T, N))

    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))

    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 1
    indd = 1
    ind3 = 0
    iid = 0
    ind5 = 0
    ii = 0
    KK = MAX_ITER
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        v_temp_old=copy.deepcopy(v_temp)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            if k>=1:
                indv_neig=indd_Vd[k-1, neig_location]
                mean_indv[k, i]=np.average(indv_neig)
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c*np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)

            if (ind5 == 0) & (iid == 1) & (k >= KK + 1) & (
                    ((r[k, i] ** 2 > eta * s[k, i] ** 2)) | (eta * r[k, i] ** 2 < s[k, i] ** 2)):
                ind = 0
                ind5 = 1
                K_end = k + 1
                print("****************the update of v ends********************")

            '''
            if (ind == 1) & (ind3 == 0) & (k >= 2):
                # print(np.linalg.norm(x_old[:, i] - x_oldd[:, i], ord=2))
                # print(np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2))
                ii = (np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2) <= mu)
            if (ind3 == 0) & ((ii == 1) | (k >= 35)):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                # print("Discretization")
                KK = k + 1
                ind3 = 1
            '''

            if (ind3 == 0) & (mean_indv[k, i]>mean_indv[k-1, i]) & (k>=10):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                print("Discretization")
                KK = k + 1
                ind3 = 1

            if indd == 1:  # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = np.linalg.pinv(B_i_d - M_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                # v_temp[:, i] = np.dot(part1, part2)
                v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                if iid==0:
                    #v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i) - alpha * (v_old[:, i] - 0.5 * np.ones(T)))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)  #iid
                else:
                    # alpha=0.01
                    # v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            indd_Vd[k, i]=(np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2
        if ind == 1:
            indd = 1
        elif k==K_end-1:
            indd = 0
            # gamma=gamma/5

        if (ind3 == 1) & (k==KK-1):
            iid = 1

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        a=v_e-np.ones((T, N))
        b=np.multiply(v_e, a)
        indd_V[k] = ((np.linalg.norm(b, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, indd_V:  %10.4f\n"
             % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))

        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
                 % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter


def decentral_l1_VR_penalty_c_1hard3(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, mu, eta):
    iter_num = 10000
    m = np.size(A, 0)
    T = np.size(A, 1)
    # print K
    M = int(m / N)
    V_array = (np.reshape(V, [N, T])).T
    x_e = np.zeros((T, N))
    v_e = np.ones((T, N))
    v_temp = np.ones((T, N))
    p_e = np.zeros((T, N))

    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))

    C = A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 1
    indd = 1
    ind3 = 0
    iid = 0
    ind5 = 0
    ii = 0
    KK = MAX_ITER
    indd_V = np.zeros(MAX_ITER)
    indd_Vd = np.zeros((MAX_ITER, N))
    mean_indv =  np.zeros((MAX_ITER, N))

    for k in range(MAX_ITER):
        # print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        v_temp_old=copy.deepcopy(v_temp)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            p_e[:, i] = p_old[:, i] + c * sum_neigh(x_old[:, i], x_neig)
            if k>=1:
                indv_neig=indd_Vd[k-1, neig_location]
                mean_indv[k, i]=np.average(indv_neig)
            # print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c*np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)

            if (ind5 == 0) & (iid == 1) & (k >= KK + 1) & (
                    ((r[k, i] ** 2 > eta * s[k, i] ** 2)) | (eta * r[k, i] ** 2 < s[k, i] ** 2)):
                ind = 0
                ind5 = 1
                K_end = k + 1
                # print("****************the update of v ends********************")

            '''
            if (ind == 1) & (ind3 == 0) & (k >= 2):
                # print(np.linalg.norm(x_old[:, i] - x_oldd[:, i], ord=2))
                # print(np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2))
                ii = (np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2) <= mu)
            if (ind3 == 0) & ((ii == 1) | (k >= 35)):  # (ind3 == 0) & ((ii == 1) | (k >= 35))
                # print("Discretization")
                KK = k + 1
                ind3 = 1
            '''
            if (ind3 == 0) & (mean_indv[k, i]>mean_indv[k-1, i]) & (k>=2):  # (k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                # print("Discretization")
                KK = k + 1
                ind3 = 1

            if (indd == 1):  # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = B_i
                M_i_d = alpha * np.dot(M_i.T, M_i)
                part1 = np.linalg.pinv(B_i_d - M_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                v_temp[:, i] = np.dot(part1, part2)
                # v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                # v_temp[:, i]=upd.update_v_l1(Y_i, B_i, alpha)
                if iid==0:
                    # v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i) - alpha * (v_old[:, i] - 0.5 * np.ones(T)))
                    # v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i) - alpha * (v_old[:, i] - 0.5 * np.ones(T)))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)  #iid
                else:
                    # alpha=0.01
                    # v_temp[:, i] = np.dot(np.linalg.pinv(B_i_d), np.dot(B_i.T, Y_i))
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            indd_Vd[k, i]=(np.linalg.norm(np.multiply(v_e[:, i], (v_e[:, i] - np.ones(T))), ord=2)) ** 2
        if ind == 1:
            indd = 1
        elif k==K_end-1:
            indd = 0
            # gamma=gamma/5

        if (ind3 == 1) & (k==KK-1):
            iid = 1

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        a=v_e-np.ones((T, N))
        b=np.multiply(v_e, a)
        indd_V[k] = ((np.linalg.norm(b, 'fro')) ** 2) / N
        # print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, indd_V:  %10.4f\n"
        #     % (k, avr_com_mse[k], avr_loc_mse[k], indd_V[k]))

        if (k == MAX_ITER-1):  # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
                 % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))
            break

    maxiter = min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER] = avr_com_mse[maxiter - 1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter


##########################################################################################################
def decentral_l1_VR_penalty_c_2hard(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, mu):  # VR: vector recovery
    # 没有离散化
    QUIET=0
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    r = np.zeros((MAX_ITER,N))
    s = np.zeros((MAX_ITER, N))

    C=A
    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 1
    indd = 1
    dd = MAX_ITER
    ind2 = 0
    ind3 = 0
    iid=0
    ind5 = 0
    ii=0
    ii2=np.zeros((MAX_ITER,N))
    K_end=MAX_ITER
    KK=MAX_ITER

    for k in range(MAX_ITER):
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            v_neig = v_old[:, neig_location]
            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k>0:
                s[k, i] = c * np.linalg.norm(np.average(x_neig, axis=1) - \
                                        np.average(x_oldd[:, neig_location], axis=1), ord=2)
                # print(r[k, i], s[k, i], r[k, i] / s[k, i], s[k, i] / r[k, i])

            if (ind5==0) & (ind3==1) & (k>=KK+1) &(((r[k, i] ** 2 > 15.0 * s[k, i] ** 2))|( 15.0 * r[k, i] ** 2 < s[k, i] ** 2)|(k>=70)):
                ind = 0
                ind5 = 1
                K_end=k+1
                print("****************the update of v ends********************")

            if (ind==1) & (ind3==0) & (k>=2):
                # print(np.linalg.norm(x_old[:, i] - x_oldd[:, i], ord=2))
                # print(np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2))
                ii = (np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2)<=mu)
            if k>MAX_ITER: #(k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                print("Discretization")
                KK=k+1
                ind3=1

            if indd==1: # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = alpha * B_i
                M_i_d = np.dot(M_i.T, M_i)
                part1 = np.linalg.pinv(B_i_d - M_i_d)
                part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                v_temp[:, i] = np.dot(part1, part2)
                v_e[:, i] =map_v_hard1(v_temp[:, i], iid)

            # ii2[k, i] = np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2)
            # if ii2[k, i]<=0.25:
            #   ind4[i]=0
            loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))   ##原本的几个实验里面loss的值有点问题
            # print("node %d: the mseracy for common x is %10.4f, the mseracy for local x is %10.4f, "
                  # "the loss is %10.4f\n" % (i, com_mse[i], loc_mse[i], loss[i]))
        if ind==1:
            indd=1
        else:
            indd=0

        if ind3==1:
            iid=1

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f, loss: %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k], avr_loss[k]))

        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2

        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        # 如果连着十次迭代变化都小于0.0001，那么就停止迭代，判断收敛到局部极小值点
        # (avr_com_mse[k] - avr_com_mse[k - 1]>=0.00001):

        if (k==99): # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("Do not need 200 Iterations")
            break

    num_correct=np.zeros(N)
    ratio_correct=np.zeros(N)
    for i in range(N):
        for t in range(T):
            if v_e[t, i]==V_array[t, i]:
                num_correct[i]=num_correct[i]+1
        ratio_correct[i]=1.0*num_correct[i]/T

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter, np.average(ratio_correct)


def decentral_l1_VR_penalty_c_3hard(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, mu):  # VR: vector recovery
    # 没有及时停止v的更新
    QUIET=0
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    r = np.zeros((MAX_ITER,N))
    s = np.zeros((MAX_ITER, N))

    C=A
    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 1
    indd = 1
    dd = MAX_ITER
    ind2 = 0
    ind3 = 0
    iid=0
    ind5 = 0
    ii=0
    ii2=np.zeros((MAX_ITER,N))
    K_end=MAX_ITER
    KK=MAX_ITER

    for k in range(MAX_ITER):
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            v_neig = v_old[:, neig_location]
            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k>0:
                s[k, i] = c * np.linalg.norm(np.average(x_neig, axis=1) - \
                                        np.average(x_oldd[:, neig_location], axis=1), ord=2)
                # print(r[k, i], s[k, i], r[k, i] / s[k, i], s[k, i] / r[k, i])


            if k>MAX_ITER:
                ind = 0
                ind5 = 1
                K_end=k+1
                print("****************the update of v ends********************")

            if (ind==1) & (ind3==0) & (k>=2):
                # print(np.linalg.norm(x_old[:, i] - x_oldd[:, i], ord=2))
                # print(np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2))
                ii = (np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2)<=mu)
            if (ind3==0) & ((ii==1)|(k>=35)): #(k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                print("Discretization")
                KK=k+1
                ind3=1

            if indd==1: # (ind == 1) & (k>=dd):
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = alpha * B_i
                M_i_d = np.dot(M_i.T, M_i)
                if iid!=5:
                    part1 = np.linalg.pinv(B_i_d - M_i_d)
                    part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] =map_v_hard1(v_temp[:, i], iid)
                else:
                    part1 = np.linalg.pinv(B_i_d)
                    part2 = np.dot(B_i.T, Y_i)
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)

            # ii2[k, i] = np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2)
            # if ii2[k, i]<=0.25:
            #   ind4[i]=0
            loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))   ##原本的几个实验里面loss的值有点问题
            # print("node %d: the mseracy for common x is %10.4f, the mseracy for local x is %10.4f, "
                  # "the loss is %10.4f\n" % (i, com_mse[i], loc_mse[i], loss[i]))
        if ind==1:
            indd=1
        else:
            indd=0

        if ind3==1:
            iid=1

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f, loss: %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k], avr_loss[k]))

        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2

        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        # 如果连着十次迭代变化都小于0.0001，那么就停止迭代，判断收敛到局部极小值点
        # (avr_com_mse[k] - avr_com_mse[k - 1]>=0.00001):
        '''
        if (ind5==1) & (k>=K_end+10) & (abs(avr_com_mse[k] - avr_com_mse[k - 10]) < 0.0001):
            iter_num = k + 1
            print("Converge to local optimal")
            break

        if (avr_com_mse[k]<0.00001) & (cserr[k]<0.1):
            iter_num = k + 1
            print("get to the optimized point")
            break'''

        if (k==99): # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("Do not need 200 Iterations")
            break

    num_correct=np.zeros(N)
    ratio_correct=np.zeros(N)
    for i in range(N):
        for t in range(T):
            if v_e[t, i]==V_array[t, i]:
                num_correct[i]=num_correct[i]+1
        ratio_correct[i]=1.0*num_correct[i]/T

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter, np.average(ratio_correct)


def decentral_l1_x(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, mu):  # VR: vector recovery
    # 不考虑遮挡
    QUIET=0
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    r = np.zeros((MAX_ITER,N))
    s = np.zeros((MAX_ITER, N))

    C=A
    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)

    for k in range(MAX_ITER):
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            v_neig = v_old[:, neig_location]
            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f, loss: %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k], avr_loss[k]))

        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2

        if (k==60): # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("Do not need 200 Iterations")
            break

    num_correct=np.zeros(N)
    ratio_correct=np.zeros(N)
    for i in range(N):
        for t in range(T):
            if v_e[t, i]==V_array[t, i]:
                num_correct[i]=num_correct[i]+1
        ratio_correct[i]=1.0*num_correct[i]/T

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter, np.average(ratio_correct)


def decentral_l1_VR_nopenalty_c_1hard(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, MAX_ITER, mu):  # VR: vector recovery
    # 不加入惩罚函数
    QUIET=0
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    r = np.zeros((MAX_ITER,N))
    s = np.zeros((MAX_ITER, N))

    C=A
    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 1
    indd = 1
    dd = MAX_ITER
    ind2 = 0
    ind3 = 0
    iid=0
    ind5 = 0
    ii=0
    ii2=np.zeros((MAX_ITER,N))
    K_end=MAX_ITER
    KK=MAX_ITER

    for k in range(MAX_ITER):
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            v_neig = v_old[:, neig_location]
            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            '''
            B_i = np.dot(C_i, np.diag(x_old[:, i]))
            B_i_d = np.dot(B_i.T, B_i)
            M_i = alpha * B_i
            M_i_d = np.dot(M_i.T, M_i)
            part1 = np.linalg.pinv(B_i_d - M_i_d)
            part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
            v_temp[:, i] = np.dot(part1, part2)'''

            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)
            r[k, i] = np.linalg.norm(x_e[:, i] - np.average(x_neig, axis=1), ord=2)
            if k>0:
                s[k, i] = c * np.linalg.norm(np.average(x_neig, axis=1) - \
                                        np.average(x_oldd[:, neig_location], axis=1), ord=2)
                # print(r[k, i], s[k, i], r[k, i] / s[k, i], s[k, i] / r[k, i])

            '''if  (ind2==0) &(k > 1) & (r[k, i] ** 2 >= 3.0 * s[k, i] ** 2):
                ind = 1
                dd = k+1
                ind2 = 1
                dd2 = (k+1)*np.ones(N)
                print("****************the update of v begins********************")'''

            if (ind5==0) & (ind3==1) & (k>=KK+1) &(((r[k, i] ** 2 > 15.0 * s[k, i] ** 2))|( 15.0 * r[k, i] ** 2 < s[k, i] ** 2)|(k>=70)):
                ind = 0
                ind5 = 1
                K_end=k+1
                print("****************the update of v ends********************")

            if (ind==1) & (ind3==0) & (k>=2):
                # print(np.linalg.norm(x_old[:, i] - x_oldd[:, i], ord=2))
                # print(np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2))
                ii = (np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2)<=mu)
            if (ind3==0) & ((ii==1)|(k>=35)): #(k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                print("Discretization")
                KK=k+1
                ind3=1

            if indd==1: # (ind == 1) & (k>=dd):
                '''
                list_zero = []
                list_nonzero = []
                x_nonzero = []
                thres=sorted(abs(x_old[:, i]), reverse=True)
                for j in range(T):
                    if abs(x_old[j, i]) < 0.001:
                        list_zero.append(j)
                        v_temp[j, i] = 0
                    else:
                        list_nonzero.append(j)
                        x_nonzero.append(x_old[j, i])
                x_nonzero = np.array(x_nonzero)
                num_nonzero=x_nonzero.shape[0]
                CC_i=np.ones((M, num_nonzero))
                for j in range(num_nonzero):
                    CC_i[:, j]=C_i[:, list_nonzero[j]]
                B_i = np.dot(CC_i, np.diag(x_nonzero))
                [U, S, VV] = np.linalg.svd(B_i)
                thres=np.where(S>0.0001)[0].shape[0]
                conv_list = np.ones(num_nonzero)
                conv_list1 = []
                for j in range(num_nonzero):
                    conv_list[j] = np.linalg.norm(np.dot(U.T, B_i[:, j]), 2)
                    conv_list1.append(conv_list[j])

                # B_i=CC_i
                ''''''
                B_i_d = np.dot(B_i.T, B_i)
                aa=np.linalg.eig(B_i_d)[0].real
                aa_thres=np.where(aa>=0.0005)[0].shape[0]
                P=np.dot(np.dot(B_i, np.linalg.pinv(B_i_d)), B_i.T)
                r=np.dot(np.identity(M)-P, Y_i)
                conv_list=np.ones(num_nonzero)
                for j in range(num_nonzero):
                    conv_list[j]=(np.dot(B_i[:, j].T, r))
                ''''''
                list_nonzero_new=[]
                x_nonzero_new=[]
                for j in range(num_nonzero):
                    tt=sorted(conv_list1, reverse=True)[thres]
                    if conv_list[j]<=tt:
                        list_zero.append(list_nonzero[j])
                        v_temp[list_nonzero[j], i] = 0
                    else:
                        list_nonzero_new.append(list_nonzero[j])
                        x_nonzero_new.append(x_old[list_nonzero[j], i])
                num_nonzero_new = len(list_nonzero_new)
                CC_i_new = np.ones((M, num_nonzero_new))
                for j in range(num_nonzero_new):
                    CC_i_new[:, j] = C_i[:, list_nonzero_new[j]]
                B_i_new = np.dot(CC_i_new, np.diag(x_nonzero_new))
                # B_i=CC_i
                B_i_d_new = np.dot(B_i_new.T, B_i_new)
                aa_new = np.linalg.eig(B_i_d_new)[0].real
                a = (alpha * 2 * min(np.linalg.eig(B_i_d_new)[0])).real
                # print(a)
                part1=np.linalg.pinv(2*B_i_d_new - a*np.identity(num_nonzero_new))
                part2=np.dot(B_i_new.T, Y_i) -  a*np.ones(num_nonzero_new)
                v_tempp=np.dot(part1, part2)
                for j in range(num_nonzero_new):
                    v_temp[list_nonzero_new[j], i]=v_tempp[j]
                '''
                B_i = np.dot(C_i, np.diag(x_e[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = alpha * B_i
                M_i_d = np.dot(M_i.T, M_i)
                if iid==5:
                    part1 = np.linalg.pinv(B_i_d - M_i_d)
                    part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5 * np.ones(T))
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] =map_v_hard1(v_temp[:, i], iid)
                else:
                    part1 = np.linalg.pinv(B_i_d)
                    part2 = np.dot(B_i.T, Y_i)
                    v_temp[:, i] = np.dot(part1, part2)
                    v_e[:, i] = map_v_hard1(v_temp[:, i], iid)

            # ii2[k, i] = np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2)
            # if ii2[k, i]<=0.25:
            #   ind4[i]=0
            loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))   ##原本的几个实验里面loss的值有点问题
            # print("node %d: the mseracy for common x is %10.4f, the mseracy for local x is %10.4f, "
                  # "the loss is %10.4f\n" % (i, com_mse[i], loc_mse[i], loss[i]))
        if ind==1:
            indd=1
        else:
            indd=0

        if ind3==1:
            iid=1

        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f, loss: %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k], avr_loss[k]))

        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2

        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        # 如果连着十次迭代变化都小于0.0001，那么就停止迭代，判断收敛到局部极小值点
        # (avr_com_mse[k] - avr_com_mse[k - 1]>=0.00001):
        '''
        if (ind5==1) & (k>=K_end+10) & (abs(avr_com_mse[k] - avr_com_mse[k - 10]) < 0.0001):
            iter_num = k + 1
            print("Converge to local optimal")
            break

        if (avr_com_mse[k]<0.00001) & (cserr[k]<0.1):
            iter_num = k + 1
            print("get to the optimized point")
            break'''

        if (k==99): # & (avr_com_mse[k]<0.02):
            iter_num = k + 1
            print("Do not need 200 Iterations")
            break

    num_correct=np.zeros(N)
    ratio_correct=np.zeros(N)
    for i in range(N):
        for t in range(T):
            if v_e[t, i]==V_array[t, i]:
                num_correct[i]=num_correct[i]+1
        ratio_correct[i]=1.0*num_correct[i]/T

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter, np.average(ratio_correct)


def decentral_l1_VR_l1_soft(A, Y, X, V, N, Adjacent_Matrix, c, gamma):  # VR: vector recovery
    # 使用对v的l1范数，考虑软判决
    QUIET=0
    MAX_ITER=100
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))
    r = np.zeros((MAX_ITER, N))
    s = np.zeros((MAX_ITER, N))
    thres = np.zeros((MAX_ITER, N))
    num_nonzerox = np.ones((MAX_ITER, N))


    C=A
    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)
    ind = 0
    dd = MAX_ITER-1
    dd2 = (MAX_ITER-1)*np.ones(N)
    ind2 = 0
    ind3=np.ones(N)
    lenn=10*np.ones((MAX_ITER, N))
    gamma2=2.0*gamma*np.ones(N)


    for k in range(MAX_ITER):
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            v_neig = v_old[:, neig_location]
            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            r[k, i] = np.linalg.norm(x_old[:, i] - np.average(x_neig, axis=1), ord=2)
            if k > 0:
                s[k, i] = c * np.linalg.norm(np.average(x_neig, axis=1) - \
                                             np.average(x_oldd[:, neig_location], axis=1), ord=2)
                # print(r[k, i], s[k, i], r[k, i]/s[k,i], s[k,i]/r[k,i])

            if  (ind2==0) &(k > 1) & (r[k, i] ** 2 >= 3.0 * s[k, i] ** 2):
                ind = 1
                dd = k+1
                ind2 = 1
                dd2 = (k+1)*np.ones(N)
                print("****************the update of v begins********************")

            if (k>=dd) & (lenn[k-1, i]<=0):
                print("stop updating v of node %d" %i)
                # ind = 0
                ind3[i] = 0

            if (k>=dd) & (ind==1) & (ind3[i]==1) : # ind==1: # ind[i]==1:
                B_i = np.dot(C_i, np.diag(x_old[:, i]))
                B_i_d = np.dot(B_i.T, B_i)
                M_i = 0.3 * B_i
                M_i_d = np.dot(M_i.T, M_i)
                # v_temp[:, i] = upd.update_v_l1_penalty(Y_i, B_i, gamma2[i], 0.3)
                    # upd.update_v_l1(Y_i, B_i, 0.1, 0.3)
                # B_i = np.dot(C_i, np.diag(x_old[:, i]))
                v_temp[:, i] = upd.update_v_l1(Y_i, B_i, gamma2[i])
                print(i)
                if k<=20:
                    v_e[:, i]=map_v_hard2(v_temp[:, i])
                else:
                    v_e[:, i], lenn[k, i], num_one= map_v_soft(v_temp[:, i], x_neig, x_old[:, i], v_neig)
                # iv.MP_inference_v_iter(Y_i, B_i, x_e[:, i], 0.01, 1, 0.2)

            num_nonzerox[k, i]=np.where(abs(x_old[:, i])>0.0001)[0].shape[0]
            if (ind3[i]==1) & (k>=dd+1) & (lenn[k-1, i]>0) & (gamma2[i]>=0.1*gamma) & (np.sqrt(np.log(num_nonzerox[int(dd2[i]), i]))-np.sqrt(np.log(num_nonzerox[k, i]))>=0.02): # ((k-dd)%5==0): #(lenn[k, i]>lenn[k-1, i]):
                gamma2[i]=gamma2[i]-0.1*gamma
                dd2[i]=k
                print("change gamma2 of %d" %i +"to %10.3f" %gamma2[i])

            # update of x   v_old[:, i]
            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_e[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)

            # update of v
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            # v_e[:, i]=V_i
            # com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))   ##原本的几个实验里面loss的值有点问题
            # print("node %d: the mseracy for common x is %10.4f, the mseracy for local x is %10.4f, "
                  # "the loss is %10.4f\n" % (i, com_mse[i], loc_mse[i], loss[i]))

        x_oldd = copy.deepcopy(x_old)
        x_tilde = np.mean(x_e, axis=1)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f, loss:  %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k], avr_loss[k]))

        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2
        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        if (avr_com_mse[k]<0.001) & (cserr[k]<0.1):
            iter_num=k+1
            print("get to the optimized point")
            break

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter, v_e

#################################### not efficient ###########################

def decentral_l1_VR_penalty_c_nopenalty(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma):  # VR: vector recovery
    QUIET=0
    MAX_ITER=300
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))

    C=A
    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    cserr = np.ones(MAX_ITER)

    for k in range(MAX_ITER):
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            # update of x
            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)

            # update of v
            B_i = np.dot(C_i, np.diag(x_e[:, i]))
            B_i_d = np.dot(B_i.T, B_i)
            # alpha2 = 0.5
            M_i = alpha * B_i
            M_i_d = np.dot(M_i.T, M_i)
            part1 = np.linalg.pinv(B_i_d)
            part2 = np.dot(B_i.T, Y_i) # - np.dot(M_i_d, 0.5 * np.ones(T))
            v_temp[:, i] = np.dot(part1, part2)
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            v_e[:, i] = map_v_soft(np.dot(part1, part2), x_neig, x_e[:, i], k, com_mse[i], V_i)

            # v_e[:, i]=V_i
            loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))   ##原本的几个实验里面loss的值有点问题
            # print("node %d: the mseracy for common x is %10.4f, the mseracy for local x is %10.4f, "
                  # "the loss is %10.4f\n" % (i, com_mse[i], loc_mse[i], loss[i]))

        x_tilde = np.mean(x_e, axis=1)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))

        mse_total[k] = 1.0/T * np.linalg.norm(x_tilde-X, 2)**2

        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        if (avr_com_mse[k]<0.003) & (cserr[k]<0.5):
            iter_num=k+1
            print("get to the optimized point")
            break

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter


def decentral_l1_VR_penalty_l1(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma):  # VR: vector recovery
    QUIET=0
    MAX_ITER=200
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    v_e=np.ones((T, N))
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))

    '''one_m=np.ones((len(b),1))
    C=np.concatenate((one_m, A), axis=1)  #m*(k+1)'''
    C=A
    judg = 10 ** (-2)

    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    # dis_to_opt = np.zeros((MAX_ITER))
    cserr = np.ones(MAX_ITER)
    obj = np.ones(MAX_ITER)

    for k in range(MAX_ITER):
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]

            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            # update of x
            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)

            # update of v
            B_i = np.dot(C_i, np.diag(x_e[:, i]))
            B_i_d = np.dot(B_i.T, B_i)
            ss=np.linalg.eig(B_i_d)[0]

            '''
            alpha = 2*min(np.linalg.eig(B_i_d)[0])
            if alpha<(-0.01):
                print("############# no alpha exists for convex optimiztion #############\n")
                return 0
            elif alpha>0:
                alpha = alpha*0.8
            else:
                alpha=0
            part1=np.linalg.pinv(B_i_d-alpha*np.identity(T))
            part2=np.dot(B_i.T, Y_i)-alpha*0.5*np.ones(T)
            v_temp[:, i] = np.dot(part1, part2)
            v_e[:, i] = map_v(np.dot(part1, part2), x_neig, x_e[:, i ])
            '''

            # alpha2 = 0.5
            M_i = alpha*B_i
            M_i_d = np.dot(M_i.T, M_i)
            part1 = np.linalg.pinv(B_i_d - M_i_d)
            part2 = np.dot(B_i.T, Y_i) - np.dot(M_i_d, 0.5*np.ones(T))
            v_temp[:, i] = np.dot(part1, part2)
            v_e[:, i] = upd.update_v_l1(Y_i, B_i, 0.1, alpha)

            # v_e[:, i]=V_i
            com_mse[i] = 1.0/T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))   ##原本的几个实验里面loss的值有点问题
            # print("node %d: the mseracy for common x is %10.4f, the mseracy for local x is %10.4f, "
                  # "the loss is %10.4f\n" % (i, com_mse[i], loc_mse[i], loss[i]))
        x_oldd = copy.deepcopy(x_old)

        x_tilde = np.mean(x_e, axis=1)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))


        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        if (avr_com_mse[k]<0.005) & (cserr[k]<1.0):
            iter_num=k+1
            print("get to the optimized point")
            break

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    avr_loss[maxiter: MAX_ITER] = avr_loss[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    '''
    X_axi = np.linspace(1, min(MAX_ITER, iter_num), min(MAX_ITER, iter_num))
    figure1 = plt.figure(1)
    ax1 = plt.subplot(121)
    plt.plot(X_axi, avr_com_mse)
    plt.title("avr_com_mse")
    plt.xlabel("Iteration")
    plt.ylabel("avr_com_mse")
    plt.legend()

    ax2 = plt.subplot(122)
    plt.plot(X_axi, avr_loc_mse)
    plt.title("avr_loc_mse")
    plt.xlabel("Iteration")
    plt.ylabel("avr_loc_mse")
    plt.legend()

    figure2 = plt.figure(2)
    plt.plot(X_axi, avr_loss)
    plt.title("avr_loss")
    plt.xlabel("Iteration")
    plt.ylabel("avr_loss")
    plt.legend()

    figure3 = plt.figure(3)
    ax3 = plt.subplot(121)
    plt.plot(X_axi, mse_total)
    plt.title("mse_total")
    plt.xlabel("Iteration")
    plt.ylabel("mse_total")
    plt.legend()

    ax4 = plt.subplot(122)
    plt.plot(X_axi, cserr)
    plt.title("cserr")
    plt.xlabel("Iteration")
    plt.ylabel("cserr")
    plt.legend()

    plt.show()
    '''

    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter


def decentral_l1_VR_logistic_multigrad(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma):  # VR: vector recovery
    QUIET=0
    MAX_ITER=200
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    # v_e = 0.5 * np.ones((T, N))
    tau = 1.0
    v_e = np.zeros((T, N))
    w_e = 1.0 * np.zeros((T, N))
    for i in range(N):
        v_e[:, i]=sigmoid(w_e[:,i], tau)
    v_temp=np.ones((T, N))
    p_e=np.zeros((T, N))

    '''one_m=np.ones((len(b),1))
    C=np.concatenate((one_m, A), axis=1)  #m*(k+1)'''
    C=A

    # compute the objective in centralized way
    # x_opt, obj_opt=central_l1_FISTA(C, Y, np.zeros((K, 1)), 10**(-20))
    # there is no centralized implementation because of the different measuring dimensions
    # so what to serve as the comparsion ???

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    # dis_to_opt = np.zeros((MAX_ITER))
    cserr = np.ones(MAX_ITER)
    obj = np.ones(MAX_ITER)

    for k in range(MAX_ITER):
        tau = tau * 1.0
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        w_old = copy.deepcopy(w_e)
        '''
        if k%3==0:
            alpha = alpha * 0.99'''
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]
            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            # update of x
            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c, x_old[:, i], x_neig, gamma)

            # update of v: consists of the update of w and v
            ## the update of w:
            B_i = np.dot(C_i, np.diag(x_e[:, i]))
            w_e[:, i] = update_v(alpha, Y_i, B_i, tau, w_old[:, i], v_old[:, i])
            v_e[:, i] = map_v_soft(sigmoid(w_e[:, i], tau), x_neig, x_e[:, i], k, com_mse[i], V_i)

            # v_e[:, i]=V_i
            com_mse[i] = 1.0/T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loc_mse[i] = 1.0/T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))
            # print("node %d: the mse for common x is %10.4f, the mse for local x is %10.4f, "
            #      "the loss is %10.4f\n" % (i, com_mse[i], loc_mse[i], loss[i]))

        x_tilde = np.mean(x_e, axis=1)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k]))

        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        if (avr_com_mse[k]<0.008) & (cserr[k]<0.1):
            iter_num=k+1
            print("get to the optimized point")
            break

    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    avr_loss[maxiter: MAX_ITER] = avr_loss[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]

    '''
    X_axi = np.linspace(1, min(MAX_ITER, iter_num), min(MAX_ITER, iter_num))
    figure1 = plt.figure(1)
    ax1 = plt.subplot(121)
    plt.plot(X_axi, avr_com_mse)
    plt.title("avr_com_mse")
    plt.xlabel("Iteration")
    plt.ylabel("avr_com_mse")
    plt.legend()

    ax2 = plt.subplot(122)
    plt.plot(X_axi, avr_loc_mse)
    plt.title("avr_loc_mse")
    plt.xlabel("Iteration")
    plt.ylabel("avr_loc_mse")
    plt.legend()

    figure2 = plt.figure(2)
    plt.plot(X_axi, avr_loss)
    plt.title("avr_loss")
    plt.xlabel("Iteration")
    plt.ylabel("avr_loss")
    plt.legend()

    figure3 = plt.figure(3)
    ax3 = plt.subplot(121)
    plt.plot(X_axi, mse_total)
    plt.title("mse_total")
    plt.xlabel("Iteration")
    plt.ylabel("mse_total")
    plt.legend()

    ax4 = plt.subplot(122)
    plt.plot(X_axi, cserr)
    plt.title("cserr")
    plt.xlabel("Iteration")
    plt.ylabel("cserr")
    plt.legend()

    plt.show()
    '''
    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter


def decentral_l1_VR_logistic_onegrad(A, Y, X, V, N, Adjacent_Matrix, alpha, c, gamma, mu):  # VR: vector recovery
    QUIET=0
    MAX_ITER=250
    iter_num=10000
    m=np.size(A, 0)
    T=np.size(A, 1)
    #print K
    M=int(m/N)
    V_array = (np.reshape(V, [N, T])).T
    x_e=np.zeros((T, N))
    # the initialzation of vi (可以先全部初始化为1，但是一种更加有效的，对于稀疏向量的方法是采用greedy pursuit的思想，将vi的无关
    # 维度置为0或者较小的值，之后再进行更新的话，可能会有效降低迭代次数)
    # v_e = 0.5 * np.ones((T, N))
    ind = 1
    ind3 = 0
    ii = 0
    iid = 0
    tau = 1.0
    v_e = np.zeros((T, N))
    grad_ww = np.zeros((T, N))
    w_e = np.random.randn(T, N)
    for i in range(N):
        v_e[:, i]=sigmoid(w_e[:,i], tau)
    p_e=np.zeros((T, N))
    c_e = 1.0 * c * np.ones(N)

    C=A

    mse_total = np.zeros(MAX_ITER)
    avr_com_mse = np.zeros(MAX_ITER)
    avr_loc_mse = np.zeros(MAX_ITER)
    avr_loss = np.zeros(MAX_ITER)
    # dis_to_opt = np.zeros((MAX_ITER))
    cserr = np.ones(MAX_ITER)
    obj = np.ones(MAX_ITER)

    for k in range(MAX_ITER):
        # tau =2.0*np.sqrt(k)
        print("iteration %d\n" % k)
        loss = np.zeros(N)
        loc_mse = np.zeros(N)
        com_mse = np.zeros(N)
        p_old = copy.deepcopy(p_e)
        c_old = copy.deepcopy(c_e)
        x_old = copy.deepcopy(x_e)
        v_old = copy.deepcopy(v_e)
        w_old = copy.deepcopy(w_e)
        alpha = alpha*1.2
        for i in range(N):
            neig_location = np.transpose(np.nonzero(Adjacent_Matrix[:, i]))[:, 0]
            x_neig = x_old[:, neig_location]


            if (k>=1) & (k<=30):
                r = np.linalg.norm(x_e[:, i]-np.average(x_neig, axis=1), 2)
                s = c_old[i] * np.linalg.norm(np.average(x_neig, axis=1)-np.average(x_oldd[:, neig_location], axis=1), 2)
                if r**2>10*(s**2):
                    c_e[i] = c_old[i] * 2.0
                elif s**2>10*(r**2):
                    c_e[i] = c_old[i] * 0.5
                else:
                    c_e[i] = c_old[i]
            else:
                c_e[i]=c_old[i]


            # update of p
            for j in neig_location:
                p_e[:, i] = p_old[:, i]+ c_e[i]*(x_old[:, i]-x_old[:, j])
            #print p[:,i].shape

            # the local measurement matrix and the measurement result
            C_i = C[i * M: (i + 1) * M, :]
            Y_i = Y[i * M: (i + 1) * M]
            V_i = V[i * T: (i + 1) * T]

            # update of x
            x_e[:, i], mse = upd.update_x_ADMM(Y_i, C_i, v_old[:, i], p_e[:, i], c_e[i], x_old[:, i], x_neig, gamma)

            if (ind==1) & (ind3==0) & (k>=2):
                # print(np.linalg.norm(x_old[:, i] - x_oldd[:, i], ord=2))
                # print(np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2))
                ii = (np.linalg.norm(x_e[:, i] - x_old[:, i], ord=2)<=mu)
            if (ind3==0) & ((ii==1)|(k>=20)): #(k>=7) & ((r[k, i] ** 2 > 10.0 * s[k, i] ** 2)):
                print("Discretization")
                ind3=1

            # update of v: consists of the update of w and v
            ## the update of w:
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            B_i = np.dot(C_i, np.diag(x_e[:, i]))
            grad_ww[:, i] = grad_w(Y_i, B_i, tau, w_old[:, i])
            w_e[:, i] = w_old[:, i] - alpha*grad_ww[:, i]
            v_e[:, i] = map_v_hard3(sigmoid(w_e[:, i], tau), 0, x_e[:, i])
            # v_e[:, i] = sigmoid(w_e[:, i], tau)

            loc_mse[i] = 1.0 / T * np.linalg.norm(np.multiply(v_e[:, i], x_e[:, i]) - np.multiply(V_i, X), 2) ** 2
            com_mse[i] = 1.0 / T * np.linalg.norm(x_e[:, i] - X, 2) ** 2
            loss[i] = loss_fn(C_i, Y_i, np.multiply(v_e[:, i], x_e[:, i]))

        if ind3==1:
            iid=1
        x_tilde = np.mean(x_e, axis=1)
        x_oldd = copy.deepcopy(x_old)
        avr_com_mse[k] = np.average(com_mse)
        avr_loc_mse[k] = np.average(loc_mse)
        avr_loss[k] = np.average(loss)
        cserr[k] = ((np.linalg.norm(np.asmatrix(x_tilde).T - x_e, 'fro')) ** 2) / N
        print("end of iteration %d, avr_com_mse: %10.4f, avr_loc_mse: %10.4f, cserr:  %10.4f, loss: %10.4f\n"
              % (k, avr_com_mse[k], avr_loc_mse[k], cserr[k], avr_loss[k]))

        mse_total[k] = 1.0 / T * np.linalg.norm(x_tilde - X, 2) ** 2

        # convergence2 conditions:
        # if (k>1) & () 不收敛的停止准则 print; break
        if (avr_com_mse[k]<0.001) & (cserr[k]<0.1):
            iter_num=k+1
            print("get to the optimized point")
            break

    num_correct = np.zeros(N)
    ratio_correct = np.zeros(N)
    for i in range(N):
        for t in range(T):
            if v_e[t, i] == V_array[t, i]:
                num_correct[i] = num_correct[i] + 1
        ratio_correct[i] = 1.0 * num_correct[i] / T
    maxiter=min(MAX_ITER, iter_num)
    avr_com_mse[maxiter: MAX_ITER]=avr_com_mse[maxiter-1]
    avr_loc_mse[maxiter: MAX_ITER] = avr_loc_mse[maxiter - 1]
    avr_loss[maxiter: MAX_ITER] = avr_loss[maxiter - 1]
    mse_total[maxiter: MAX_ITER] = mse_total[maxiter - 1]
    cserr[maxiter: MAX_ITER] = cserr[maxiter - 1]
    return avr_com_mse, avr_loc_mse, mse_total, cserr, maxiter, np.average(ratio_correct)

