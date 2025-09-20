import numpy as np
import random
import copy

def gener_a(M, T, N):
    a_stack = np.zeros((M*N, T))  # measure/sensing matrix
    for i in range(M*N):
        for j in range(T):
            a_stack[i, j] =np.random.normal(0, scale=1.0/np.sqrt(M))
    return a_stack


def gener_x(T, K):
    X_est = np.zeros(T)
    non_zero_x = random.sample(range(T), K)
    for j in non_zero_x:
        X_est[j] = np.random.normal(0, scale=1)  # 2*np.random.random()-1
    return X_est


def gener_v(X_est, N, T, K, p_block):
    v_stack = np.zeros((T * N))
    v_array=np.zeros((T, N))
    non_zero_x=np.where(X_est!=0)[0]
    for i in range(N):
        for j in range(T):
            if X_est[j]==0:
                v_array[j, i]=0
                v_stack[i * T + j] = 0
            else:
                v_array[j, i]=np.random.binomial(1, 1-p_block)
                v_stack[i * T + j] = v_array[j, i]

    while np.any(np.sum(v_array[list(non_zero_x), :], axis=1)==0):
        for i in range(N):
            for j in range(T):
                if X_est[j] == 0:
                    v_array[j, i] = 0
                    v_stack[i * T + j] = 0
                else:
                    v_array[j, i] = np.random.binomial(1, 1 - p_block)
                    v_stack[i * T + j] = v_array[j, i]
    return v_stack


'''
# different M
# M=[70, 90, 110, 130, 150]
# M=[40]
p_block=[0.8]
ep_block=40
T=500
N=6
K=50
# d_blocked=10
num_a=5
num_x=5
num_v=4

for i in range(num_x):
    X_est = np.loadtxt('./new_d_compareall_noise/Data_Sample/data_X_%d.txt' % i)
    for p in p_block:
        for j in range(num_v):
            v_stack = np.loadtxt('./new_d_compareall_noise/Data_Sample/d%d' % ep_block +'/data_V_%d' % i + '_%d.txt' % j)
            v_array = (np.reshape(v_stack, [N, T])).T
            indd=np.sum(v_array, axis=0)
            non_zero_x = np.where(X_est != 0)[0]
            print(np.any(np.sum(v_array[list(non_zero_x), :], axis=1) == 0))
'''

