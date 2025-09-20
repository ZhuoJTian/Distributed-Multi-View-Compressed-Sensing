import numpy as np
import random
import copy

def dele_list(list1, list2):
    new_l=[]
    for i in list1:
        if i not in list2:
            new_l.append(i)
    return new_l


def gener_a(M, T, N):
    a_stack = np.zeros((M*N, T))  # measure/sensing matrix
    for i in range(M*N):
        for j in range(T):
            a_stack[i, j] =np.random.normal(0, scale=1.0/np.sqrt(M))
    return a_stack


def gener_a_one(M_one, M, T, N):
    a_stack = np.zeros((M_one+M*(N-1), T))  # measure/sensing matrix
    for i in range(M_one):
        for j in range(T):
            a_stack[i, j] =np.random.normal(0, scale=1.0/np.sqrt(M_one))
    for i in range(M_one, M_one+M*(N-1)):
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


def gener_v_bad(X_est, N, T, p_block):
    v_stack = np.zeros((T * N))
    v_array = np.zeros((T, N))
    non_zero_x = np.where(X_est != 0)[0]
    for i in range(N-1):
        for j in range(T):
            if X_est[j] == 0:
                v_array[j, i] = 0
                v_stack[i * T + j] = 0
            else:
                v_array[j, i] = np.random.binomial(1, 1 - p_block)
                v_stack[i * T + j] = v_array[j, i]

    if np.any(np.sum(v_array[list(non_zero_x), :], axis=1) == 0):
        index=np.where(np.sum(v_array[list(non_zero_x), :], axis=1) == 0)[0]
        index2=non_zero_x[index]
        for i in index2:
            v_array[i, N-1] = 1
            v_stack[(N-1) * T +i] = v_array[i, N-1]
    return v_stack


def gener_v_worst(X_est, N, T):
    v_stack = np.zeros((T * N))
    v_array=np.zeros((T, N))
    non_zero_x=np.where(X_est!=0)[0]
    non_list=list(non_zero_x)
    new_l=non_list
    for i in range(N-1):
        l=random.sample(new_l, 8)
        for ll in l:
            v_array[ll, i] = 1
            v_stack[i * T + ll] = 1
        new_l=dele_list(new_l, l)
    for ll in new_l:
        v_array[ll, N-1] = 1
        v_stack[(N-1) * T + ll] = 1
    return v_stack


def gener_v_one(X_est, N, T, p, b_one):
    v_stack = np.zeros((T * N))
    v_array = np.zeros((T, N))
    non_zero_x = np.where(X_est != 0)[0]
    non_list = list(non_zero_x)
    K = len(non_list)
    new_l = non_list
    # for node 0
    l = random.sample(new_l, b_one)
    for ll in l:
        v_array[ll, 0] = 1
        v_stack[ll] = 1
    new_l = dele_list(new_l, l)
    # for other nodes
    for i in range(N-1):
        for j in new_l:
            v_array[j, i + 1] = np.random.binomial(1, 1 - p)
            v_stack[(i + 1) * T + j] = v_array[j, i + 1]

    while np.any(np.sum(v_array[list(non_zero_x), :], axis=1)==0):
        for i in range(N - 1):
            for j in new_l:
                v_array[j, i + 1] = np.random.binomial(1, 1 - p)
                v_stack[(i + 1) * T + j] = v_array[j, i + 1]

    for j in new_l:
        v_array[j, 0] = np.random.binomial(1, 1 - p - b_one/K)
        v_stack[j] = v_array[j, 0]

    return v_stack


def generate_MVSVR(M, T, N, K, d_blocked):
    # m is the dimension of y in each node
    # n is the dimension of x
    # N is the number of nodes
    # K is the sparsity of x, i.e., the number of non-zero entries in x
    # d_blocked is the number of blocked dimensions observed by each node: d_blocked<K
                    # (can be extended to be different among nodes)
    # stack measurement matrix of all nodes: a_stack (m*N, n) using random i.i.d. Gaussian matrices
    # x to be estimated: X_est (n, 1)
    # stack measurement result of all nodes: y_stack (m*N, 1)
    # stack blockage vector: v_stack (n*N, 1)

    a_stack = np.zeros((M*N, T))  # measure/sensing matrix
    for i in range(M*N):
        for j in range(T):
            a_stack[i, j] =np.random.normal(0, scale=1.0)

    X_est = np.zeros((T))
    non_zero_x=random.sample(range(T), K)
    for j in non_zero_x:
        X_est[j] = np.random.normal(0, scale=1)   #2*np.random.random()-1

    v_stack = np.zeros((T*N))
    for i in range(N):
        non_zero_v = [non_zero_x[ii] for ii in random.sample(range(K), K-d_blocked)]
        for j in non_zero_v:
            v_stack[i*T+j]=1

    y_stack = np.zeros((M*N))
    noise_add=np.random.normal(loc=0, scale=0.1, size=M*N) # add noise?
    for i in range(N):
        x_o=np.multiply(v_stack[i*T: (i+1)*T], X_est)
        y_stack[i*M: (i+1)*M]=np.dot(a_stack[i*M: (i+1)*M, :], x_o)

    V_array = (np.reshape(v_stack, [N, T])).T

    return a_stack, X_est, y_stack, v_stack


def generate_MVSVR_noise(M, T, N, K, d_blocked):
    # m is the dimension of y in each node
    # n is the dimension of x
    # N is the number of nodes
    # K is the sparsity of x, i.e., the number of non-zero entries in x
    # d_blocked is the number of blocked dimensions observed by each node: d_blocked<K
                    # (can be extended to be different among nodes)
    # stack measurement matrix of all nodes: a_stack (m*N, n) using random i.i.d. Gaussian matrices
    # x to be estimated: X_est (n, 1)
    # stack measurement result of all nodes: y_stack (m*N, 1)
    # stack blockage vector: v_stack (n*N, 1)

    a_stack = np.zeros((M*N, T))  # measure/sensing matrix
    for i in range(M*N):
        for j in range(T):
            a_stack[i, j] =np.random.normal(0, scale=1)

    X_est = np.zeros((T))
    non_zero_x=random.sample(range(T), K)
    for j in non_zero_x:
        X_est[j] = np.random.normal(0, scale=1)   #2*np.random.random()-1

    v_stack = np.zeros((T*N))
    for i in range(N):
        non_zero_v = [non_zero_x[ii] for ii in random.sample(range(K), K-d_blocked)]
        for j in non_zero_v:
            v_stack[i*T+j]=1

    y_stack = np.zeros(M*N)
    noise_add=np.random.normal(loc=0, scale=0.01, size=M*N) # add noise?
    for i in range(N):
        x_o=np.multiply(v_stack[i*T: (i+1)*T], X_est)
        y_stack[i*M: (i+1)*M]=np.dot(a_stack[i*M: (i+1)*M, :], x_o)

    y_stack=y_stack+noise_add
    V_array = (np.reshape(v_stack, [N, T])).T

    return a_stack, X_est, y_stack, v_stack


'''
# compare_all_convergence(20average)
M=80
T=500
N=6
K=50
d_blocked=20
num_a=1
num_x=1
num_v=20
for i in range(num_a):
    a_stack=gener_a(M, T, N)
    np.savetxt('./Compare_all_convergence(20average)/Data_sample3/data_A'+'%d.txt' %i, a_stack)
for i in range(num_x):
    X_est=gener_x(T, N, K, d_blocked)
    np.savetxt('./Compare_all_convergence(20average)/Data_sample3/data_X' + '%d.txt' % i, X_est)
for i in range(num_v):
    v_stack = gener_v(X_est, N, T, K, d_blocked)
    np.savetxt('./Compare_all_convergence(20average)/Data_sample3/data_V' + '%d.txt' % i, v_stack)
'''

'''
M=20
T=100
N=4
K=10
d_blocked=3
a_stack, X_est, y_stack, v_stack = generate_MVSVR_noise(M, T, N, K, d_blocked)
np.savetxt("data_A_n.txt", a_stack)
np.savetxt('data_X_n.txt', X_est)
np.savetxt('data_Y_n.txt', y_stack)
np.savetxt('data_V_n.txt', v_stack)
'''

'''
# different M
M=[60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
# M=[40]
p_block=[0.2]
ep_block=10
T=500
N=6
K=50
# d_blocked=10
num_a=5
num_x=5
num_v=4

for i in range(num_x):
    X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_%d.txt' % i)
    for p in p_block:
        for j in range(num_v):
            v_stack = gener_v(X_est, N, T, K, p)
            np.savetxt('./new_m_compareall_noise/Data_Sample/d%d' % ep_block +'/data_V_%d' % i + '_%d.txt' % j, v_stack)

for m in M:
    for i in range(num_a):
        a_stack=gener_a(m, T, N)
        np.savetxt('./new_m_compareall_noise/Data_Sample/data_A_%d' %m+'_%d.txt'%i, a_stack)
'''

'''
# different d_block
M=80
T=500
N=6
K=50
p_blocked=[0.8]  #[0, 5, 10, 15, 20, 25]
d_blocked=[40]
num_a=5
num_x=5
num_v=4

for i in range(num_x):
    X_est=np.loadtxt('./new_d_compareall_noise/Data_Sample/data_X_%d.txt' % i)
    for m in range(len(p_blocked)):
        for j in range(num_v):
            v_stack = gener_v_bad(X_est, N, T, p_blocked[m])
            np.savetxt('./new_d_compareall_noise/Data_Sample/d%d'%d_blocked[m] +'/data_V_%d' % i +'_%d.txt' % j, v_stack)
'''

'''
# worst blockage case, no overlapping of measurements
M=[150, 200, 250]
T=500
N=6
K=50
num_a=5
num_x=5
num_v=4

for m in M:
    for i in range(num_a):
        a_stack = gener_a(m, T, N)
        np.savetxt('./new_compare_worstd/Data_Sample/data_A_%d' % m + '_%d.txt' % i, a_stack)

for i in range(num_x):
    X_est = np.loadtxt('./new_compare_worstd/Data_Sample/data_X_%d.txt' % i)
    for j in range(num_v):
        v_stack = gener_v_worst(X_est, N, T)
        np.savetxt('./new_d_compareall_noise/Data_Sample/data_V_%d' % i + '_%d.txt' % j,
                   v_stack)
'''

'''
# different N
M = 60
N = [4, 6, 8, 10, 12, 14, 16]
p_block = [0.4]
ep_block = 20
T = 500
K = 50
# d_blocked=10
num_a=5
num_x=2
num_v=4

for n in N:
    for i in range(num_x):
        X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_%d.txt' % i)
        for p in p_block:
            for j in range(num_v):
                v_stack = gener_v(X_est, n, T, K, p)
                v_array = (np.reshape(v_stack, [n, T])).T
                non_zero_x = np.where(X_est != 0)[0]
                while (np.any(np.sum(v_array[list(non_zero_x), :], axis=1) == 0)):
                    v_stack = gener_v(X_est, n, T, K, p)
                    v_array = (np.reshape(v_stack, [n, T])).T
                np.savetxt('./Node/Data_Sample/%d'%n + '/data_V_%d' % i + '_%d.txt' % j, v_stack)
    
    for i in range(num_a):
        a_stack = gener_a(M, T, n)
        np.savetxt('./Node/Data_Sample/%d'%n +'/data_A_%d' %M+'_%d.txt'%i, a_stack)
'''

'''
# unbalanced M
M_List = [50, 60, 70, 80, 90, 100] # 60 70 70 100 80 90
N = 6
T = 500
num_a=5
num_x=2
num_v=4
for i in range(N):
    for j in range(num_a):
        m = random.choice(M_List)
        a_stack = gener_a(m, T, 1)
        np.savetxt('./new_unbalanceM/Data_Sample/Ai2/%d/'%i + 'data_A_%d.txt' % j, a_stack)

    for j in range(num_x):
        x = np.loadtxt('./new_unbalanceM/Data_Sample/data_X_%d.txt' % j)
        for k in range(num_v):
            v = np.loadtxt('./new_unbalanceM/Data_Sample/data_V_%d' % j +'_%d.txt' % k)
'''

'''
# Theorem 1, generate data where [5, 10, 15, 20, 25] indices are only observable for sensor 0.
B_one = [5, 10, 15, 20, 25]
p=0.2
T=500
N=6
K=50
num_a=5
num_x=2
num_v=4

for i in range(num_x):
    X_est = np.loadtxt('./new_T1/Data_Sample/data_X_%d.txt' % i)
    for b in B_one:
        for j in range(num_v):
            v_stack = gener_v_one(X_est, N, T, p, b)
            np.savetxt('./new_T1/Data_Sample/%d/'% b + 'data_V_%d' % i + '_%d.txt' % j, v_stack, fmt='%i')
print("a")

T=500
N=6
K=50
num_a = 5
M_0 = [150, 180, 210, 240, 270, 300, 330, 360] # M=100 for other nodes, so that their measurements are enough
for m in M_0:
    for i in range(num_a):
        a_stack = gener_a_one(m, 100, T, N)
        np.savetxt('./new_T1/Data_Sample/M/data_A_%d' % m+'_%d.txt' % i, a_stack)
'''