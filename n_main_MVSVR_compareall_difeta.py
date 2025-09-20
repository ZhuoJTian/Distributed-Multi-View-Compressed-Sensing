import numpy as np
import random
import ADMM_function
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#20, 30, 40, 50, 60
# 40, 50, 60, 70, 80, 90,
# [30, 50, 70, 90, 110]  #40, 60, 80, 100, 120
    #[20, 30, 40, 50, 60, 70, 80, 90] # the measurement size in each sensor
# M=[20, 40, 60, 80, 100, 120, 140, 160, 180]
# M=[40, 60, 80, 100, 120, 140] #10  #, 140, 160, 180,  200, 100, 120, 140
M=[80, 100]
M2=[60]
T=500
N=6
K=50
d=10 #5, 15, 25,   15和25的20,40，60还没跑
num_a=2   #2,2,4; 5,2,4
num_x=2
num_v=2
eta_set= [10, 14, 18, 22, 26] #] [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
# eta_set=[20]

repeat_times=num_a*num_x*num_v

Adjacent_Matrix=np.matrix([[0, 0, 1, 1, 1, 1]
                    , [0, 0, 1, 1, 1, 0]
                    , [1, 1, 0, 1, 0, 0]
                    , [1, 1, 1, 0, 1, 1]
                    , [1, 1, 0, 1, 0, 1]
                    , [1, 0, 0, 1, 1, 0]])

Max_iter=100
num_compare=2
'''
# 不跑JSM算法了
for m in M2:
    avr_com_mse = np.zeros((num_compare, repeat_times, Max_iter, len(eta_set)))
    avr_loc_mse = np.zeros((num_compare, repeat_times, Max_iter, len(eta_set)))
    mse_total = np.zeros((num_compare, repeat_times, Max_iter, len(eta_set)))
    cserr = np.zeros((num_compare, repeat_times, Max_iter, len(eta_set)))
    ratio_correct = np.zeros((num_compare, repeat_times, len(eta_set)))
    maxiter = np.zeros((num_compare, repeat_times, len(eta_set)))

    for t1 in range(num_a):
        a_stack = np.loadtxt(
            './new_compare_eta/Data_Sample/data_A_%d' % m + '_%d.txt' % t1)  # './new_d_compareall_noise/Data_Sample/
        for t2 in range(num_x):
            X_est = np.loadtxt('./new_compare_eta/Data_Sample/data_X_%d.txt' % t2)
            for t3 in range(num_v):
                v_stack = np.loadtxt('./new_compare_eta/Data_Sample/d%d' % d + '/data_V_%d' % t2 + '_%d.txt' % t3)
                y_stack = np.zeros(m * N)
                for i in range(N):
                    x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
                    y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
                    SNR = 12
                    s = 10 ** (-1.0 * (SNR / 10.0)) / T * (np.linalg.norm(x_o, ord=2) ** 2)
                    # s1=0.01/m*(np.linalg.norm(y_stack[i * m: (i + 1) * m], ord=2)**2)
                    noise_add = np.random.randn(m) * np.sqrt(s)
                    y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add

                for i in range(len(eta_set)):
                    eta=eta_set[i]
                    if m==60:
                        c2 = 1.0
                        gamma_new = 0.25
                        alpha = 0.2
                        mu = 0.2
                        c22 = 0.8
                        rho2 = 0.8
                        gamma12 = 0.012
                        gamma22 = 0.15
                        gamma32 = 0.009
                    elif m==80:
                        c2 = 1.0
                        gamma_new = 0.04
                        alpha = 0.2
                        mu = 0.2
                        c22 = 0.8
                        rho2 = 0.8
                        gamma12 = 0.012
                        gamma22 = 0.15
                        gamma32 = 0.009
                    else:
                        c2 = 1.0
                        gamma_new = 0.04
                        alpha = 0.2
                        mu = 0.2
                        c22 = 0.8
                        rho2 = 0.8
                        gamma12 = 0.012
                        gamma22 = 0.15
                        gamma32 = 0.009

                    print(m, eta, t1, t2, t3)
                    # import data

                    print("the ADMM-VPD:\n")
                    avr_com_mse[0, t1 * num_x * num_v + t2 * num_v + t3, :, i], avr_loc_mse[0,
                                                                                 t1 * num_x * num_v + t2 * num_v + t3, :, i], \
                    mse_total[0, t1 * num_x * num_v + t2 * num_v + t3, :, i], cserr[0,
                                                                               t1 * num_x * num_v + t2 * num_v + t3, :, i], \
                    maxiter[0, t1 * num_x * num_v + t2 * num_v + t3, i]\
                        = ADMM_function.decentral_l1_VR_penalty_c_1hard(a_stack, y_stack, X_est, v_stack, N,
                                                                            Adjacent_Matrix, alpha, c2,
                                                                            gamma_new, Max_iter, mu, eta)

                    print("VPD-JSM1 Algorithm")
                    avr_com_mse[1, t1 * num_x * num_v + t2 * num_v + t3, :, i], avr_loc_mse[1,
                                                                                 t1 * num_x * num_v + t2 * num_v + t3, :, i], \
                    mse_total[1, t1 * num_x * num_v + t2 * num_v + t3, :, i], cserr[1,
                                                                               t1 * num_x * num_v + t2 * num_v + t3, :, i], \
                    maxiter[1, t1 * num_x * num_v + t2 * num_v + t3, i]\
                        = ADMM_function.ADMM_VPD_JSM1(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, alpha,
                                                          c22, rho2, gamma12, gamma22, gamma32,
                                                          c2, gamma_new, Max_iter, mu, eta)

                    for j in range(num_compare):
                        np.savetxt('./new_compare_eta/convergence2/%d' %m+'/%d_' % eta + 'avr_com_mse%d.txt' % j,
                                       avr_com_mse[j, :, :, i])
                        np.savetxt('./new_compare_eta/convergence2/%d' %m+'/%d_' % eta + 'avr_loc_mse%d.txt' % j,
                                       avr_loc_mse[j, :, :, i])
'''

avr_c0=np.zeros((len(eta_set), len(M)))
avr_l0=np.zeros((len(eta_set), len(M)))
avr_c1=np.zeros((len(eta_set), len(M)))
avr_l1=np.zeros((len(eta_set), len(M)))

for j in range(len(eta_set)):
    for i in range(len(M)):
        m=M[i]
        eta=eta_set[j]
        avr_cc0 = np.loadtxt('./new_compare_eta/convergence2/%d' % m + '/%d_' % eta + 'avr_com_mse0.txt')
        avr_ll0 = np.loadtxt('./new_compare_eta/convergence2/%d' % m + '/%d_' % eta + 'avr_loc_mse0.txt')
        # cserr00 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr0.txt')

        avr_cc1 = np.loadtxt('./new_compare_eta/convergence2/%d' % m + '/%d_' % eta + 'avr_com_mse1.txt')
        avr_ll1 = np.loadtxt('./new_compare_eta/convergence2/%d' % m + '/%d_' % eta + 'avr_loc_mse1.txt')
        '''
        avr_c0[j, i] = avr_cc0[Max_iter - 1]
        avr_l0[j, i] = avr_ll0[Max_iter - 1]

        avr_c1[j, i] = avr_cc1[Max_iter - 1]
        avr_l1[j, i] = avr_ll1[Max_iter - 1]

        '''
        avr_c0[j, i] = np.average(avr_cc0[:, Max_iter - 1])
        avr_l0[j, i] = np.average(avr_ll0[:, Max_iter - 1])

        avr_c1[j, i] = np.average(avr_cc1[:,Max_iter - 1])
        avr_l1[j, i] = np.average(avr_ll1[:,Max_iter - 1])


X=eta_set
figure1=plt.figure(1)
plt.plot(X, avr_c0[:, 0], linewidth=1, label="ADMM-VPD(m=%d)"%M[0], color='b', linestyle='-.', marker='o')
plt.plot(X, avr_c1[:, 0], linewidth=1, label="VPD-CD(m=%d)"%M[0], color='b', linestyle=':', marker='>')
plt.plot(X, avr_c0[:, 1], linewidth=1, label="ADMM-VPD(m=%d)"%M[1], color='r', linestyle='-.', marker='*')
plt.plot(X, avr_c1[:, 1], linewidth=1, label="VPD-CD(m=%d)"%M[1], color='r', linestyle='--', marker='s')
#plt.plot(X, avr_c0[:, 2], linewidth=1, label="ADMM-VPD(m=%d)"%M[2], color='y', linestyle='-.', marker='<')
#plt.plot(X, avr_c1[:, 2], linewidth=1, label="VPD-CD(m=%d)"%M[2], color='y', linestyle='--', marker='+')
# plt.xlim(min(eta_set), max(eta_set))
# plt.ylim(0, 0.03)
# plt.title("avr_com_mse")
plt.xlabel("eta")
plt.ylabel("Glo_AMSE")
plt.grid(True)
plt.legend()

figure2=plt.figure(2)
plt.plot(X, avr_l0[:, 0], linewidth=1, label="ADMM-VPD(m=%d)"%M[0], color='b', linestyle='-.', marker='o')
plt.plot(X, avr_l1[:, 0], linewidth=1, label="VPD-CD(m=%d)"%M[0], color='b', linestyle=':', marker='>')
plt.plot(X, avr_l0[:, 1], linewidth=1, label="ADMM-VPD(m=%d)"%M[1], color='r', linestyle='-.', marker='*')
plt.plot(X, avr_l1[:, 1], linewidth=1, label="VPD-CD(m=%d)"%M[1], color='r', linestyle='--', marker='s')
#plt.plot(X, avr_l0[:, 2], linewidth=1, label="ADMM-VPD(m=%d)"%M[2], color='y', linestyle='-.', marker='<')
#plt.plot(X, avr_l1[:, 2], linewidth=1, label="VPD-CD(m=%d)"%M[2], color='y', linestyle='--', marker='+')
# plt.xlim(min(eta_set), max(eta_set))
# plt.ylim(0, 0.03)
# plt.title("avr_loc_mse")
plt.xlabel("eta")
plt.ylabel("Loc_AMSE")
plt.grid(True)
plt.legend()
plt.show()
