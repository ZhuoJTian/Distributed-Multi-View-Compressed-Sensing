# coding=utf-8
import numpy as np
import random
import ADMM_function
import matplotlib.pyplot as plt

m = 60
T = 500
N = [14]
K = 50
d = 20
num_a = 5  # 2,2,4; 5,2,4
num_x = 2
num_v = 4
repeat_times = num_a * num_x * num_v

Max_iter = 100
num_compare = 5

# fix the parameter of JSM1
c = 0.05
gamma1 = 0.0005
gamma2 = 0.05
gamma3 = 0.01
rho = 0.05
USTp=35
Umin=20

cs = 0.01
gammas = 0.07

for n in N:
    Adjacent_Matrix = np.loadtxt('./Node/Data_Sample/%d'%n + '/adj_%d'%n)
    if n==4:
        gamma_new_bl = 0.4
        c2 = 0.5
        gamma_new = 0.05
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.001
        gamma22 = 0.03
        gamma32 = 0.02
        alpha = 0.0
        eta = 20

    elif n==6:
        gamma_new_bl = 0.35
        c2 = 0.5
        gamma_new = 0.05
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.003
        gamma22 = 0.05
        gamma32 = 0.001
        alpha = 0.15
        eta = 20

    else:
        if n==8:
            gamma_new_bl = 0.35
        elif n==10:
            gamma_new_bl = 1.0  #1.0
        else:
            gamma_new_bl = 1.2  # 1.2
        c2 = 0.5
        gamma_new = 0.05
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.001
        gamma22 = 0.05
        gamma32 = 0.007
        alpha = 0.15
        eta = 20

    avr_com_mse = np.zeros((num_compare, repeat_times))
    avr_loc_mse = np.zeros((num_compare, repeat_times))
    maxiter = np.zeros((num_compare, repeat_times))

    # for j in range(num_compare):
    '''j=4
    avr_com_mse[j, 0:16] = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse%d.txt' % j)[0:16]
    avr_loc_mse[j, 0:16] = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse%d.txt' % j)[0:16]'''

    # import data
    for t1 in range(num_a):
        a_stack = np.loadtxt('./Node/Data_Sample/%d'%n + '/data_A_%d' %m + '_%d.txt'%t1)
        for t2 in range(num_x):
            X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_%d.txt' %t2)
            for t3 in range(num_v):
                print(n, m, t1, t2, t3)
                v_stack = np.loadtxt('./Node/Data_Sample/%d'%n + '/data_V_%d' %t2 + '_%d.txt' %t3)
                y_stack = np.zeros(m * n)
                for i in range(n):
                    x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
                    y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
                    SNR = 12
                    s = 10 ** (-1.0 * (SNR / 10.0)) / T * (np.linalg.norm(x_o, ord=2) ** 2)
                    noise_add = np.random.randn(m) * np.sqrt(s)
                    y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add
                # y_stack=y_stack+noise_add
                '''
                print("the Initial BSL:\n")
                # compute the base line
                avr_com_mse[0, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[
                    0, t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.baseline_Cint(a_stack, y_stack, X_est, v_stack, n, gamma_new_bl)
                maxiter[0, t1 * num_x * num_v + t2 * num_v + t3] = 1


                # compute the combination function
                print("VPD-JSM1 Algorithm")
                avr_com_mse[3, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[3,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.ADMM_VPD_JSM1_ind(a_stack, y_stack, X_est, v_stack, n, Adjacent_Matrix, alpha,
                                                      c22, rho2, gamma12, gamma22, gamma32,
                                                      c2, gamma_new, Max_iter, eta, USTp, Umin)

                # compute the l1-soft
                print("the ADMM-VPD:\n")
                avr_com_mse[1, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[1,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.decentral_l1_VR_penalty_c_ind_hard(a_stack, y_stack, X_est, v_stack, n,
                                                                       Adjacent_Matrix, alpha, c2, gamma_new,
                                                                       Max_iter, eta, USTp, Umin)

                # Baseline2 _JSM1
                print("the JSM1 algorithm")
                avr_com_mse[2, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[2,
                                                                         t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.baseline_JSM1(a_stack, y_stack, X_est, v_stack, n, Adjacent_Matrix, c, rho,
                                                  gamma1, gamma2, gamma3, Max_iter)'''

                print("the standard algorithm")
                avr_com_mse[4, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[4,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.stand_DLasso(a_stack, y_stack, X_est, v_stack, n, Adjacent_Matrix, cs, gammas,
                                                 Max_iter)

                # print("a")
                j=4
                # for j in range(num_compare):
                np.savetxt('./Node/convergence/d%d' % n + '_avr_com_mse%d.txt' % j,
                        avr_com_mse[j, :])
                np.savetxt('./Node/convergence/d%d' % n + '_avr_loc_mse%d.txt' % j,
                        avr_loc_mse[j, :])


N=[4, 6, 8, 10, 12, 14, 16]
avr_c0=np.zeros((len(N)))
avr_l0=np.zeros((len(N)))
avr_c1=np.zeros((len(N)))
avr_l1=np.zeros((len(N)))
avr_c2=np.zeros((len(N)))
avr_l2=np.zeros((len(N)))
avr_c3=np.zeros((len(N)))
avr_l3=np.zeros((len(N)))
avr_c4=np.zeros((len(N)))
avr_l4=np.zeros((len(N)))

for j in range(len(N)):
    n=N[j]

    avr_cc0 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse0.txt')

    avr_cc1 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse1.txt')

    avr_cc2 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse2.txt')

    avr_cc3 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse3.txt')

    avr_cc4 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse4.txt')
    avr_ll4 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse4.txt')

    avr_c0[j] = np.average(avr_cc0) #:,
    avr_l0[j] = np.average(avr_ll0)

    avr_c1[j] = np.average(avr_cc1)
    avr_l1[j] = np.average(avr_ll1)

    avr_c2[j] = np.average(avr_cc2)
    avr_l2[j] = np.average(avr_ll2)

    avr_c3[j] = np.average(avr_cc3)
    avr_l3[j] = np.average(avr_ll3)

    avr_c4[j] = np.average(avr_cc4)
    avr_l4[j] = np.average(avr_ll4)
    # cserr3[j, i] = np.average(cserr33[:, Max_iter - 1])
    '''
    avr_c0[j] = np.average(avr_cc0[0])
    avr_l0[j] = np.average(avr_ll0[0])

    avr_c1[j] = np.average(avr_cc1[Max_iter - 1])
    avr_l1[j] = np.average(avr_ll1[Max_iter - 1])

    avr_c2[j] = np.average(avr_cc2[Max_iter - 1])
    avr_l2[j] = np.average(avr_ll2[Max_iter - 1])

    avr_c3[j] = np.average(avr_cc3[Max_iter - 1])
    avr_l3[j] = np.average(avr_ll3[Max_iter - 1])'''


X=[4, 6, 8, 10, 12, 14, 16]
font1={'family':'Times New Roman', 'size':25}
font2={'family':'Times New Roman', 'size':22}
figure1=plt.figure(1)
plt.plot(X, avr_c0, linewidth=2, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
plt.plot(X, avr_c2, linewidth=2, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
plt.plot(X, avr_c1, linewidth=2, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
plt.plot(X, avr_c3, linewidth=2, label="VPD-EC", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
plt.plot(X, avr_c4, linewidth=2, label="Stand", color='gray', linestyle='--', marker='<')
#plt.xlim(min(M), max(M))
#plt.ylim(0, 0.14)
# plt.title("avr_com_mse")
plt.xlabel("N", font1)
plt.ylabel("Glo_AMSE", font1)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font2, bbox_to_anchor=(0.5, 0.85))

figure2=plt.figure(2)
plt.plot(X, avr_l0, linewidth=2, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
plt.plot(X, avr_l2, linewidth=2, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
plt.plot(X, avr_l1, linewidth=2, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
plt.plot(X, avr_l3, linewidth=2, label="VPD-EC", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
plt.plot(X, avr_l4, linewidth=2, label="Stand", color='gray', linestyle='--', marker='<')
# plt.xlim(min(M), max(M))
# plt.ylim(0, 0.1)
# plt.title("avr_loc_mse")
plt.xlabel("N", font1)
plt.ylabel("Loc_AMSE", font1)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font2, bbox_to_anchor=(0.5,0.85))

plt.show()


