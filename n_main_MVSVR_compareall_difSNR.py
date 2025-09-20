import numpy as np
import random
import ADMM_function
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

m = 100
T = 500
N = 6
K = 50
d_blocked = 10  # 5, 15, 25,   15和25的20,40，60还没跑
# 先仿真16个看
num_a = 2  # 2,2,4
num_x = 2
num_v = 4
SNR_set1 = [0, 5, 10, 15, 20, 25]  # 0
repeat_times = num_a * num_x * num_v

Adjacent_Matrix = np.matrix([[0, 0, 1, 1, 1, 1]
                                , [0, 0, 1, 1, 1, 0]
                                , [1, 1, 0, 1, 0, 0]
                                , [1, 1, 1, 0, 1, 1]
                                , [1, 1, 0, 1, 0, 1]
                                , [1, 0, 0, 1, 1, 0]])

Max_iter = 100
num_compare = 5
USTp=35
Umin=20

cs = 0.01
gammas = 0.03

for SNR in SNR_set1:
    if SNR==-5:
        c = 0.5
        gamma1 = 0.2
        gamma2 = 1.2
        gamma3 = 0.05
        rho = 0.5

        gamma_bl = 1.0
        c2 = 2.0
        gamma_new = 0.5
        c22 = 1.0
        rho2 = 1.0
        gamma12 = 0.03
        gamma22 = 0.3
        gamma32 = 0.02
        alpha = 0.1
        eta = 10

    elif SNR==0:
        c = 0.3
        gamma1 = 0.09
        gamma2 = 0.25
        gamma3 = 0.12
        rho = 0.3

        gamma_bl = 0.8
        c2 = 1.0
        gamma_new = 0.5
        c22 = 1.0
        rho2 = 1.0
        alpha = 0.2
        eta = 10
        gamma12 = 0.03
        gamma22 = 0.3
        gamma32 = 0.02

    elif SNR==5:
        c = 0.2
        gamma1 = 0.01
        gamma2 = 0.12
        gamma3 = 0.004
        rho = 0.2

        gamma_bl = 0.5
        c2 = 1.0
        gamma_new = 0.1
        c22 = 1.0
        rho2 = 1.0
        gamma12 = 0.07
        gamma22 = 0.5
        gamma32 = 0.1
        alpha = 0.1
        eta = 10.0

    elif SNR==10:
        c = 0.1
        gamma1 = 0.008
        gamma2 = 0.1
        gamma3 = 0.003
        rho = 0.1

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 1.0
        rho2 = 1.0
        gamma12 = 0.05
        gamma22 = 0.3
        gamma32 = 0.07
        alpha = 0.2
        eta = 15

    elif SNR==15:
        c = 0.1
        gamma1 = 0.006
        gamma2 = 0.1
        gamma3 = 0.002
        rho = 0.1

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 1.0
        rho2 = 1.0
        gamma12 = 0.03
        gamma22 = 0.1
        gamma32 = 0.05
        alpha = 0.2
        eta = 15

    elif SNR>=20:
        c = 0.1
        gamma1 = 0.006
        gamma2 = 0.1
        gamma3 = 0.002
        rho = 0.1

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.02
        c22 = 1.0
        rho2 = 1.0
        gamma12 = 0.02
        gamma22 = 0.1
        gamma32 = 0.04
        alpha = 0.2
        eta = 15

    avr_com_mse = np.zeros((num_compare, repeat_times))
    avr_loc_mse = np.zeros((num_compare, repeat_times))
    maxiter = np.zeros((num_compare, repeat_times))
    '''
    a1 = np.loadtxt('./new_compare_SNR/convergence/SNR%d' % SNR + '_avr_com_mse2.txt')
    a2 = np.loadtxt('./new_compare_SNR/convergence/SNR%d' % SNR + '_avr_loc_mse2.txt')
    b1 = np.loadtxt('./new_compare_SNR/convergence/SNR%d' % SNR + '_avr_com_mse1.txt')
    b2 = np.loadtxt('./new_compare_SNR/convergence/SNR%d' % SNR + '_avr_loc_mse1.txt')
    
    
    for j in range(num_compare):
        avr_com_mse[j, 0:16] = np.loadtxt(
            './new_compare_SNR/convergence/SNR%d' %SNR+ '_avr_com_mse%d.txt' % j)[0:16]
        avr_loc_mse[j, 0:16] = np.loadtxt(
            './new_compare_SNR/convergence/SNR%d' %SNR+ '_avr_loc_mse%d.txt' % j)[0:16]'''

    # import data
    for t1 in range(num_a):
        a_stack = np.loadtxt('./new_compare_SNR/Data_Sample/data_A_%d' %m +'_%d.txt'% t1) #'./new_d_compareall_noise/Data_Sample/
        for t2 in range(num_x):
            X_est = np.loadtxt('./new_compare_SNR/Data_Sample/data_X_%d.txt' % t2)
            for t3 in range(num_v):
                # print(SNR, t1, t2, t3)
                v_stack = np.loadtxt('./new_compare_SNR/Data_Sample/d%d'%d_blocked+'/data_V_%d'%t2 + '_%d.txt' % t3)
                y_stack = np.zeros(m * N)
                for i in range(N):
                    x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
                    y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
                    s=10**(-1.0*SNR/10)* (np.linalg.norm(x_o, ord=2)**2)/T
                    noise_add = np.random.randn(m)*np.sqrt(s)
                    y_stack[i * m: (i + 1) * m]=y_stack[i * m: (i + 1) * m]+noise_add
                    #y_stack=y_stack+noise_add

                # Baseline2 _JSM1
                # print("the JSM1 algorithm: %.4f," %a1[t1 * num_x * num_v + t2 * num_v + t3]+" %.4f"%a2[t1 * num_x * num_v + t2 * num_v + t3])
                #  print("the VPD algorithm: %.4f," % b1[t1 * num_x * num_v + t2 * num_v + t3] + " %.4f" % b2[
                #     t1 * num_x * num_v + t2 * num_v + t3])
                '''
                print("the Initial BSL:\n")
                # compute the base line
                avr_com_mse[0, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[
                    0, t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.baseline_Cint(a_stack, y_stack, X_est, v_stack, N, gamma_bl)

                

                # compute the l1-soft
                print("the ADMM-VPD:\n")
                avr_com_mse[1, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[1,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.decentral_l1_VR_penalty_c_ind_hard(a_stack, y_stack, X_est, v_stack, N,
                                                                       Adjacent_Matrix, alpha, c2, gamma_new, Max_iter,
                                                                       eta, USTp, Umin)

                print("JSM1")
                avr_com_mse[2, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[2,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.baseline_JSM1(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, c, rho,
                                                  gamma1, gamma2, gamma3, Max_iter)

                # compute the combination function
                print("VPD-JSM1 Algorithm")
                avr_com_mse[3, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[3,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.ADMM_VPD_JSM1_ind(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, alpha,
                                                      c22, rho2, gamma12, gamma22, gamma32,
                                                      c2, gamma_new, Max_iter, eta, USTp, Umin)

                print("the standard algorithm")
                avr_com_mse[4, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[4,
                                                                         t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.stand_DLasso(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, cs, gammas,
                                                 Max_iter)

                # for j in range(num_compare):
                j=4
                np.savetxt('./new_compare_SNR/convergence2/SNR%d' %SNR+ '_avr_com_mse%d.txt' % j,
                                   avr_com_mse[j, :])
                np.savetxt('./new_compare_SNR/convergence2/SNR%d' %SNR+ '_avr_loc_mse%d.txt' % j,
                                   avr_loc_mse[j, :])'''


SNR_set=[0, 5, 10, 15, 20, 25]
avr_c0 = np.zeros((len(SNR_set)))
avr_l0 = np.zeros((len(SNR_set)))
avr_c1 = np.zeros((len(SNR_set)))
avr_l1 = np.zeros((len(SNR_set)))
avr_c2 = np.zeros((len(SNR_set)))
avr_l2 = np.zeros((len(SNR_set)))
avr_c3 = np.zeros((len(SNR_set)))
avr_l3 = np.zeros((len(SNR_set)))
avr_c4 = np.zeros((len(SNR_set)))
avr_l4 = np.zeros((len(SNR_set)))

for j in range(len(SNR_set)):
    SNR = SNR_set[j]
    avr_cc0 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse0.txt')

    avr_cc1 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse1.txt')

    avr_cc2 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse2.txt')

    avr_cc3 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse3.txt')

    avr_cc4 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse4.txt')
    avr_ll4 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse4.txt')

    avr_c0[j] = np.average(avr_cc0)  #:,
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
    avr_l3[j] = np.average(avr_ll3[Max_iter - 1])
    '''

X = SNR_set
font1 = {'family': 'Times New Roman', 'size': 30}
font2 = {'family': 'Times New Roman', 'size': 25}
figure1=plt.figure()
ax1 = plt.subplot(111)
plt.plot(X, avr_c0, linewidth=2, label="IRAS", color='b', linestyle='-', marker='o', markersize=10)
plt.plot(X, avr_c2, linewidth=2, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=10)
plt.plot(X, avr_c1, linewidth=2, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=10)
plt.plot(X, avr_c3, linewidth=2, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=10)
plt.plot(X, avr_c4, linewidth=2, label="Stand", color='gray', linestyle='--', marker='<', markersize=10)
# plt.xlim(min(M), max(M))
# plt.ylim(0, 0.14)
# plt.title("avr_com_mse")
yLocator=MultipleLocator(0.02)
ax1.yaxis.set_major_locator(yLocator)
plt.xlabel("SNR(dB)", font1)
plt.ylabel("Glo_AMSE", font1)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font2, loc=1)

figure2=plt.figure()
ax2 = plt.subplot(111)
plt.plot(X, avr_l0, linewidth=2, label="IRAS", color='b', linestyle='-', marker='o', markersize=10)
plt.plot(X, avr_l2, linewidth=2, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=10)
plt.plot(X, avr_l1, linewidth=2, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=10)
plt.plot(X, avr_l3, linewidth=2, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=10)
plt.plot(X, avr_l4, linewidth=2, label="Stand", color='gray', linestyle='--', marker='<', markersize=10)
# plt.xlim(min(M), max(M))
# plt.ylim(0, 0.1)
# plt.title("avr_loc_mse")
yLocator=MultipleLocator(0.01)
ax2.yaxis.set_major_locator(yLocator)
plt.xlabel("SNR(dB)", font1)
plt.ylabel("Loc_AMSE", font1)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font2, loc=1)

plt.show()

