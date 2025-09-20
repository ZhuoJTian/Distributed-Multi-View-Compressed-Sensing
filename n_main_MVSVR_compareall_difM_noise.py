import numpy as np
import random
import ADMM_function
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


M = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
# 60.70 for Standard is
T = 500
N = 6
K = 50
d_blocked = [10]  # 5, 15, 25,   15和25的20,40，60还没跑
num_a = 2  # 2,2,4; 5,2,4
num_x = 2
num_v = 4
repeat_times = num_a * num_x * num_v

Adjacent_Matrix = np.matrix([[0, 0, 1, 1, 1, 1]
                                , [0, 0, 1, 1, 1, 0]
                                , [1, 1, 0, 1, 0, 0]
                                , [1, 1, 1, 0, 1, 1]
                                , [1, 1, 0, 1, 0, 1]
                                , [1, 0, 0, 1, 1, 0]])

Max_iter = 100
num_compare = 5

# fix the parameter of JSM1
c = 0.05
gamma1 = 0.007
gamma2 = 0.2
gamma3 = 0.01
rho = 0.05
USTp=35
Umin=20

cs = 1.0
gammas = 0.1

for d in d_blocked:
    for m in M:
        if m == 60:
            gamma_new_bl = 0.4
            c2 = 1.0
            gamma_new = 0.2
            c22 = 0.5
            rho2 = 0.5
            gamma12 = 0.009
            gamma22 = 0.1
            gamma32 = 0.007
            alpha = 0.5
            eta = 20
        elif m == 70:
            gamma_new_bl = 0.4
            c2 = 1.0
            gamma_new = 0.08
            c22 = 0.5
            rho2 = 0.5
            gamma12 = 0.009
            gamma22 = 0.1
            gamma32 = 0.007
            alpha = 0.3
            eta = 20
        elif m == 80:
            gamma_new_bl = 0.3
            c2 = 1.0
            gamma_new = 0.05
            c22 = 0.5
            rho2 = 0.5
            gamma12 = 0.007
            gamma22 = 0.1
            gamma32 = 0.005
            alpha = 0.2
            eta = 20
        elif m <= 100:
            gamma_new_bl = 0.3
            c2 = 1.0
            gamma_new = 0.03
            c22 = 0.5
            rho2 = 0.5
            gamma12 = 0.005
            gamma22 = 0.08
            gamma32 = 0.003
            alpha = 0.1
            mu = 0.2
            eta = 20
        elif m == 110:
            gamma_new_bl = 0.3
            c2 = 1.0
            gamma_new = 0.02
            c22 = 0.5
            rho2 = 0.5
            gamma12 = 0.005
            gamma22 = 0.08
            gamma32 = 0.003
            alpha = 0.1
            mu = 0.2
            eta = 20
        elif m <= 120:
            gamma_new_bl = 0.3
            c2 = 1.0
            gamma_new = 0.01
            c22 = 0.5
            rho2 = 0.5
            gamma12 = 0.005
            gamma22 = 0.06
            gamma32 = 0.003
            alpha = 0.1
            eta = 15
        else:
            gamma_new_bl = 0.3
            c2 = 1.0
            gamma_new = 0.01
            c22 = 0.5
            rho2 = 0.5
            gamma12 = 0.005
            gamma22 = 0.06
            gamma32 = 0.003
            alpha = 0.1
            eta = 10

        avr_com_mse = np.zeros((num_compare, repeat_times, Max_iter))
        avr_loc_mse = np.zeros((num_compare, repeat_times, Max_iter))
        maxiter = np.zeros((num_compare, repeat_times))

        '''
        avr_com_mse[4, 0:16, :] = np.loadtxt(
            './new_m_compareall_noise/convergence/d%d/' % d + '/%d_' % m + 'avr_com_mse4.txt')[0:16, :]
        avr_loc_mse[4, 0:16, :] = np.loadtxt(
            './new_m_compareall_noise/convergence/d%d/' % d + '/%d_' % m + 'avr_loc_mse4.txt')[0:16, :]

        
        for j in range(num_compare):
            avr_com_mse[j, 0:16, :] = np.loadtxt(
                './new_m_compareall_noise/convergence3/d%d/' % d + '/%d_' % m + 'avr_com_mse%d.txt' % j)[0:16, :]
            avr_loc_mse[j, 0:16, :] = np.loadtxt(
                './new_m_compareall_noise/convergence3/d%d/' % d + '/%d_' % m + 'avr_loc_mse%d.txt' % j)[0:16, :]'''


        # import data
        for t1 in range(num_a):
            a_stack = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_A_%d' % m + '_%d.txt' % t1)
            for t2 in range(num_x):
                X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_%d.txt' % t2)
                for t3 in range(num_v):
                    # print(d, m, t1, t2, t3)
                    v_stack = np.loadtxt(
                        './new_m_compareall_noise/Data_Sample/d%d' % d + '/data_V_%d' % t2 + '_%d.txt' % t3)
                    y_stack = np.zeros(m * N)
                    for i in range(N):
                        x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
                        y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
                        SNR = 12
                        s = 10 ** (-1.0 * (SNR / 10.0)) / T * (np.linalg.norm(x_o, ord=2) ** 2)
                        # s = 0.01 / m * (np.linalg.norm(y_stack[i * m: (i + 1) * m], ord=2) ** 2)
                        noise_add = np.random.randn(m) * np.sqrt(s)
                        y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add
                    # y_stack=y_stack+noise_add

                    '''
                    # compute the combination function
                    print("VPD-JSM1 Algorithm")
                    avr_com_mse[3, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[3,
                                                                                      t1 * num_x * num_v + t2 * num_v + t3] \
                        = ADMM_function.ADMM_VPD_JSM1_ind(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, alpha,
                                                          c22, rho2, gamma12, gamma22, gamma32,
                                                          c2, gamma_new, Max_iter, eta, USTp, Umin)

                    # compute the l1-soft
                    print("the ADMM-VPD:\n")
                    avr_com_mse[1, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[1,
                                                                                      t1 * num_x * num_v + t2 * num_v + t3] \
                        = ADMM_function.decentral_l1_VR_penalty_c_ind_hard(a_stack, y_stack, X_est, v_stack, N,
                                                                           Adjacent_Matrix, alpha, c2, gamma_new,
                                                                           Max_iter, eta, USTp, Umin)

                    print("the Initial BSL:\n")
                    # compute the base line
                    avr_com_mse[0, t1 * num_x * num_v + t2 * num_v + t3, 0], avr_loc_mse[
                        0, t1 * num_x * num_v + t2 * num_v + t3, 0]\
                        = ADMM_function.baseline_Cint(a_stack, y_stack, X_est, v_stack, N, gamma_new_bl)
                    maxiter[0, t1 * num_x * num_v + t2 * num_v + t3] = 1

                    # Baseline2 _JSM1
                    print("the JSM1 algorithm")
                    avr_com_mse[2, t1 * num_x * num_v + t2 * num_v + t3, :], avr_loc_mse[2,
                                                                             t1 * num_x * num_v + t2 * num_v + t3, :]\
                        = ADMM_function.baseline_JSM1(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, c, rho,
                                                      gamma1, gamma2, gamma3, Max_iter)
                    '''
                    print("the standard algorithm")
                    avr_com_mse[4, t1 * num_x * num_v + t2 * num_v + t3, :], avr_loc_mse[4,
                                                                             t1 * num_x * num_v + t2 * num_v + t3, :] \
                        = ADMM_function.stand_DLasso(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, cs, gammas, Max_iter)
                    # for j in range(num_compare):
                    j=4
                    np.savetxt(
                            './new_m_compareall_noise/convergence2/d%d' % d + '/%d_' % m + 'avr_com_mse%d.txt' % j,
                            avr_com_mse[j, :, :])
                    np.savetxt(
                            './new_m_compareall_noise/convergence2/d%d' % d + '/%d_' % m + 'avr_loc_mse%d.txt' % j,
                            avr_loc_mse[j, :, :])


M = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
avr_c0 = np.zeros((len(d_blocked), len(M)))
avr_l0 = np.zeros((len(d_blocked), len(M)))
avr_c1 = np.zeros((len(d_blocked), len(M)))
avr_l1 = np.zeros((len(d_blocked), len(M)))
avr_c2 = np.zeros((len(d_blocked), len(M)))
avr_l2 = np.zeros((len(d_blocked), len(M)))
avr_c3 = np.zeros((len(d_blocked), len(M)))
avr_l3 = np.zeros((len(d_blocked), len(M)))
avr_c4 = np.zeros((len(d_blocked), len(M)))
avr_l4 = np.zeros((len(d_blocked), len(M)))

cserr0 = np.zeros((len(d_blocked), len(M)))
cserr1 = np.zeros((len(d_blocked), len(M)))
cserr2 = np.zeros((len(d_blocked), len(M)))
cserr3 = np.zeros((len(d_blocked), len(M)))
cserr4 = np.zeros((len(d_blocked), len(M)))

for j in range(len(d_blocked)):
    for i in range(len(M)):
        m = M[i]
        d = d_blocked[j]
        avr_cc0 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse0.txt')
        avr_ll0 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse0.txt')
        # cserr00 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr0.txt')

        avr_cc1 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse1.txt')
        avr_ll1 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse1.txt')
        # cserr11 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr1.txt')

        avr_cc2 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse2.txt')
        avr_ll2 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse2.txt')
        # cserr22 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr2.txt')

        avr_cc3 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse3.txt')
        avr_ll3 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse3.txt')
        # cserr33 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr3.txt')

        avr_cc4 = np.loadtxt('./new_m_compareall_noise/convergence2/d%d' % d + '/%d_' % m + 'avr_com_mse4.txt')
        avr_ll4 = np.loadtxt('./new_m_compareall_noise/convergence2/d%d' % d + '/%d_' % m + 'avr_loc_mse4.txt')

        avr_c0[j, i] = np.average(avr_cc0[:, 0])  #:,
        avr_l0[j, i] = np.average(avr_ll0[:, 0])
        # cserr0[j, i] = np.average(cserr00[:, 0])

        avr_c1[j, i] = np.average(avr_cc1[:, Max_iter - 1])
        avr_l1[j, i] = np.average(avr_ll1[:, Max_iter - 1])
        # cserr1[j, i] = np.average(cserr11[:,Max_iter - 1])

        avr_c2[j, i] = np.average(avr_cc2[:, Max_iter - 1])
        avr_l2[j, i] = np.average(avr_ll2[:, Max_iter - 1])
        # cserr2[j, i] = np.average(cserr22[:, Max_iter - 1])

        avr_c3[j, i] = np.average(avr_cc3[:, Max_iter - 1])
        avr_l3[j, i] = np.average(avr_ll3[:, Max_iter - 1])
        # cserr3[j, i] = np.average(cserr33[:, Max_iter - 1])

        avr_c4[j, i] = np.average(avr_cc4[:, Max_iter - 1])
        avr_l4[j, i] = np.average(avr_ll4[:, Max_iter - 1])

        '''
        avr_c0[j, i] = np.average(avr_cc0[0])  #:,
        avr_l0[j, i] = np.average(avr_ll0[0])
        cserr0[j, i] = np.average(cserr00[0])

        avr_c1[j, i] = np.average(avr_cc1[Max_iter - 1])
        avr_l1[j, i] = np.average(avr_ll1[Max_iter - 1])
        cserr1[j, i] = np.average(cserr11[Max_iter - 1])

        avr_c2[j, i] = np.average(avr_cc2[Max_iter - 1])
        avr_l2[j, i] = np.average(avr_ll2[Max_iter - 1])
        cserr2[j, i] = np.average(cserr22[Max_iter - 1])

        avr_c3[j, i] = np.average(avr_cc3[Max_iter - 1])
        avr_l3[j, i] = np.average(avr_ll3[Max_iter - 1])
        cserr3[j, i] = np.average(cserr33[Max_iter - 1])
        '''

X = M
figure1 = plt.figure(1)
plt.plot(X, avr_c0[0, :], linewidth=2, label="IRAS(p=0.2)", color='b', linestyle='-', marker='o')
plt.plot(X, avr_c2[0, :], linewidth=2, label="D-AJSM(p=0.2)", color='r', linestyle=':', marker='>')
plt.plot(X, avr_c1[0, :], linewidth=2, label="VPD-ADMM(p=0.2)", color='g', linestyle='-.', marker='*')
plt.plot(X, avr_c3[0, :], linewidth=2, label="VPD-EC(p=0.2)", color='y', linestyle='--', marker='s')
plt.plot(X, avr_c4[0, :], linewidth=2, label="Stand(p=0.2)", color='gray', linestyle='--', marker='<')
plt.xlim(min(M), max(M))
plt.ylim(0, 0.12)
# plt.title("avr_com_mse")
plt.xlabel("M")
plt.ylabel("avr_Glo_mse")
plt.grid(True)
plt.legend()

figure2 = plt.figure(2)
plt.plot(X, avr_l0[0, :], linewidth=2, label="IRAS(p=0.2)", color='b', linestyle='-', marker='o')
plt.plot(X, avr_l2[0, :], linewidth=2, label="D-AJSM(p=0.2)", color='r', linestyle=':', marker='>')
plt.plot(X, avr_l1[0, :], linewidth=2, label="VPD-ADMM(p=0.2)", color='g', linestyle='-.', marker='*')
plt.plot(X, avr_l3[0, :], linewidth=2, label="VPD-EC(p=0.2)", color='y', linestyle='--', marker='s')
plt.plot(X, avr_l4[0, :], linewidth=2, label="Stand(p=0.2)", color='gray', linestyle='--', marker='<')
plt.xlim(min(M), max(M))
plt.ylim(0, 0.08)
# plt.title("avr_loc_mse")
plt.xlabel("M")
plt.ylabel("avr_Loc_mse")
plt.grid(True)
plt.legend()
'''
figure3=plt.figure(3)
plt.plot(X, cserr0[0, :], linewidth=2, label="BL(d=%d)"%d_blocked[0], color='b', linestyle='-', marker='o')
# plt.plot(X, cserr0[1, :], linewidth=2, label="BL(d=%d)"%d_blocked[1], color='r', linestyle='-', marker='>')
# plt.plot(X, cserr0[2, :], linewidth=2, label="BL(d=%d)"%d_blocked[2], color='g', linestyle='-', marker='o')

plt.plot(X, cserr1[0, :], linewidth=2, label="ADMMVPD(d=%d)"%d_blocked[0], color='g', linestyle='-.', marker='*')
# plt.plot(X, cserr1[1, :], linewidth=2, label="ADMMVPD(d=%d)"%d_blocked[1], color='r', linestyle='-.', marker='>')
# plt.plot(X, cserr1[2, :], linewidth=2, label="ADMMVPD(d=%d)"%d_blocked[2], color='g', linestyle='-.', marker='o')
# plt.plot(X, ratio_c1[3, :], linewidth=2, label="ADMMVPD(d=20)", color='c', linestyle='-.', marker='+')
# plt.plot(X, ratio_c1[4, :], linewidth=2, label="ADMMVPD(d=25)", color='m', linestyle='-.', marker='s')
# plt.plot(X, avr_loc_mse_s[3, :], linewidth=2, label="ADMMOGD(beta=1.6)", color='g', linestyle=':')
# plt.yscale('log')

plt.plot(X, cserr2[0, :], linewidth=2, label="JSM1(d=%d)"%d_blocked[0], color='r', linestyle='-.', marker='>')

plt.plot(X, cserr3[0, :], linewidth=2, label="VPD-JSM1(d=%d)"%d_blocked[0], color='y', linestyle='-.', marker='+')

plt.title("Consenus Error")
plt.xlabel("M")
plt.ylabel("cserr")
plt.grid(True)
plt.legend()
'''
plt.show()

