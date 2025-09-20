import numpy as np
import random
import ADMM_function
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#20, 30, 40, 50, 60
# 40, 50, 60, 70, 80, 90,
# [30, 50, 70, 90, 110]  #40, 60, 80, 100, 120
    #[20, 30, 40, 50, 60, 70, 80, 90] # the measurement size in each sensor
# M=[20, 40, 60, 80, 100, 120, 140, 160, 180]

M=[60, 70, 80, 90]
T=500
N=6
K=50
d_blocked=[10,  20]
Alpha_l=[0, 0.3, 0.6, 0.9, 0.99]# [round(i*0.05, 2) for i in range(0, 20)]
num_a=2
num_x=2
num_v=4
repeat_times=num_a*num_x*num_v

Adjacent_Matrix=np.matrix([[0, 0, 1, 1, 1, 1]
                    , [0, 0, 1, 1, 1, 0]
                    , [1, 1, 0, 1, 0, 0]
                    , [1, 1, 1, 0, 1, 1]
                    , [1, 1, 0, 1, 0, 1]
                    , [1, 0, 0, 1, 1, 0]])

Max_iter=70
num_compare=len(Alpha_l)

com_mse = np.zeros((num_compare, repeat_times))
loc_mse = np.zeros((num_compare, repeat_times))
Umin=20
for m in M:
    if m==60:
        USTp = 35
        c2 = 1.0
        gamma_new = 0.2
        eta = 10
    elif m==70:
        USTp = 35
        c2 = 1.0
        gamma_new = 0.1
        eta = 20
    elif m==80:
        USTp = 40
        c2 = 1.0
        gamma_new = 0.05
        eta = 20
    else:
        USTp = 40
        c2 = 1.0
        gamma_new = 0.03
        eta = 20

    for d in d_blocked:
        com_mse = np.zeros((num_compare, repeat_times))
        loc_mse = np.zeros((num_compare, repeat_times))
        inddV = np.zeros((4, USTp + 1, repeat_times))
        for t1 in range(num_a):
            # t11=t1+1
            a_stack = np.loadtxt(
                './new_m_compareall_noise/Data_Sample/data_A_%d' % m + '_%d.txt' % t1)  # './new_d_compareall_noise/Data_Sample/
            for t2 in range(num_x):
                X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_%d.txt' % t2)
                for t3 in range(num_v):
                    print(m, d, t1, t2, t3)
                    v_stack = np.loadtxt(
                        './new_d_compareall_noise/Data_Sample/d%d' % d + '/data_V_%d' % t2 + '_%d.txt' % t3)
                    y_stack = np.zeros(m * N)
                    for i in range(N):
                        x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
                        y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
                        SNR = 10
                        s = 10 ** (-1.0 * (SNR / 10.0)) / T * (np.linalg.norm(x_o, ord=2) ** 2)
                        noise_add = np.random.randn(m) * np.sqrt(s)
                        y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add

                    for i in range(len(Alpha_l)):
                        # eta=10 for snr=10, m=60/80; eta=15 for snr=15, m=60
                        alpha = Alpha_l[i]
                        print("SNR10, the ADMM-VPD: %.2f \n" % alpha)
                        avr_com_mse, avr_loc_mse \
                            = ADMM_function.observe_v(a_stack, y_stack, X_est, v_stack, N,
                                                                   Adjacent_Matrix, alpha, c2, gamma_new, Max_iter, eta, USTp, Umin)
                        com_mse[i, t1 * num_x * num_v + t2 * num_v + t3] = avr_com_mse
                        loc_mse[i, t1 * num_x * num_v + t2 * num_v + t3] = avr_loc_mse

                        np.savetxt('./new_compare_V/SNR10/d%d' % d + '/%d' % m + '_avr_com_mse0.txt', com_mse[:, :])
                        np.savetxt('./new_compare_V/SNR10/d%d' % d + '/%d' % m + '_avr_loc_mse0.txt', loc_mse[:, :])
                        '''
                        if i == 0:
                            inddV[0, :, t1 * num_x * num_v + t2 * num_v + t3] = indV
                            np.savetxt('./new_compare_V/SNR10/d%d' % d + '/%d' % m + '_inddV%d.txt' % i, inddV[0, :, :])
                        elif i == 1:
                            inddV[1, :, t1 * num_x * num_v + t2 * num_v + t3] = indV
                            np.savetxt('./new_compare_V/SNR10/d%d' % d + '/%d' % m + '_inddV%d.txt' % i, inddV[1, :, :])
                        elif i == 2:
                            inddV[2, :, t1 * num_x * num_v + t2 * num_v + t3] = indV
                            np.savetxt('./new_compare_V/SNR10/d%d' % d + '/%d' % m + '_inddV%d.txt' % i, inddV[2, :, :])
                        elif i == 3:
                            inddV[3, :, t1 * num_x * num_v + t2 * num_v + t3] = indV
                            np.savetxt('./new_compare_V/SNR10/d%d' % d + '/%d' % m + '_inddV%d.txt' % i, inddV[3, :, :])'''


USTp=35  #mu=0.08 m=60 snr=10; mu=0.06, m=80, snr=10; mu=0.06, m=60, snr=15
ac = np.zeros((len(M), len(d_blocked), len(Alpha_l)))
al = np.zeros((len(M), len(d_blocked), len(Alpha_l)))


for i in range(len(M)):
    for j in range(len(d_blocked)):
        m=M[i]
        d=d_blocked[j]
        avr_c01 = np.loadtxt('./new_compare_V/SNR10/d%d'%d +'/%d'%m+'_avr_com_mse0.txt')
        avr_l01 = np.loadtxt('./new_compare_V/SNR10/d%d'%d +'/%d'%m+'_avr_loc_mse0.txt')
        ac[i, j, :] = np.average(avr_c01, axis=1)
        al[i, j, :] = np.average(avr_l01, axis=1)

d=10
m=60


print("p=0.2")
for i in range(len(M)):
    print(ac[i, 0, :]*1000)
    print(al[i, 0, :]*1000)