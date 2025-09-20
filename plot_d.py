import numpy as np
import random
import ADMM_function
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
num_compare=4
d_blocked=[0, 5, 10, 15, 20, 25, 30, 35, 40]

avr_c0=np.zeros((len(d_blocked)))
avr_l0=np.zeros((len(d_blocked)))
avr_c1=np.zeros((len(d_blocked)))
avr_l1=np.zeros((len(d_blocked)))
avr_c2=np.zeros((len(d_blocked)))
avr_l2=np.zeros((len(d_blocked)))
avr_c3=np.zeros((len(d_blocked)))
avr_l3=np.zeros((len(d_blocked)))

for j in range(len(d_blocked)):
    d=d_blocked[j]

    avr_cc0 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse0.txt')

    avr_cc1 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse1.txt')

    avr_cc2 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse2.txt')

    avr_cc3 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse3.txt')


    avr_c0[j] = np.average(avr_cc0) #:,
    avr_l0[j] = np.average(avr_ll0)

    avr_c1[j] = np.average(avr_cc1)
    avr_l1[j] = np.average(avr_ll1)

    avr_c2[j] = np.average(avr_cc2)
    avr_l2[j] = np.average(avr_ll2)

    avr_c3[j] = np.average(avr_cc3)
    avr_l3[j] = np.average(avr_ll3)
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


X=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
font1={'family':'Times New Roman', 'size':30}
font2={'family':'Times New Roman', 'size':30}
figure1=plt.figure(1)
plt.plot(X, avr_c0, linewidth=2, label="IRAS", color='b', linestyle='-', marker='o', markersize=10)
plt.plot(X, avr_c2, linewidth=2, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=10)
plt.plot(X, avr_c1, linewidth=2, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=10)
plt.plot(X, avr_c3, linewidth=2, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=10)
#plt.xlim(min(M), max(M))
#plt.ylim(0, 0.14)
# plt.title("avr_com_mse")
plt.xlabel("p", font1)
plt.ylabel("Glo_AMSE", font1)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font2)

figure2=plt.figure(2)
plt.plot(X, avr_l0, linewidth=2, label="IRAS", color='b', linestyle='-', marker='o', markersize=10)
plt.plot(X, avr_l2, linewidth=2, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=10)
plt.plot(X, avr_l1, linewidth=2, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=10)
plt.plot(X, avr_l3, linewidth=2, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=10)
# plt.xlim(min(M), max(M))
# plt.ylim(0, 0.1)
# plt.title("avr_loc_mse")
plt.xlabel("p", font1)
plt.ylabel("Loc_AMSE", font1)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font2)

plt.show()

