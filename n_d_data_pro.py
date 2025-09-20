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
m=100
T=500
N=6
K=50
d_blocked=[20] #0, 5, 10, 15, 20, 25  # 5和10， 15, 20, 25都跑过了
# 先仿真16个看
num_a=5   #2,2,4;  5,2,4
num_x=2
num_v=4
repeat_times=num_a*num_x*num_v

Adjacent_Matrix=np.matrix([[0, 0, 1, 1, 1, 1]
                    , [0, 0, 1, 1, 1, 0]
                    , [1, 1, 0, 1, 0, 0]
                    , [1, 1, 1, 0, 1, 1]
                    , [1, 1, 0, 1, 0, 1]
                    , [1, 0, 0, 1, 1, 0]])

Max_iter=100
num_compare=4

# fix the parameter of JSM1
'''
c=0.05
gamma1=0.01
gamma2=0.5
gamma3=0.02
rho=0.05
'''

for d in d_blocked:
    avr_com_mse1 = np.zeros((num_compare, 16, Max_iter))
    avr_loc_mse1 = np.zeros((num_compare, 16, Max_iter))
    avr_com_mse2 = np.zeros((num_compare, repeat_times, Max_iter))
    avr_loc_mse2 = np.zeros((num_compare, repeat_times, Max_iter))


    for j in range(num_compare):
        avr_com_mse1[j, :, :] = np.loadtxt(
            './new_d_compareall_noise/convergence/d%d/' % d + 'avr_com_mse%d.txt' % j)
        avr_loc_mse1[j, :, :] = np.loadtxt(
            './new_d_compareall_noise/convergence/d%d/' % d + 'avr_loc_mse%d.txt' % j)
        avr_com_mse2[j, :, :] = np.loadtxt(
            './new_d_compareall_noise/convergence2/d%d/' % d + 'avr_com_mse%d.txt' % j)
        avr_loc_mse2[j, :, :] = np.loadtxt(
            './new_d_compareall_noise/convergence2/d%d/' % d + 'avr_loc_mse%d.txt' % j)

    for j in range(num_compare):
        avr_com_mse2[j, 0:16 :]= avr_com_mse1[j, :, :]
        avr_loc_mse2[j, 0:16, :] = avr_loc_mse1[j, :, :]

    for j in range(num_compare):
        np.savetxt('./new_d_compareall_noise/convergence2/d%d' % d + '/avr_com_mse%d.txt' % j,
                   avr_com_mse2[j, :, :])
        np.savetxt('./new_d_compareall_noise/convergence2/d%d' % d + '/avr_loc_mse%d.txt' % j,
                   avr_loc_mse2[j, :, :])

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
    avr_cc0 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_loc_mse0.txt')

    avr_cc1 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_loc_mse1.txt')

    avr_cc2 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_loc_mse2.txt')

    avr_cc3 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/avr_loc_mse3.txt')

    avr_c0[j] = np.average(avr_cc0[:, 0]) #:,
    avr_l0[j] = np.average(avr_ll0[:, 0])

    avr_c1[j] = np.average(avr_cc1[:,Max_iter - 1])
    avr_l1[j] = np.average(avr_ll1[:,Max_iter - 1])

    avr_c2[j] = np.average(avr_cc2[:, Max_iter - 1])
    avr_l2[j] = np.average(avr_ll2[:, Max_iter - 1])

    avr_c3[j] = np.average(avr_cc3[:, Max_iter - 1])
    avr_l3[j] = np.average(avr_ll3[:, Max_iter - 1])
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


X=d_blocked
figure1=plt.figure(1)
plt.plot(X, avr_c0, linewidth=2, label="BL(m=%d)"%m, color='b', linestyle='-', marker='o')
plt.plot(X, avr_c2, linewidth=2, label="MJSM-1A(m=%d)"%m, color='r', linestyle=':', marker='>')
plt.plot(X, avr_c1, linewidth=2, label="ADMM-VPD(m=%d)"%m, color='g', linestyle='-.', marker='*')
plt.plot(X, avr_c3, linewidth=2, label="VPD-CD(m=%d)"%m, color='y', linestyle='--', marker='s')
#plt.xlim(min(M), max(M))
#plt.ylim(0, 0.14)
# plt.title("avr_com_mse")
plt.xlabel("d")
plt.ylabel("Glo_AMSE")
plt.grid(True)
plt.legend()

figure2=plt.figure(2)
plt.plot(X, avr_l0, linewidth=2, label="BL(m=%d)"%m, color='b', linestyle='-', marker='o')
plt.plot(X, avr_l2, linewidth=2, label="MJSM-1A(m=%d)"%m, color='r', linestyle=':', marker='>')
plt.plot(X, avr_l1, linewidth=2, label="ADMM-VPD(m=%d)"%m, color='g', linestyle='-.', marker='*')
plt.plot(X, avr_l3, linewidth=2, label="VPD-CD(m=%d)"%m, color='y', linestyle='--', marker='s')
# plt.xlim(min(M), max(M))
# plt.ylim(0, 0.1)
# plt.title("avr_loc_mse")
plt.xlabel("d")
plt.ylabel("Loc_AMSE")
plt.grid(True)
plt.legend()

plt.show()
