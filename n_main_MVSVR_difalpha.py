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

m=80
T=500
N=6
K=50
d_blocked=10
Alpha_l=[0.2, 0.8]
num_a=1
num_x=1
num_v=1
repeat_times=num_a*num_x*num_v

Adjacent_Matrix=np.matrix([[0, 0, 1, 1, 1, 1]
                    , [0, 0, 1, 1, 1, 0]
                    , [1, 1, 0, 1, 0, 0]
                    , [1, 1, 1, 0, 1, 1]
                    , [1, 1, 0, 1, 0, 1]
                    , [1, 0, 0, 1, 1, 0]])

Max_iter=100
num_compare=3

c = 0.05
gamma1 = 0.007
gamma2 = 0.2
gamma3 = 0.015
rho = 0.05

c2 = 1.0
gamma_new = 0.1
c22 = 0.5
rho2 = 0.5
gamma12 = 0.009
gamma22 = 0.1
gamma32 = 0.007
eta = 15
mu=0.2
USTp=35
Umin=20
'''
a_stack = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_A_80_1.txt')  # './new_d_compareall_noise/Data_Sample/
X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_2.txt')
v_stack = np.loadtxt('./new_m_compareall_noise/Data_Sample/d10/data_V_2_0.txt')
y_stack = np.zeros(m * N)
for i in range(N):
    x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
    y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
    SNR=12
    s=10**(-1.0*(SNR/10.0))/T * (np.linalg.norm(x_o, ord=2)**2)
    noise_add = np.random.randn(m)*np.sqrt(s)
    y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add


avr_com_mse = np.zeros((num_compare, Max_iter))
avr_loc_mse = np.zeros((num_compare, Max_iter))
cserr = np.zeros((num_compare, Max_iter))

for i in range(2):
    alpha=Alpha_l[i]
    print("the ADMM-VPD:\n")
    avr_com_mse[0, :], avr_loc_mse[0, :]\
        = ADMM_function.decentral_l1_VR_penalty_c_ind_hard(a_stack, y_stack, X_est, v_stack, N,
                                                           Adjacent_Matrix, alpha, c2, gamma_new, Max_iter,
                                                          eta, USTp, Umin)

    print("VPD-JSM1 Algorithm")
    avr_com_mse[1, :], avr_loc_mse[1, :]\
        = ADMM_function.ADMM_VPD_JSM1_ind(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, alpha,
                                                      c22, rho2, gamma12, gamma22, gamma32,
                                                      c2, gamma_new, Max_iter, eta, USTp, Umin)

    for j in range(num_compare-1):
        np.savetxt('./new_compare_alpha/convergence/%d_' % i + 'avr_com_mse%d.txt' % j,
                   avr_com_mse[j, :])
        np.savetxt('./new_compare_alpha/convergence/%d_' % i + 'avr_loc_mse%d.txt' % j,
                   avr_loc_mse[j, :])


print("JSM1")
avr_com_mse[2, :], avr_loc_mse[2, :]\
        = ADMM_function.baseline_JSM1(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, c, rho,
                                      gamma1, gamma2, gamma3, Max_iter)

np.savetxt('./new_compare_alpha/convergence/avr_com_mse%d.txt' % 2,  avr_com_mse[2, :])
np.savetxt('./new_compare_alpha/convergence/avr_loc_mse%d.txt' % 2, avr_loc_mse[2, :])
'''
gap=1
num=int(np.ceil(Max_iter/gap))
avr_c0=np.zeros((len(Alpha_l), num))
avr_l0=np.zeros((len(Alpha_l), num))
avr_c1=np.zeros((len(Alpha_l), num))
avr_l1=np.zeros((len(Alpha_l), num))
cserr0=np.zeros((len(Alpha_l), num))
cserr1=np.zeros((len(Alpha_l), num))

avr_c2=np.zeros(num)
avr_l2=np.zeros(num)
cserr2=np.zeros(num)

for i in range(len(Alpha_l)):
    alpha=Alpha_l[i]
    avr_c01 = np.loadtxt('./new_compare_alpha/convergence/%d_' % i + 'avr_com_mse0.txt')
    avr_l01 = np.loadtxt('./new_compare_alpha/convergence/%d_' % i + 'avr_loc_mse0.txt')

    avr_c11 = np.loadtxt('./new_compare_alpha/convergence/%d_' % i + 'avr_com_mse1.txt')
    avr_l11 = np.loadtxt('./new_compare_alpha/convergence/%d_' % i + 'avr_loc_mse1.txt')

    for j in range(num):
        avr_c0[i, j] = avr_c01[j*gap]
        avr_l0[i, j] = avr_l01[j*gap]
        avr_c1[i, j] = avr_c11[j*gap]
        avr_l1[i, j] = avr_l11[j*gap]

avr_c21 = np.loadtxt('./new_compare_alpha/convergence/avr_com_mse2.txt')
avr_l21 = np.loadtxt('./new_compare_alpha/convergence/avr_loc_mse2.txt')

for j in range(num):
    avr_c2[j] = avr_c21[j * gap]
    avr_l2[j] = avr_l21[j * gap]


X=np.linspace(0, Max_iter, num)
'''
plt.plot(X, avr_c0[0, :], linewidth=1.5, label="Global Estimation(ADMM-VPD$\{alpha}=0.2$)", color='b', linestyle=':')
plt.plot(X, avr_l0[0, :], linewidth=1.5, label="Local Estimation(ADMM-VPD$\{alpha}=0.2$)", color='b', linestyle=':')
plt.plot(X, avr_c0[1, :], linewidth=1.5, label="Global Estimation(ADMM-VPD$\{alpha}=0.5$)", color='g', linestyle='-.')
plt.plot(X, avr_l0[1, :], linewidth=1.5, label="Local Estimation(ADMM-VPD$\{alpha}=0.5$)", color='g', linestyle='-.')
plt.plot(X, avr_c0[2, :], linewidth=1.5, label="Global Estimation(ADMM-VPD$\{alpha}=0.8$)", color='r', linestyle='--')
plt.plot(X, avr_l0[2, :], linewidth=1.5, label="Local Estimation(ADMM-VPD$\{alpha}=0.8$)", color='r', linestyle='--')
plt.plot(X, avr_c2[:], linewidth=1.5, label="Global Estimation(MJSM-1A)", color='y')
plt.plot(X, avr_l2[:], linewidth=1.5, label="Local Estimation(MJSM-1A)", color='y')

plt.scatter(X, avr_c0[0, :], label="Global Estimation(VPD$\{alpha}=0.2$)", color='b', marker='+')
plt.scatter(X, avr_l0[0, :], label="Local Estimation(VPD$\{alpha}=0.2$)", color='b', marker='o')
plt.scatter(X, avr_c0[1, :], label="Global Estimation(VPD$\{alpha}=0.5$)", color='g', marker='D')
plt.scatter(X, avr_l0[1, :], label="Local Estimation(VPD$\{alpha}=0.5$)", color='g', marker='x')
plt.scatter(X, avr_c0[2, :], label="Global Estimation(VPD$\{alpha}=0.8$)", color='r', marker='<')
plt.scatter(X, avr_l0[2, :], label="Local Estimation(VPD$\{alpha}=0.8$)", color='r', marker='X')
plt.scatter(X, avr_c2[:], label="Global Estimation(JSM1)", color='y', marker='>')
plt.scatter(X, avr_l2[:], label="Local Estimation(JSM1)", color='y', marker='*')
'''

font1={'family':'Times New Roman', 'size': 30}
font2={'family':'Times New Roman', 'size': 23}
X2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
figure1=plt.figure(1)
ax=plt.subplot(111)
plt.plot(X, avr_c0[0, :], linewidth=1.5, label="Glo_AMSE(VPD-ADMM "+r'$\alpha=0.2$'+")", color='b', linestyle='-.', markevery=X2, marker='o')
plt.plot(X, avr_l0[0, :], linewidth=1.5, label="Loc_AMSE(VPD-ADMM "+r'$\alpha=0.2$'+")", color='b', linestyle=':', markevery=X2, marker='s')
plt.plot(X, avr_c0[1, :], linewidth=1.5, label="Glo_AMSE(VPD-ADMM "+r'$\alpha=0.8$'+")", color='r', linestyle='-.', markevery=X2, marker='x')
plt.plot(X, avr_l0[1, :], linewidth=1.5, label="Loc_AMSE(VPD-ADMM "+r'$\alpha=0.8$'+")", color='r', linestyle=':', markevery=X2, marker='+')
#plt.plot(X, avr_c0[2, :], linewidth=1.5, label="Global Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
# plt.plot(X, avr_l0[2, :], linewidth=1.5, label="Local Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
plt.plot(X, avr_c2[:], linewidth=1.5, label="Glo_AMSE(D-AJSM)", color='y', linestyle='-')
plt.plot(X, avr_l2[:], linewidth=1.5, label="Loc_AMSE(D-AJSM)", color='y', linestyle='--')
xLocator=MultipleLocator(10)
ax.xaxis.set_major_locator(xLocator)
plt.xlim(1, Max_iter)
plt.ylim(0, 0.1)
plt.tick_params(labelsize=20)
# plt.title("MSE")
plt.xlabel("Iter", font1)
plt.ylabel("AMSE", font1)
plt.grid(False)
plt.legend(prop=font2)

ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)

axins = ax.inset_axes((0.1, 0.6, 0.2, 0.3))
axins.plot(X, avr_c0[0, :], linewidth=1.5, label="Glo_AMSE(VPD-ADMM "+r'$\alpha=0.2$'+")", color='b', linestyle='-.')
axins.plot(X, avr_l0[0, :], linewidth=1.5, label="Local Estimation(VPD-ADMM "+r'$\alpha=0.2$'+")", color='b', linestyle=':')
axins.plot(X, avr_c0[1, :], linewidth=1.5, label="Global Estimation(VPD-ADMM "+r'$\alpha=0.8$'+")", color='r', linestyle='-.')
axins.plot(X, avr_l0[1, :], linewidth=1.5, label="Local Estimation(VPD-ADMM "+r'$\alpha=0.8$'+")", color='r', linestyle=':')
#plt.plot(X, avr_c0[2, :], linewidth=1.5, label="Global Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
# plt.plot(X, avr_l0[2, :], linewidth=1.5, label="Local Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
axins.plot(X, avr_c2[:], linewidth=1.5, label="Glo_AMSE(D-AJSM)", color='y', linestyle='-')
axins.plot(X, avr_l2[:], linewidth=1.5, label="Glo_AMSE(D-AJSM)", color='y', linestyle='--')
axins.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

# sub region of the original image
x1, x2 = 95, 100
xx1=int(np.ceil(x1/gap))-1
xx2=int(np.ceil(x2/gap))-1
y1 = min([avr_c0[0, xx2], avr_l0[0, xx2], avr_c0[1, xx2], avr_l0[1, xx2]])-0.001
y2 = max([avr_c0[0, xx1], avr_l0[0, xx1], avr_c0[1, xx1], avr_l0[1, xx1]])+0.001
# y2 = max([avr_c2[xx1], avr_l2[xx1]])+0.002
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# fix the number of ticks on the inset axes
axins.yaxis.get_major_locator().set_params(nbins=5)
axins.xaxis.get_major_locator().set_params(nbins=3)
#plt.xticks(fontsize=13)
#plt.yticks(fontsize=13)
plt.xticks(visible=True)
plt.yticks(visible=True)
axins.grid(False)
axins.tick_params(labelsize=20)
# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")


figure3=plt.figure(2)
ax2=plt.subplot(111)
'''
plt.scatter(X, avr_c1[0, :], label="(Global Estimation)alpha=0.2", color='b', marker='+')
plt.scatter(X, avr_l1[0, :], label="(Local Estimation)alpha=0.2", color='b', marker='o')
plt.scatter(X, avr_c1[1, :], label="(Global Estimation)alpha=0.5", color='g', marker='D')
plt.scatter(X, avr_l1[1, :], label="(Local Estimation)alpha=0.5", color='g', marker='x')
plt.scatter(X, avr_c1[2, :], label="(Global Estimation)alpha=0.8", color='r', marker='<')
plt.scatter(X, avr_l1[2, :], label="(Local Estimation)alpha=0.8", color='r', marker='X')
plt.scatter(X, avr_c2[:], label="Global Estimation(JSM1)", color='y', marker='>')
plt.scatter(X, avr_l2[:], label="Local Estimation(JSM1)", color='y', marker='*')'''
plt.plot(X, avr_c1[0, :], linewidth=1.5, label="Glo_AMSE(VPD-EM "+r'$\alpha=0.2$'+")", color='b', linestyle='-.', markevery=X2, marker='o')
plt.plot(X, avr_l1[0, :], linewidth=1.5, label="Loc_AMSE(VPD-EM "+r'$\alpha=0.2$'+")", color='b', linestyle=':', markevery=X2, marker='s')
plt.plot(X, avr_c1[1, :], linewidth=1.5, label="Glo_AMSE(VPD-EM "+r'$\alpha=0.8$'+")", color='r', linestyle='-.', markevery=X2, marker='x')
plt.plot(X, avr_l1[1, :], linewidth=1.5, label="Loc_AMSE(VPD-EM "+r'$\alpha=0.8$'+")", color='r', linestyle=':', markevery=X2, marker='+')
#plt.plot(X, avr_c0[2, :], linewidth=1.5, label="Global Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
# plt.plot(X, avr_l0[2, :], linewidth=1.5, label="Local Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
plt.plot(X, avr_c2[:], linewidth=1.5, label="Glo_AMSE(D-AJSM)", color='y', linestyle='-')
plt.plot(X, avr_l2[:], linewidth=1.5, label="Loc_AMSE(D-AJSM)", color='y', linestyle='--')
xLocator=MultipleLocator(10)
ax2.xaxis.set_major_locator(xLocator)
plt.xlim(1, Max_iter)
plt.ylim(0, 0.1)
# plt.title("MSE")
plt.tick_params(labelsize=20)
plt.xlabel("Iter", font1)
plt.ylabel("AMSE", font1)
plt.grid(False)
plt.legend(prop=font2)

ax2.xaxis.grid(False, which='major')
ax2.yaxis.grid(False)

axins2 = ax2.inset_axes((0.1, 0.6, 0.2, 0.3))
axins2.plot(X, avr_c1[0, :], linewidth=1.5, label="Global Estimation(VPD-EM "+r'$\alpha=0.2$'+")", color='b', linestyle='-.')
axins2.plot(X, avr_l1[0, :], linewidth=1.5, label="Local Estimation(VPD-EM "+r'$\alpha=0.2$'+")", color='b', linestyle=':')
axins2.plot(X, avr_c1[1, :], linewidth=1.5, label="Global Estimation(VPD-EM "+r'$\alpha=0.8$'+")", color='r', linestyle='-.')
axins2.plot(X, avr_l1[1, :], linewidth=1.5, label="Local Estimation(VPD-EM "+r'$\alpha=0.8$'+")", color='r', linestyle=':')
#plt.plot(X, avr_c0[2, :], linewidth=1.5, label="Global Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
# plt.plot(X, avr_l0[2, :], linewidth=1.5, label="Local Estimation(VPD$\{alpha}=0.8$)", color='r', linestyle='--')
axins2.plot(X, avr_c2[:], linewidth=1.5, label="Global Estimation(MJSM-1A)", color='y', linestyle='-')
axins2.plot(X, avr_l2[:], linewidth=1.5, label="Local Estimation(MJSM-1A)", color='y', linestyle='--')
axins2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
# sub region of the original image
x1, x2 = 95, 100
xx1=int(np.ceil(x1/gap))-1
xx2=int(np.ceil(x2/gap))-1
y1 = min([avr_c1[0, xx2], avr_l1[0, xx2], avr_c1[1, xx2], avr_l1[1, xx2]])-0.001
# y2 = max([avr_c2[xx1], avr_l2[xx1]])+0.001
y2 = max([avr_c1[0, xx1], avr_l1[0, xx1], avr_c1[1, xx1], avr_l1[1, xx1]])+0.001
axins2.set_xlim(x1, x2)
axins2.set_ylim(y1, y2)
# fix the number of ticks on the inset axes
axins2.yaxis.get_major_locator().set_params(nbins=3)
axins2.xaxis.get_major_locator().set_params(nbins=3)
#plt.xticks(fontsize=13)
#plt.yticks(fontsize=13)
plt.xticks(visible=True)
plt.yticks(visible=True)
axins2.tick_params(labelsize=20)
axins2.grid(False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax2, axins2, loc1=1, loc2=4, fc="none", ec="0.5")
plt.show()
