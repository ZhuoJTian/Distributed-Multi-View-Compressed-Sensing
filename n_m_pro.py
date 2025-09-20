import numpy as np
import random
import ADMM_function
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#20, 30, 40, 50, 60
# 40, 50, 60, 70, 80, 90,
# [30, 50, 70, 90, 110]  #40, 60, 80, 100, 120
    #[20, 30, 40, 50, 60, 70, 80, 90] # the measurement size in each sensor
# M=[20, 40, 60, 80, 100, 120, 140, 160, 180]
# M=[40, 60, 80, 100, 120, 140] #10  #, 140, 160, 180,  200, 100, 120, 140
M=[60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    #[60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]  # 40不要啦  60.80.100, 120, done
T=500
N=6
K=50
d_blocked=[10] #5, 15, 25,   15和25的20,40，60还没跑
num_a=5   #2,2,4; 5,2,4
num_x=5
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

avr_c0=np.zeros((len(d_blocked), len(M)))
avr_l0=np.zeros((len(d_blocked), len(M)))
avr_c1=np.zeros((len(d_blocked), len(M)))
avr_l1=np.zeros((len(d_blocked), len(M)))
avr_c2=np.zeros((len(d_blocked), len(M)))
avr_l2=np.zeros((len(d_blocked), len(M)))
avr_c3=np.zeros((len(d_blocked), len(M)))
avr_l3=np.zeros((len(d_blocked), len(M)))

avr_ca0=np.zeros((len(d_blocked), len(M)))
avr_la0=np.zeros((len(d_blocked), len(M)))
avr_ca1=np.zeros((len(d_blocked), len(M)))
avr_la1=np.zeros((len(d_blocked), len(M)))
avr_ca2=np.zeros((len(d_blocked), len(M)))
avr_la2=np.zeros((len(d_blocked), len(M)))
avr_ca3=np.zeros((len(d_blocked), len(M)))
avr_la3=np.zeros((len(d_blocked), len(M)))


for d in d_blocked:
    for i in range(len(M)):
        m = M[i]
        avr_com_mse1 = np.zeros((num_compare, 40, Max_iter))
        avr_loc_mse1 = np.zeros((num_compare, 40, Max_iter))
        avr_com_mse2 = np.zeros((num_compare, repeat_times, Max_iter))
        avr_loc_mse2 = np.zeros((num_compare, repeat_times, Max_iter))
        avr_cc0 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse0.txt')[0:40, :]
        avr_ll0 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse0.txt')[0:40, :]

        for j in range(num_compare):
            avr_com_mse1[j, :, :] = np.loadtxt(
                './new_m_compareall_noise/convergence/d%d/' % d + '/%d_' % m + 'avr_com_mse%d.txt' % j)
            avr_loc_mse1[j, :, :] = np.loadtxt(
                './new_m_compareall_noise/convergence/d%d/' % d + '/%d_' % m + 'avr_loc_mse%d.txt' % j)
            avr_com_mse2[j, :, :] = np.loadtxt(
                './new_m_compareall_noise/convergence2/d%d/' % d + '/%d_' % m + 'avr_com_mse%d.txt' % j)
            avr_loc_mse2[j, :, :] = np.loadtxt(
                './new_m_compareall_noise/convergence2/d%d/' % d + '/%d_' % m + 'avr_loc_mse%d.txt' % j)

        for j in range(num_compare):
            avr_com_mse2[j, 0:8, :]= avr_com_mse1[j, 0:8, :]
            avr_loc_mse2[j, 0:8, :] = avr_loc_mse1[j, 0:8, :]

            avr_com_mse2[j, 20:28, :] = avr_com_mse1[j, 8:16, :]
            avr_loc_mse2[j, 20:28, :] = avr_loc_mse1[j, 8:16, :]

            avr_com_mse2[j, 40:48, :] = avr_com_mse1[j, 16:24, :]
            avr_loc_mse2[j, 40:48, :] = avr_loc_mse1[j, 16:24, :]

            avr_com_mse2[j, 60:68, :] = avr_com_mse1[j, 24:32, :]
            avr_loc_mse2[j, 60:68, :] = avr_loc_mse1[j, 24:32, :]

            avr_com_mse2[j, 80:88, :] = avr_com_mse1[j, 32:40, :]
            avr_loc_mse2[j, 80:88, :] = avr_loc_mse1[j, 32:40, :]


        for j in range(num_compare):
            np.savetxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_com_mse%d.txt' % j,
                       avr_com_mse2[j, :, :])
            np.savetxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_loc_mse%d.txt' % j,
                       avr_loc_mse2[j, :, :])


for j in range(len(d_blocked)):
    for i in range(len(M)):
        m = M[i]
        d = d_blocked[j]
        avr_cc0 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_com_mse0.txt')
        avr_ll0 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_loc_mse0.txt')
        # cserr00 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr0.txt')

        avr_cc1 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_com_mse1.txt')
        avr_ll1 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_loc_mse1.txt')
        # cserr11 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr1.txt')

        avr_cc2 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_com_mse2.txt')
        avr_ll2 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_loc_mse2.txt')
        # cserr22 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr2.txt')

        avr_cc3 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_com_mse3.txt')
        avr_ll3 = np.loadtxt('./new_m_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'avr_loc_mse3.txt')
        # cserr33 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr3.txt')

        avr_c0[j, i] = np.average(avr_cc0[:, 0]) #:,
        avr_l0[j, i] = np.average(avr_ll0[:, 0])
        # cserr0[j, i] = np.average(cserr00[:, 0])

        avr_c1[j, i] = np.average(avr_cc1[:,Max_iter - 1])
        avr_l1[j, i] = np.average(avr_ll1[:,Max_iter - 1])
        # cserr1[j, i] = np.average(cserr11[:,Max_iter - 1])

        avr_c2[j, i] = np.average(avr_cc2[:, Max_iter - 1])
        avr_l2[j, i] = np.average(avr_ll2[:, Max_iter - 1])
        # cserr2[j, i] = np.average(cserr22[:, Max_iter - 1])

        avr_c3[j, i] = np.average(avr_cc3[:, Max_iter - 1])
        avr_l3[j, i] = np.average(avr_ll3[:, Max_iter - 1])

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

font1={'family':'Times New Roman', 'size':40}
font2={'family':'Times New Roman', 'size':40}
X=M
figure1=plt.figure(1)
ax=plt.subplot(111)
plt.plot(X, avr_c0[0, :], linewidth=2, label="BL(d=%d)"%d_blocked[0], color='b', linestyle='-', marker='o')
plt.plot(X, avr_c2[0, :], linewidth=2, label="MJSM-1A(d=%d)"%d_blocked[0], color='r', linestyle=':', marker='>')
plt.plot(X, avr_c1[0, :], linewidth=2, label="ADMM-VPD(d=%d)"%d_blocked[0], color='g', linestyle='-.', marker='*')
plt.plot(X, avr_c3[0, :], linewidth=2, label="VPD-EC(d=%d)"%d_blocked[0], color='y', linestyle='--', marker='s')
plt.xlim(min(M), max(M))
plt.ylim(0, 0.12)
xLocator=MultipleLocator(20)
ax.xaxis.set_major_locator(xLocator)
# plt.title("avr_com_mse")
plt.xlabel("M", font1)
plt.ylabel("Glo_AMSE", font1)
plt.tick_params(labelsize=15)
# plt.grid(True)
plt.legend(prop=font2)
# ax.xaxis.grid(True, which='major')
# ax.yaxis.grid(True)

axins = inset_axes(ax, width="30%", height="40%", loc=5)  # zoom = 6
axins.plot(X, avr_c0[0, :], linewidth=2, label="BL(d=%d)"%d_blocked[0], color='b', linestyle='-', marker='o')
axins.plot(X, avr_c2[0, :], linewidth=2, label="MJSM-1A(d=%d)"%d_blocked[0], color='r', linestyle=':', marker='>')
axins.plot(X, avr_c1[0, :], linewidth=2, label="ADMM-VPD(d=%d)"%d_blocked[0], color='g', linestyle='-.', marker='*')
axins.plot(X, avr_c3[0, :], linewidth=2, label="VPD-EC(d=%d)"%d_blocked[0], color='y', linestyle='--', marker='s')
axins.ticklabel_format(style='sci',scilimits=(0,0),axis='y')

# sub region of the original image
x1, x2 = 60, 100
xx1=0
xx2=4
y1 = avr_c3[0, xx2]-0.002
y2 = max([avr_c1[0, xx1], avr_c3[0, xx1]])+0.002
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# fix the number of ticks on the inset axes
axins.yaxis.get_major_locator().set_params(nbins=10)
axins.xaxis.get_major_locator().set_params(nbins=5)
#plt.xticks(fontsize=13)
#plt.yticks(fontsize=13)
plt.xticks(visible=True)
plt.yticks(visible=True)
plt.tick_params(labelsize=15)
# axins.grid(True)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")


figure2=plt.figure(2)
ax2=plt.subplot(111)
plt.plot(X, avr_l0[0, :], linewidth=2, label="BL(d=%d)"%d_blocked[0], color='b', linestyle='-', marker='o')
plt.plot(X, avr_l2[0, :], linewidth=2, label="MJSM-1A(d=%d)"%d_blocked[0], color='r', linestyle=':', marker='>')
plt.plot(X, avr_l1[0, :], linewidth=2, label="ADMM-VPD(d=%d)"%d_blocked[0], color='g', linestyle='-.', marker='*')
plt.plot(X, avr_l3[0, :], linewidth=2, label="VPD-EC(d=%d)"%d_blocked[0], color='y', linestyle='--', marker='s')
plt.xlim(min(M), max(M))
plt.ylim(0, 0.08)
xLocator=MultipleLocator(10)
ax2.xaxis.set_major_locator(xLocator)
# plt.title("avr_loc_mse")
plt.xlabel("M", font1)
plt.ylabel("Loc_AMSE", font1)
plt.tick_params(labelsize=15)
# plt.grid(True)
plt.legend(prop=font2)
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