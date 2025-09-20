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
d_blocked=[20]  # 15, 20
Alpha_l=[0, 0.3, 0.6, 0.9, 0.99]# [round(i*0.05, 2) for i in range(0, 20)]

Adjacent_Matrix=np.matrix([[0, 0, 1, 1, 1, 1]
                    , [0, 0, 1, 1, 1, 0]
                    , [1, 1, 0, 1, 0, 0]
                    , [1, 1, 1, 0, 1, 1]
                    , [1, 1, 0, 1, 0, 1]
                    , [1, 0, 0, 1, 1, 0]])

Max_iter=70
num_compare=len(Alpha_l)

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
font1={'family':'Times New Roman', 'size':20}
font2={'family':'Times New Roman', 'size':15}

print("p=0.2")
for i in range(len(M)):
    print(ac[i, 0, :]*1000)
    print(al[i, 0, :]*1000)

'''
print("p=0.4")
for i in range(len(M)):
    print(ac[i, 3, :]*1000)
    print(al[i, 3, :]*1000)


color_set=['r', 'b', 'y', 'g']
label_set=['M=60', 'M=70', 'M=80', 'M=90']

figure1=plt.figure(1)
ax=plt.subplot(111)
for i in range(len(M)):
    plt.plot(Alpha_l, ac[i, 0, :], linewidth=1.5, label=label_set[i], color=color_set[i], linestyle='-.', marker='o')

plt.xlim(0, max(Alpha_l))
plt.tick_params(labelsize=15)
plt.xlabel("Alpha (d=10)", font=font1)
plt.ylabel("AMSE", font=font1)
plt.legend(prop=font2)


figure2=plt.figure(2)
ax=plt.subplot(111)
for i in range(len(M)):
    plt.plot(Alpha_l, ac[i, 1, :], linewidth=1.5, label=label_set[i], color=color_set[i], linestyle='-.', marker='o')

plt.xlim(0, max(Alpha_l))
plt.tick_params(labelsize=15)
plt.xlabel("Alpha (d=20)", font=font1)
plt.ylabel("AMSE", font=font1)
plt.legend(prop=font2)



Y=np.linspace(0, 30, 31)
figure2=plt.figure(2)
ax2=plt.subplot(111)
plt.plot(Y, indV0, linewidth=1.5, label=r'$\alpha=0$', color='b', linestyle='-')
plt.plot(Y, indV1, linewidth=1.5, label=r'$\alpha=0.3$', color='r', linestyle='-.')
plt.plot(Y, indV2, linewidth=1.5, label=r'$\alpha=0.6$', color='g', linestyle='--')
plt.plot(Y, indV3, linewidth=1.5, label=r'$\alpha=0.9$', color='y', linestyle=':')
# xLocator=MultipleLocator(10)
# ax.xaxis.set_major_locator(xLocator)
plt.xlim(0, 30)
# plt.ylim(0, 0.1)
plt.tick_params(labelsize=15)
# plt.title("MSE")
plt.xlabel("Iter", font=font1)
plt.ylabel("DisV", font=font1)
plt.grid(True)
plt.legend(prop=font2)
plt.show()'''
