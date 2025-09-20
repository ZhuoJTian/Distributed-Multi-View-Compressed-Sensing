import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

B_one = [5, 15, 20, 25] #, 15, 20, 25
M = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]

M_c =[30, 90, 150, 210, 270, 330]# [30, 50, 70, 90, 110] # [30, 60, 90, 120]#
avr_com = np.zeros((len(B_one),len(M)))
avr_loc = np.zeros((len(B_one),len(M)))

for bi in range(len(B_one)):
    for mi in range(len(M)):
        ac = np.loadtxt('./new_T1/convergence/%d/' %B_one[bi] + 'avr_com_mse%d.txt' % mi)
        al = np.loadtxt('./new_T1/convergence/%d/' %B_one[bi] + 'avr_loc_mse%d.txt' % mi)
        avr_com[bi, mi] = np.average(ac)
        avr_loc[bi, mi] = np.average(al)


font1={'family':'Times New Roman', 'size':30}
font2={'family':'Times New Roman', 'size':30}
X=M
plt.rcParams['mathtext.default'] = 'regular'
I_c= range(len(X)) # [0,2,4,6,8, 10]
figure1=plt.figure(1)
ax=plt.subplot(111)
plt.plot(X, avr_com[0, I_c], linewidth=3, label=r"$|\Pi(\{0\})|=5$", color='red', linestyle='--', marker='o', markersize=8)
plt.plot(X, avr_com[1, I_c], linewidth=3, label=r"$|\Pi(\{0\})|=15$", color='purple', linestyle='--', marker='>', markersize=8)
plt.plot(X, avr_com[2, I_c], linewidth=3, label=r"$|\Pi(\{0\})|=20$", color='green', linestyle='--', marker='*', markersize=8)
plt.plot(X, avr_com[3, I_c], linewidth=3, label=r"$|\Pi(\{0\})|=25$", color='deepskyblue', linestyle='--', marker='s', markersize=8)
plt.xlim(min(M), max(M))
# plt.ylim(0.001, 0.1)
xLocator=MultipleLocator(60)
ax.xaxis.set_major_locator(xLocator)
# yLocator=MultipleLocator(0.03)
# ax.yaxis.set_major_locator(yLocator)
# plt.title("avr_com_mse")
# plt.yscale('log')
plt.xlabel(r"$M_0$", font1)
plt.ylabel("Glo_AMSE", font1)
plt.tick_params(labelsize=25)
plt.grid(False)
plt.legend(prop=font2)

figure2=plt.figure(2)
ax2=plt.subplot(111)
plt.plot(X, avr_loc[0, I_c], linewidth=2, label="IRAS", color='b', linestyle='-', marker='o', markersize=10)
plt.plot(X, avr_loc[1, I_c], linewidth=2, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=10)
plt.plot(X, avr_loc[2, I_c], linewidth=2, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=10)
plt.plot(X, avr_loc[3, I_c], linewidth=2, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=10)
plt.xlim(min(M), max(M))
plt.ylim(0, 0.06)
xLocator=MultipleLocator(60)
ax.xaxis.set_major_locator(xLocator)
# yLocator=MultipleLocator(0.03)
# ax.yaxis.set_major_locator(yLocator)
# plt.title("avr_com_mse")
# plt.yscale('log')
plt.xlabel("M", font1)
plt.ylabel("Loc_AMSE", font1)
plt.tick_params(labelsize=25)
plt.grid(False)
plt.legend(prop=font2)

plt.show()
