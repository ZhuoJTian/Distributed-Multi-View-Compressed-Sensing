import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

avr_com_mse1 = np.loadtxt('./new_T2/avr_com1.txt')
avr_com_mse2 = np.loadtxt('./new_T2/avr_com2.txt')
avr_com_mse3 = np.loadtxt('./new_T2/avr_com3.txt')
avr_com_mse4 = np.loadtxt('./new_T2/avr_com4.txt')

avr_loc_mse1 = np.loadtxt('./new_T2/avr_loc1.txt')
avr_loc_mse2 = np.loadtxt('./new_T2/avr_loc2.txt')
avr_loc_mse3 = np.loadtxt('./new_T2/avr_loc3.txt')
avr_loc_mse4 = np.loadtxt('./new_T2/avr_loc4.txt')

avr_v1 = np.loadtxt('./new_T2/avr_v1.txt')
avr_v2 = np.loadtxt('./new_T2/avr_v2.txt')
avr_v3 = np.loadtxt('./new_T2/avr_v3.txt')
avr_v4 = np.loadtxt('./new_T2/avr_v4.txt')


font1={'family':'Times New Roman', 'size':30}
font2={'family':'Times New Roman', 'size':30}
X=np.linspace(0, 100, 100)

plt.rcParams['mathtext.default'] = 'regular'
I_c= range(len(X)) # [0,2,4,6,8, 10]

figure1=plt.figure(1)
ax=plt.subplot(111)
plt.plot(X, avr_com_mse1, linewidth=3, label='NoDiscre', color='red', linestyle='--', marker='o', markersize=2)
plt.plot(X, avr_com_mse2, linewidth=3, label='NoEarlyStop', color='purple', linestyle='--', marker='>', markersize=2)
plt.plot(X, avr_com_mse3, linewidth=3, label='VPD-ADMM', color='blue', linestyle='--', marker='*', markersize=2)
plt.plot(X, avr_com_mse4, linewidth=3, label='VPD-EM', color='deepskyblue', linestyle='--', marker='s', markersize=2)
xLocator=MultipleLocator(20)
ax.xaxis.set_major_locator(xLocator)
# yLocator=MultipleLocator(0.03)
# ax.yaxis.set_major_locator(yLocator)
# plt.title("avr_com_mse")
# plt.yscale('log')
plt.xlabel("M", font1)
plt.ylabel("Glo_AMSE", font1)
plt.tick_params(labelsize=25)
plt.grid(False)
plt.legend(prop=font2)

'''
ax2.xaxis.grid(False, which='major')
ax2.yaxis.grid(False)

axins = ax2.inset_axes((0.1, 0.6, 0.2, 0.3))
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
'''


figure2=plt.figure(2)
ax2=plt.subplot(111)
plt.plot(X, avr_loc_mse3, linewidth=3, label='VPD-ADMM', color='green', linestyle='-', marker='*', markersize=2) #blue
plt.plot(X, avr_loc_mse4, linewidth=3, label='VPD-EM', color='deepskyblue', linestyle='-.', marker='s', markersize=2) #deepskyblue
plt.plot(X, avr_loc_mse1, linewidth=3, label='NoDiscre', color='red', linestyle='--', marker='o', markersize=2)
plt.plot(X, avr_loc_mse2, linewidth=3, label='NoEarlyStop', color='purple', linestyle='--', marker='>', markersize=2)
xLocator=MultipleLocator(20)
ax2.xaxis.set_major_locator(xLocator)
# yLocator=MultipleLocator(0.03)
# ax.yaxis.set_major_locator(yLocator)
# plt.title("avr_com_mse")
# plt.yscale('log')
plt.xlabel("Iter", font1)
plt.ylabel("Loc_AMSE", font1)
plt.tick_params(labelsize=25)
plt.grid(False)
plt.legend(prop=font2)

plt.show()
