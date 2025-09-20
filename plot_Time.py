import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

avr_com_mse1 = np.loadtxt('./new_time/M120/avr_com1.txt')
avr_com_mse2 = np.loadtxt('./new_time/M120/avr_com2.txt')
avr_com_mse3 = np.loadtxt('./new_time/M120/avr_com3.txt')
avr_com_mse4 = np.loadtxt('./new_time/M120/avr_com4.txt')

avr_loc_mse1 = np.loadtxt('./new_time/M120/avr_loc1.txt')
avr_loc_mse2 = np.loadtxt('./new_time/M120/avr_loc2.txt')
avr_loc_mse3 = np.loadtxt('./new_time/M120/avr_loc3.txt')
avr_loc_mse4 = np.loadtxt('./new_time/M120/avr_loc4.txt')

time1 = np.loadtxt('./new_time/M120/time1.txt')
time2 = np.loadtxt('./new_time/M120/time2.txt')
time3 = np.loadtxt('./new_time/M120/time3.txt')
time4 = np.loadtxt('./new_time/M120/time4.txt')

sum_time1 = [np.sum(time1[0:k]) for k in range(69)]
sum_time2 = [np.sum(time2[0:k]) for k in range(100)]
sum_time3 = [np.sum(time3[0:k]) for k in range(160)]
sum_time4 = [np.sum(time4[0:k]) for k in range(79)]

font1={'family':'Times New Roman', 'size':30}
font3={'family':'Times New Roman', 'size':35}
font2={'family':'Times New Roman', 'size':20}

figure1=plt.figure(1)
ax1 = figure1.add_subplot(111)

line11 = ax1.plot(sum_time1, avr_com_mse1[0:69], linestyle='-', linewidth = 3, label='D-LASSO')  # 'VPD-ADMMA', 'VPD-EM'
# line12 = ax1.plot(sum_time2, avr_com_mse2, label = 'D-AJSM')
line13 = ax1.plot(sum_time3, avr_com_mse3, linestyle='-', linewidth = 3, label = 'VPD-ADMM')
line14 = ax1.plot(sum_time4, avr_com_mse4[0:79], linestyle='-', linewidth = 3, label = 'VPD-EM')
xLocator=MultipleLocator(5)
ax1.xaxis.set_major_locator(xLocator)
yLocator=MultipleLocator(0.02)
ax1.yaxis.set_major_locator(yLocator)
# yLocator=MultipleLocator(0.03)
# ax2.yaxis.set_major_locator(yLocator)
# plt.title("avr_loc_mse")
plt.xlabel("Run Time (s)", font1)
plt.ylabel("Glo_AMSE", font1)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font3)

figure2=plt.figure(2)
ax2 = figure2.add_subplot(111)

line21 = ax2.plot(sum_time1, avr_loc_mse1[0:69], linestyle='-', linewidth = 3, label='D-LASSO')  # 'VPD-ADMMA', 'VPD-EM'
# line22 = ax1.plot(sum_time2, avr_loc_mse2, label = 'D-AJSM')
line23 = ax2.plot(sum_time3, avr_loc_mse3, linestyle='-', linewidth = 3, label = 'VPD-ADMM')
line24 = ax2.plot(sum_time4, avr_loc_mse4[0:79], linestyle='-', linewidth = 3, label = 'VPD-EM')
xLocator=MultipleLocator(5)
ax2.xaxis.set_major_locator(xLocator)
yLocator=MultipleLocator(0.02)
ax2.yaxis.set_major_locator(yLocator)
# yLocator=MultipleLocator(0.03)
# ax2.yaxis.set_major_locator(yLocator)
# plt.title("avr_loc_mse")
plt.xlabel("Run Time (s)", font1)
plt.ylabel("Loc_AMSE", font1)
plt.ylim(0, 0.06)
plt.tick_params(labelsize=20)
plt.grid(False)
plt.legend(prop=font3)

plt.show()