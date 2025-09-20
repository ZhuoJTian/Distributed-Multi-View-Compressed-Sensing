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

m=60  #, 100
T=500
N=6
K=50
d_blocked=[10]
alpha=0.9# [round(i*0.05, 2) for i in range(0, 20)]
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

Max_iter=100

com_mse = np.zeros(Max_iter)
loc_mse = np.zeros(Max_iter)
Umin=30

USTp = 40
c2 = 1.0
gamma_new = 0.2
eta = 10

t1=1
t2=1
t3=1

for d in d_blocked:
    com_mse = np.zeros(Max_iter)
    loc_mse = np.zeros(Max_iter)
    v_mse = np.zeros(Max_iter)
    a_stack = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_A_%d' % m + '_%d.txt' % t1)  # './new_d_compareall_noise/Data_Sample/
    X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_%d.txt' % t2)
    print(m, d, t1, t2, t3)
    v_stack = np.loadtxt('./new_d_compareall_noise/Data_Sample/d%d' % d + '/data_V_%d' % t2 + '_%d.txt' % t3)
    y_stack = np.zeros(m * N)
    for i in range(N):
        x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
        y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
        SNR = 10
        s = 10 ** (-1.0 * (SNR / 10.0)) / T * (np.linalg.norm(x_o, ord=2) ** 2)
        noise_add = np.random.randn(m) * np.sqrt(s)
        y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add
    print("SNR10, the ADMM-VPD: %.2f \n" % alpha)
    com_mse, loc_mse, v_mse = ADMM_function.curve_v(a_stack, y_stack, X_est, v_stack, N,
                                                    Adjacent_Matrix, alpha, c2, gamma_new, Max_iter, eta, USTp, Umin)

    np.savetxt('./new_plot_V/%d' % m + '_avr_com_mse0.txt', com_mse)
    np.savetxt('./new_plot_V/%d' % m + '_avr_loc_mse0.txt', loc_mse)
    np.savetxt('./new_plot_V/%d' % m + '_avr_v_mse0.txt', v_mse)

font1={'family':'Times New Roman', 'size':30}
font2={'family':'Times New Roman', 'size':23}
figure1=plt.figure(1)
num = Max_iter
X=np.linspace(0, Max_iter, num)
ax=plt.subplot(111)
plt.plot(X, v_mse, linewidth=1.5, color='b', linestyle='--')

plt.xlim(0, Max_iter)
plt.tick_params(labelsize=15)
plt.xlabel("Alpha (d=10)", font=font1)
plt.ylabel("AMSE", font=font1)
plt.legend(prop=font2)

plt.show()