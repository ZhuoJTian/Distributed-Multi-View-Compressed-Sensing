import numpy as np
import random
import ADMM_function_T2 as AD
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
m=80
T=500
N=6
K=50
d_blocked=10
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

c = 0.05
gamma1 = 0.007
gamma2 = 0.2
gamma3 = 0.015
rho = 0.05

c2 = 1.0
gamma_new = 0.09
c22 = 0.5
rho2 = 0.5
gamma12 = 0.009
gamma22 = 0.1
gamma32 = 0.007
eta = 20
mu=0.2
USTp=35
Umin=20
alpha=0.3

a_stack = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_A_80_1.txt')  # './new_d_compareall_noise/Data_Sample/
X_est = np.loadtxt('./new_m_compareall_noise/Data_Sample/data_X_2.txt')
v_stack = np.loadtxt('./new_m_compareall_noise/Data_Sample/d10/data_V_2_1.txt')
y_stack = np.zeros(m * N)
for i in range(N):
    x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
    y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
    SNR=12
    s=10**(-1.0*(SNR/10.0))/T * (np.linalg.norm(x_o, ord=2)**2)
    noise_add = np.random.randn(m)*np.sqrt(s)
    y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add


print("No descretization:\n")
avr_com_mse1, avr_loc_mse1, avr_v1\
    = AD.VPDADMM_nohardmapping(a_stack, y_stack, X_est, v_stack, N,
                                                        Adjacent_Matrix, alpha, c2, gamma_new, Max_iter,
                                                        eta, USTp, Umin)

print("No earlystopping:\n")
avr_com_mse2, avr_loc_mse2, avr_v2\
    = AD.VPDADMM_noearlystopping(a_stack, y_stack, X_est, v_stack, N,
                                                        Adjacent_Matrix, alpha, c2, gamma_new, Max_iter,
                                                        eta, USTp, Umin)


print("the ADMM-VPD:\n")
avr_com_mse3, avr_loc_mse3, avr_v3\
    = AD.decentral_l1_VR_penalty_c_ind_hard(a_stack, y_stack, X_est, v_stack, N,
                                                        Adjacent_Matrix, alpha, c2, gamma_new, Max_iter,
                                                        eta, USTp, Umin)

print("VPD-JSM1 Algorithm")
avr_com_mse4, avr_loc_mse4, avr_v4\
    = AD.ADMM_VPD_JSM1_ind(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, alpha,
                            c22, rho2, gamma12, gamma22, gamma32,
                            c2, gamma_new, Max_iter, eta, USTp, Umin)


np.savetxt('./new_T2/avr_com1.txt', avr_com_mse1)
np.savetxt('./new_T2/avr_com2.txt', avr_com_mse2)
np.savetxt('./new_T2/avr_com3.txt', avr_com_mse3)
np.savetxt('./new_T2/avr_com4.txt', avr_com_mse4)
np.savetxt('./new_T2/avr_loc1.txt', avr_loc_mse1)
np.savetxt('./new_T2/avr_loc2.txt', avr_loc_mse2)
np.savetxt('./new_T2/avr_loc3.txt', avr_loc_mse3)
np.savetxt('./new_T2/avr_loc4.txt', avr_loc_mse4)
np.savetxt('./new_T2/avr_v1.txt', avr_v1)
np.savetxt('./new_T2/avr_v2.txt', avr_v2)
np.savetxt('./new_T2/avr_v3.txt', avr_v3)
np.savetxt('./new_T2/avr_v4.txt', avr_v4)