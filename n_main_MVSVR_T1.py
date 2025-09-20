import numpy as np
import random
import ADMM_function_u as AD
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

B_one = [5, 15, 20, 25]
M = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360] # [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
T = 500
N = 6
K = 50
d = 10  # 5, 15, 25,   15和25的20,40，60还没跑
num_a = 2  # 2,2,4; 5,2,4
num_x = 2
num_v = 4
repeat_times = num_a * num_x * num_v

Adjacent_Matrix = np.matrix([[0, 0, 1, 1, 1, 1]
                            , [0, 0, 1, 1, 1, 0]
                            , [1, 1, 0, 1, 0, 0]
                            , [1, 1, 1, 0, 1, 1]
                            , [1, 1, 0, 1, 0, 1]
                            , [1, 0, 0, 1, 1, 0]])

Max_iter = 100
num_compare = 1

# fix the parameter of JSM1
c = 0.05
gamma1 = 0.007
gamma2 = 0.2
gamma3 = 0.01
rho = 0.05
USTp=35
Umin=20

cs = 0.01
gammas = 0.03

gamma_new_bl = 0.3
c2 = 1.0
gamma_new = 0.05
c22 = 0.5
rho2 = 0.5
gamma12 = 0.007
gamma22 = 0.1
gamma32 = 0.005
alpha = 0.2
eta = 20

# import data
for b in B_one:
    avr_com_mse = np.zeros((len(M), repeat_times))
    avr_loc_mse = np.zeros((len(M), repeat_times))
    for mi in range(len(M)):
        m = M[mi]
        for t1 in range(num_a):
            M_list = [m, 100, 100, 100, 100, 100]
            a_stack = np.loadtxt('./new_T1/Data_Sample/M/data_A_%d'% m +'_%d.txt' % t1)
            assert np.size(a_stack, 0) == sum(M_list)
            for t2 in range(num_x):
                X_est = np.loadtxt('./new_T1/Data_Sample/data_X_%d.txt' % t2)
                for t3 in range(num_v):
                    print(t1,t2,t3)
                    v_stack = np.loadtxt('./new_T1/Data_Sample/%d' % b +'/data_V_%d' % t2 + '_%d.txt' % t3)
                    y_stack = np.zeros(sum(M_list))
                    for i in range(N):
                        x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
                        y_stack[sum(M_list[0:i]):sum(M_list[0:i+1])] = \
                            np.dot(a_stack[int(sum(M_list[0:i])):int(sum(M_list[0:i+1])), :], x_o)
                        SNR = 12
                        s = 10 ** (-1.0 * (SNR / 10.0)) / T * (np.linalg.norm(x_o, ord=2) ** 2)
                        noise_add = np.random.randn(int(M_list[i])) * np.sqrt(s)
                        y_stack[sum(M_list[0:i]):sum(M_list[0:i+1])] = y_stack[sum(M_list[0:i]):sum(M_list[0:i+1])] + noise_add

                    # compute the l1-soft
                    print("the ADMM-VPD:\n")
                    avr_com_mse[mi, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[mi, t1 * num_x * num_v + t2 * num_v + t3] \
                        = AD.decentral_l1_VR_penalty_c_ind_hard(a_stack, y_stack, X_est, v_stack, N, M_list,
                                                            Adjacent_Matrix, alpha, c2, gamma_new,
                                                            Max_iter, eta, USTp, Umin)
                    
                    print("the Initial BSL:\n")
                    # compute the base line
                    avr_com_mse[0, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[0, t1 * num_x * num_v + t2 * num_v + t3]\
                                = AD.baseline_Cint(a_stack, y_stack, X_est, v_stack, N, M_list, gamma_new_bl)

                    # Baseline2 _JSM1
                    print("the JSM1 algorithm")
                    avr_com_mse[2, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[2, t1 * num_x * num_v + t2 * num_v + t3]\
                                = AD.baseline_JSM1(a_stack, y_stack, X_est, v_stack, N, M_list, Adjacent_Matrix, c, rho,
                                                              gamma1, gamma2, gamma3, Max_iter)

                    # compute the combination function
                    print("VPD-JSM1 Algorithm")
                    avr_com_mse[3, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[
                        3, t1 * num_x * num_v + t2 * num_v + t3] \
                        = AD.ADMM_VPD_JSM1_ind(a_stack, y_stack, X_est, v_stack, N, M_list, Adjacent_Matrix, alpha,
                                               c22, rho2, gamma12, gamma22, gamma32,
                                               c2, gamma_new, Max_iter, eta, USTp, Umin)

                    # print("a")
                    print("the standard algorithm")
                    avr_com_mse[4, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[4, t1 * num_x * num_v + t2 * num_v + t3] \
                                = AD.stand_DLasso(a_stack, y_stack, X_est, v_stack, N, M_list, Adjacent_Matrix, cs, gammas, Max_iter)

                    for j in range(num_compare):
                      np.savetxt('./new_T1/convergence/%d/' %b + 'avr_com_mse%d.txt' % mi,
                                      avr_com_mse[mi, :])
                      np.savetxt('./new_T1/convergence/%d/' %b + 'avr_loc_mse%d.txt' % mi,
                                      avr_loc_mse[mi, :])


