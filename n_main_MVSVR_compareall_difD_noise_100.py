import numpy as np
import random
import ADMM_function
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


m=100
T=500
N=6
K=50
d_blocked=[0, 5, 10, 15, 20, 25, 30, 35, 40] 
# 先仿真16个看
num_a=2   #2,2,4;  5,2,4
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
num_compare=5
USTp=35
Umin=20

for d in d_blocked:
    if d == 0:
        c = 0.5
        gamma1 = 0.01
        gamma2 = 0.5
        gamma3 = 0.02
        rho = 0.5

        gamma_bl = 0.3
        c2 = 0.5
        gamma_new = 0.03
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.05
        gamma22 = 0.2
        gamma32 = 0.03
        alpha = 0.3
        mu = 0.2
        eta = 15

        cs = 0.01
        gammas = 0.2

    elif d == 5:
        c = 0.05
        gamma1 = 0.01
        gamma2 = 0.3
        gamma3 = 0.02
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.02
        gamma22 = 0.2
        gamma32 = 0.01
        alpha = 0.2
        mu = 0.15
        eta = 15

        cs = 0.01
        gammas = 0.2

    elif d == 10:
        c = 0.05
        gamma1 = 0.001
        gamma2 = 0.05
        gamma3 = 0.005
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.005
        gamma22 = 0.08
        gamma32 = 0.003
        alpha = 0.2
        eta = 20  # 15

        cs = 0.01
        gammas = 0.2

    elif d == 15:
        c = 0.05
        gamma1 = 0.001
        gamma2 = 0.05
        gamma3 = 0.008
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03  # 0.05
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.003
        gamma22 = 0.06
        gamma32 = 0.002
        alpha = 0.2
        eta = 20

        cs = 0.01
        gammas = 0.2

    elif d == 20:
        c = 0.05
        gamma1 = 0.0005
        gamma2 = 0.05
        gamma3 = 0.01
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.003
        gamma22 = 0.05
        gamma32 = 0.002
        alpha = 0.15
        eta = 20

        cs = 0.01
        gammas = 0.2

    elif d == 25:
        c = 0.05
        gamma1 = 0.001
        gamma2 = 0.05
        gamma3 = 0.008
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.003
        gamma22 = 0.05
        gamma32 = 0.002
        alpha = 0.15
        eta = 20

        cs = 0.01
        gammas = 0.2

    elif d == 30:
        c = 0.05
        gamma1 = 0.001
        gamma2 = 0.06
        gamma3 = 0.008
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.003
        gamma22 = 0.05
        gamma32 = 0.002
        alpha = 0.15
        eta = 20

        cs = 0.01
        gammas = 0.2

    elif d == 35:
        c = 0.05
        gamma1 = 0.05
        gamma2 = 0.45
        gamma3 = 0.004
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.03
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.003
        gamma22 = 0.04
        gamma32 = 0.002
        alpha = 0.15
        eta = 25

        cs = 0.01
        gammas = 0.2

    elif d == 40:
        c = 0.05
        gamma1 = 0.06
        gamma2 = 0.75
        gamma3 = 0.005
        rho = 0.05

        gamma_bl = 0.3
        c2 = 1.0
        gamma_new = 0.04
        c22 = 0.5
        rho2 = 0.5
        gamma12 = 0.003
        gamma22 = 0.04
        gamma32 = 0.002
        alpha = 0.15
        eta = 25

        cs = 0.01
        gammas = 0.05

    avr_com_mse = np.zeros((num_compare, repeat_times))
    avr_loc_mse = np.zeros((num_compare, repeat_times))
    maxiter = np.zeros((num_compare, repeat_times))

    # import data
    for t1 in range(num_a):
        a_stack = np.loadtxt(
            './new_d_compareall_noise/Data_Sample/data_A_%d' % m + '_%d.txt' % t1)  # './new_d_compareall_noise/Data_Sample/
        for t2 in range(num_x):
            X_est = np.loadtxt('./new_d_compareall_noise/Data_Sample/data_X_%d.txt' % t2)
            for t3 in range(num_v):
                print(d, m, t1, t2, t3)
                v_stack = np.loadtxt(
                    './new_d_compareall_noise/Data_Sample/d%d' % d + '/data_V_%d' % t2 + '_%d.txt' % t3)
                y_stack = np.zeros(m * N)
                for i in range(N):
                    x_o = np.multiply(v_stack[i * T: (i + 1) * T], X_est)
                    y_stack[i * m: (i + 1) * m] = np.dot(a_stack[i * m: (i + 1) * m, :], x_o)
                    SNR = 12
                    s = 10 ** (-1.0 * (SNR / 10.0)) / T * (np.linalg.norm(x_o, ord=2) ** 2)
                    noise_add = np.random.randn(m) * np.sqrt(s)
                    y_stack[i * m: (i + 1) * m] = y_stack[i * m: (i + 1) * m] + noise_add
                    # y_stack=y_stack+noise_add

                print("the Initial BSL:\n")
                # compute the base line
                avr_com_mse[0, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[
                    0, t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.baseline_Cint(a_stack, y_stack, X_est, v_stack, N, gamma_bl)

                # compute the l1-soft
                print("the ADMM-VPD:\n")
                avr_com_mse[1, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[1,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.decentral_l1_VR_penalty_c_ind_hard(a_stack, y_stack, X_est, v_stack, N,
                                                                       Adjacent_Matrix, alpha, c2, gamma_new, Max_iter,
                                                                       eta, USTp, Umin)

                print("the JSM1:\n")
                avr_com_mse[2, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[2,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.baseline_JSM1(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, c, rho,
                                                  gamma1, gamma2, gamma3, Max_iter)

                # compute the combination function
                print("VPD-JSM1 Algorithm")
                avr_com_mse[3, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[3,
                                                                                  t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.ADMM_VPD_JSM1_ind(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, alpha,
                                                      c22, rho2, gamma12, gamma22, gamma32,
                                                      c2, gamma_new, Max_iter, eta, USTp, Umin)

                print("the standard algorithm")
                avr_com_mse[4, t1 * num_x * num_v + t2 * num_v + t3], avr_loc_mse[4,
                                                                         t1 * num_x * num_v + t2 * num_v + t3] \
                    = ADMM_function.stand_DLasso(a_stack, y_stack, X_est, v_stack, N, Adjacent_Matrix, cs, gammas,
                                                 Max_iter)

                for j in range(num_compare):
                    np.savetxt('./new_d_compareall_noise/glo/convergence2/d%d/' % d + 'avr_com_mse%d.txt' % j,
                                   avr_com_mse[j, :])
                    np.savetxt('./new_d_compareall_noise/glo/convergence2/d%d/' % d + 'avr_loc_mse%d.txt' % j,
                                   avr_loc_mse[j, :])


