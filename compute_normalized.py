import numpy as np

M=80
T=500
N=6
K=50
d_blocked=20
num_a=1
num_x=1
num_v=20
repeat_times=num_a*num_x*num_v

for t1 in range(num_a):
    for t2 in range(num_x):
        for t3 in range(num_v):
            # import data
            a_stack = np.loadtxt('./Compare_all_convergence(20average)/Data_sample3/data_A'+'%d.txt' % t1)
            X_est = np.loadtxt('./Compare_all_convergence(20average)/Data_sample3/data_X' + '%d.txt' % t2)
            v_stack = np.loadtxt('./Compare_all_convergence(20average)/Data_sample3/data_V' + '%d.txt' % t3)

            avr_com_mse0 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/avr_com_mse0.txt')
            avr_loc_mse0 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/avr_loc_mse0.txt')
            mse_total0 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/mse_total0.txt')
            cserr0 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/cserr0.txt')

            avr_com_mse1 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/avr_com_mse1.txt')
            avr_loc_mse1 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/avr_loc_mse1.txt')
            mse_total1 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/mse_total1.txt')
            cserr1 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/cserr1.txt')

            avr_com_mse2 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/avr_com_mse2.txt')
            avr_loc_mse2 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/avr_loc_mse2.txt')
            mse_total2 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/mse_total2.txt')
            cserr2 = np.loadtxt('./Compare_all_convergence(20average)/convergence3/cserr2.txt')

            n_avr_com_mse1=avr_com_mse1*T/(np.linalg.norm(X_est, 2) ** 2)
            a=1
            b=3