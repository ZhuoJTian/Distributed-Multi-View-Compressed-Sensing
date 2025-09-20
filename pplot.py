import numpy as np
import matplotlib.pyplot as plt

avr_com_mse0=np.loadtxt('./Compare_all_convergence(20average)/convergence1/avr_com_mse0.txt')
avr_loc_mse0=np.loadtxt('./Compare_all_convergence(20average)/convergence1/avr_loc_mse0.txt')
mse_total0=np.loadtxt('./Compare_all_convergence(20average)/convergence1/mse_total0.txt')
cserr0=np.loadtxt('./Compare_all_convergence(20average)/convergence1/cserr0.txt')

avr_com_mse1=np.loadtxt('./Compare_all_convergence(20average)/convergence1/avr_com_mse1.txt')
avr_loc_mse1=np.loadtxt('./Compare_all_convergence(20average)/convergence1/avr_loc_mse1.txt')
mse_total1=np.loadtxt('./Compare_all_convergence(20average)/convergence1/mse_total1.txt')
cserr1=np.loadtxt('./Compare_all_convergence(20average)/convergence1/cserr1.txt')

avr_com_mse2=np.loadtxt('./Compare_all_convergence(20average)/convergence1/avr_com_mse2.txt')
avr_loc_mse2=np.loadtxt('./Compare_all_convergence(20average)/convergence1/avr_loc_mse2.txt')
mse_total2=np.loadtxt('./Compare_all_convergence(20average)/convergence1/mse_total2.txt')
cserr2=np.loadtxt('./Compare_all_convergence(20average)/convergence1/cserr2.txt')

avr_com_mse11=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/avr_com_mse1.txt')
avr_loc_mse11=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/avr_loc_mse1.txt')
mse_total11=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/mse_total1.txt')
cserr11=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/cserr1.txt')

avr_com_mse21=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/avr_com_mse2.txt')
avr_loc_mse21=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/avr_loc_mse2.txt')
mse_total21=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/mse_total2.txt')
cserr21=np.loadtxt('./Compare_all_convergence(20average)/convergence1_1/cserr2.txt')



'''
avr_com_mse3=np.loadtxt('./Compare_all(20average)/convergence2/avr_com_mse3.txt')
avr_loc_mse3=np.loadtxt('./Compare_all(20average)/convergence2/avr_loc_mse3.txt')
mse_total3=np.loadtxt('./Compare_all(20average)/convergence2/mse_total3.txt')
cserr3=np.loadtxt('./Compare_all(20average)/convergence2/cserr3.txt')
'''
maxiter=100
gap=1
num=int(np.ceil(maxiter/gap))

X=np.linspace(1, maxiter, num)

figure1=plt.figure(1)
ax1=plt.subplot(121)
plt.plot(X, avr_com_mse0[0, :], linewidth=2, label="BL", color='y', linestyle='-')
plt.plot(X, np.average(avr_com_mse1, axis=0), linewidth=2, label="ADMMV", color='b', linestyle='-.')
plt.plot(X, np.average(avr_com_mse2, axis=0), linewidth=2, label="ADMMPD(alpha=0.01)", color='r', linestyle='--')
plt.plot(X, np.average(avr_com_mse11, axis=0), linewidth=2, label="ADMMV1", color='g', linestyle='-.')
plt.plot(X, np.average(avr_com_mse21, axis=0), linewidth=2, label="ADMMPD1(alpha=0.01)", color='r', linestyle='-.')
# plt.plot(X, avr_com_mse_s[3, :], linewidth=2, label="ADMMOGD(beta=1.6)", color='g', linestyle=':')
plt.title("avr_com_mse")
plt.xlabel("Iteration")
plt.ylabel("avr_com_mse")
plt.xlim(1, maxiter)
plt.legend()


ax2=plt.subplot(122)
plt.plot(X, avr_loc_mse0[0, :], linewidth=2, label="BL", color='y', linestyle='-')
plt.plot(X, np.average(avr_loc_mse1, axis=0), linewidth=2, label="ADMMV", color='b', linestyle='-.')
plt.plot(X, np.average(avr_loc_mse2, axis=0), linewidth=2, label="ADMMPD(alpha=0.01)", color='r', linestyle='--')
plt.plot(X, np.average(avr_loc_mse11, axis=0), linewidth=2, label="ADMMV1", color='g', linestyle='-.')
plt.plot(X, np.average(avr_loc_mse21, axis=0), linewidth=2, label="ADMMPD1(alpha=0.01)", color='r', linestyle='-.')
# plt.plot(X, avr_loc_mse_s[3, :], linewidth=2, label="ADMMOGD(beta=1.6)", color='g', linestyle=':')
plt.title("avr_loc_mse")
plt.xlabel("Iteration")
plt.ylabel("avr_loc_mse")
plt.xlim(1, maxiter)
plt.legend()

figure3=plt.figure(2)
ax3=plt.subplot(121)
plt.plot(X, mse_total0[0, :], linewidth=2, label="BL", color='y', linestyle='-')
plt.plot(X, np.average(mse_total1, axis=0), linewidth=2, label="ADMMV", color='b', linestyle='-.')
plt.plot(X, np.average(mse_total2, axis=0), linewidth=2, label="ADMMPD(alpha=0.01)", color='r', linestyle='--')
plt.plot(X, np.average(mse_total11, axis=0), linewidth=2, label="ADMMV1", color='g', linestyle='-.')
plt.plot(X, np.average(mse_total21, axis=0), linewidth=2, label="ADMMPD1(alpha=0.01)", color='r', linestyle='-.')
# plt.plot(X, mse_total_s[3, :], linewidth=2, label="ADMMOGD(beta=1.6)", color='g', linestyle=':')
plt.title("mse_total")
plt.xlabel("Iteration")
plt.ylabel("mse_total")
plt.xlim(1, maxiter)
plt.legend()

ax4=plt.subplot(122)
plt.plot(X, cserr0[0, :], linewidth=2, label="BL", color='y', linestyle='-')
plt.plot(X, np.average(cserr1, axis=0), linewidth=2, label="ADMMV", color='b', linestyle='-.')
plt.plot(X, np.average(cserr2, axis=0), linewidth=2, label="ADMMPD(alpha=0.01)", color='r', linestyle='--')
plt.plot(X, np.average(cserr11, axis=0), linewidth=2, label="ADMMV1", color='g', linestyle='-.')
plt.plot(X, np.average(cserr21, axis=0), linewidth=2, label="ADMMPD1(alpha=0.01)", color='r', linestyle='-.')
# plt.plot(X, cserr_s[3, :], linewidth=2, label="ADMMOGD(beta=1.6)", color='g', linestyle=':')
plt.title("cserr")
plt.xlabel("Iteration")
plt.ylabel("cserr")
plt.xlim(1, maxiter)
plt.legend()
plt.show()