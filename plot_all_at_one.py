import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
d = 10
Max_iter = 100
M = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

dd_blocked=[0, 5, 10, 15, 20, 25, 30, 35, 40]
P = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SNR_set=[0, 5, 10, 15, 20, 25]
N = [4, 6, 8, 10, 12, 14, 16]


aavr_c0 = np.zeros(len(M))
aavr_l0 = np.zeros(len(M))
aavr_c1 = np.zeros(len(M))
aavr_l1 = np.zeros(len(M))
aavr_c2 = np.zeros(len(M))
aavr_l2 = np.zeros(len(M))
aavr_c3 = np.zeros(len(M))
aavr_l3 = np.zeros(len(M))
aavr_c4 = np.zeros(len(M))
aavr_l4 = np.zeros(len(M))

for i in range(len(M)):
    m = M[i]
    avr_cc0 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse0.txt')
    # cserr00 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr0.txt')

    avr_cc1 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse1.txt')
    # cserr11 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr1.txt')

    avr_cc2 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse2.txt')
    # cserr22 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr2.txt')

    avr_cc3 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse3.txt')
    # cserr33 = np.loadtxt('./new_d_compareall_noise/convergence3/d%d' % d + '/%d_' % m + 'cserr3.txt')

    avr_cc4 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_com_mse4.txt')
    avr_ll4 = np.loadtxt('./new_m_compareall_noise/convergence/d%d' % d + '/%d_' % m + 'avr_loc_mse4.txt')

    aavr_c0[i] = np.average(avr_cc0[:, 0])  #:,
    aavr_l0[i] = np.average(avr_ll0[:, 0])
    # cserr0[j, i] = np.average(cserr00[:, 0])

    aavr_c1[i] = np.average(avr_cc1[:, Max_iter - 1])
    aavr_l1[i] = np.average(avr_ll1[:, Max_iter - 1])
    # cserr1[j, i] = np.average(cserr11[:,Max_iter - 1])

    aavr_c2[i] = np.average(avr_cc2[:, Max_iter - 1])
    aavr_l2[i] = np.average(avr_ll2[:, Max_iter - 1])
    # cserr2[j, i] = np.average(cserr22[:, Max_iter - 1])

    aavr_c3[i] = np.average(avr_cc3[:, Max_iter - 1])
    aavr_l3[i] = np.average(avr_ll3[:, Max_iter - 1])

    aavr_c4[i] = np.average(avr_cc4[:, Max_iter - 1])
    aavr_l4[i] = np.average(avr_ll4[:, Max_iter - 1])


bavr_c0=np.zeros((len(dd_blocked)))
bavr_l0=np.zeros((len(dd_blocked)))
bavr_c1=np.zeros((len(dd_blocked)))
bavr_l1=np.zeros((len(dd_blocked)))
bavr_c2=np.zeros((len(dd_blocked)))
bavr_l2=np.zeros((len(dd_blocked)))
bavr_c3=np.zeros((len(dd_blocked)))
bavr_l3=np.zeros((len(dd_blocked)))
bavr_c4=np.zeros((len(dd_blocked)))
bavr_l4=np.zeros((len(dd_blocked)))


for j in range(len(dd_blocked)):
    d=dd_blocked[j]

    avr_cc0 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse0.txt')

    avr_cc1 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse1.txt')

    avr_cc2 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse2.txt')

    avr_cc3 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse3.txt')

    avr_cc4 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_com_mse4.txt')
    avr_ll4 = np.loadtxt('./new_d_compareall_noise/glo/convergence2/d%d' % d + '/avr_loc_mse4.txt')

    bavr_c0[j] = np.average(avr_cc0) #:,
    bavr_l0[j] = np.average(avr_ll0)

    bavr_c1[j] = np.average(avr_cc1)
    bavr_l1[j] = np.average(avr_ll1)

    bavr_c2[j] = np.average(avr_cc2)
    bavr_l2[j] = np.average(avr_ll2)

    bavr_c3[j] = np.average(avr_cc3)
    bavr_l3[j] = np.average(avr_ll3)

    bavr_c4[j] = np.average(avr_cc4)
    bavr_l4[j] = np.average(avr_ll4)


cavr_c0 = np.zeros((len(SNR_set)))
cavr_l0 = np.zeros((len(SNR_set)))
cavr_c1 = np.zeros((len(SNR_set)))
cavr_l1 = np.zeros((len(SNR_set)))
cavr_c2 = np.zeros((len(SNR_set)))
cavr_l2 = np.zeros((len(SNR_set)))
cavr_c3 = np.zeros((len(SNR_set)))
cavr_l3 = np.zeros((len(SNR_set)))
cavr_c4 = np.zeros((len(SNR_set)))
cavr_l4 = np.zeros((len(SNR_set)))


for j in range(len(SNR_set)):
    SNR = SNR_set[j]
    avr_cc0 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse0.txt')

    avr_cc1 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse1.txt')

    avr_cc2 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse2.txt')

    avr_cc3 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse3.txt')

    avr_cc4 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_com_mse4.txt')
    avr_ll4 = np.loadtxt('./new_compare_SNR/convergence2/SNR%d' % SNR + '_avr_loc_mse4.txt')

    cavr_c0[j] = np.average(avr_cc0)  #:,
    cavr_l0[j] = np.average(avr_ll0)

    cavr_c1[j] = np.average(avr_cc1)
    cavr_l1[j] = np.average(avr_ll1)

    cavr_c2[j] = np.average(avr_cc2)
    cavr_l2[j] = np.average(avr_ll2)

    cavr_c3[j] = np.average(avr_cc3)
    cavr_l3[j] = np.average(avr_ll3)

    cavr_c4[j] = np.average(avr_cc4)
    cavr_l4[j] = np.average(avr_ll4)

davr_c0=np.zeros((len(N)))
davr_l0=np.zeros((len(N)))
davr_c1=np.zeros((len(N)))
davr_l1=np.zeros((len(N)))
davr_c2=np.zeros((len(N)))
davr_l2=np.zeros((len(N)))
davr_c3=np.zeros((len(N)))
davr_l3=np.zeros((len(N)))
davr_c4=np.zeros((len(N)))
davr_l4=np.zeros((len(N)))

for j in range(len(N)):
    n=N[j]

    avr_cc0 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse0.txt')
    avr_ll0 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse0.txt')

    avr_cc1 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse1.txt')
    avr_ll1 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse1.txt')

    avr_cc2 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse2.txt')
    avr_ll2 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse2.txt')

    avr_cc3 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse3.txt')
    avr_ll3 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse3.txt')

    avr_cc4 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_com_mse4.txt')
    avr_ll4 = np.loadtxt('./Node/convergence/d%d' % n + '_avr_loc_mse4.txt')


    davr_c0[j] = np.average(avr_cc0) #:,
    davr_l0[j] = np.average(avr_ll0)

    davr_c1[j] = np.average(avr_cc1)
    davr_l1[j] = np.average(avr_ll1)

    davr_c2[j] = np.average(avr_cc2)
    davr_l2[j] = np.average(avr_ll2)

    davr_c3[j] = np.average(avr_cc3)
    davr_l3[j] = np.average(avr_ll3)

    davr_c4[j] = np.average(avr_cc4)
    davr_l4[j] = np.average(avr_ll4)


font1={'family':'Times New Roman', 'size':20}
font2={'family':'Times New Roman', 'size':18}
plt.rcParams['figure.figsize'] = (18, 9)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


figure1, axs=plt.subplots(2,4)
X=M
ax=axs[0][0]
ax.plot(X, aavr_c0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, aavr_c4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, aavr_c2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, aavr_c1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, aavr_c3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(20)
ax.xaxis.set_major_locator(xLocator)
ax.set_ylabel("Glo_AMSE", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)

ax=axs[1][0]
ax.plot(X, aavr_l0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, aavr_l4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, aavr_l2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, aavr_l1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, aavr_l3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(20)
ax.xaxis.set_major_locator(xLocator)
ax.set_ylabel("Loc_AMSE", font1)
ax.set_xlabel("M", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)
ax.set_title("(a) AMSE under different M", font2, y=-0.28)

X=P
ax=axs[0][1]
ax.plot(X, bavr_c0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, bavr_c4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, bavr_c2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, bavr_c1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, bavr_c3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(0.2)
ax.xaxis.set_major_locator(xLocator)
# ax.set_ylabel("Glo_AMSE", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)

ax=axs[1][1]
ax.plot(X, bavr_l0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, bavr_l4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, bavr_l2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, bavr_l1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, bavr_l3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(0.2)
ax.xaxis.set_major_locator(xLocator)
# ax.set_ylabel("Loc_AMSE", font1)
ax.set_xlabel("p", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)
ax.set_title("(b) AMSE under different p", font2, y=-0.28)

X=SNR_set
ax=axs[0][2]
ax.plot(X, cavr_c0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, cavr_c4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, cavr_c2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, cavr_c1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, cavr_c3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(5)
ax.xaxis.set_major_locator(xLocator)
# ax.set_ylabel("Glo_AMSE", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)

ax=axs[1][2]
ax.plot(X, cavr_l0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, cavr_l4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, cavr_l2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, cavr_l1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, cavr_l3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(5)
ax.xaxis.set_major_locator(xLocator)
# ax.set_ylabel("Loc_AMSE", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.set_xlabel("SNR(dB)", font1)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)
ax.set_title("(c) AMSE under different SNR", font2, y=-0.28)

X=N
ax=axs[0][3]
ax.plot(X, davr_c0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, davr_c4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, davr_c2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, davr_c1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, davr_c3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(2)
ax.xaxis.set_major_locator(xLocator)
# ax.set_ylabel("Glo_AMSE", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)

ax=axs[1][3]
ax.plot(X, davr_l0, linewidth=1.5, label="IRAS", color='b', linestyle='-', marker='o', markersize=8)
ax.plot(X, davr_l4, linewidth=1.5, label="D-LASSO", color='gray', linestyle='--', marker='<', markersize=8)
ax.plot(X, davr_l2, linewidth=1.5, label="D-AJSM", color='r', linestyle=':', marker='>', markersize=8)
ax.plot(X, davr_l1, linewidth=1.5, label="VPD-ADMM", color='g', linestyle='-.', marker='*', markersize=8)
ax.plot(X, davr_l3, linewidth=1.5, label="VPD-EM", color='y', linestyle='--', marker='s', markersize=8, markerfacecolor='none')
# ax.xlim(min(M), max(M))
# plt.ylim(0, 0.12)
xLocator=MultipleLocator(2)
ax.xaxis.set_major_locator(xLocator)
# ax.set_ylabel("Loc_AMSE", font1)
ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax.yaxis.offsetText.set_fontsize(15)
ax.set_xlabel("N", font1)
ax.xaxis.grid(False, which='major')
ax.yaxis.grid(False)
ax.tick_params(labelsize=15)
ax.set_title("(d) AMSE under different N", font2, y=-0.28)


plt.legend(bbox_to_anchor = (0.4, 2.32) ,loc=7, ncol=5, prop=font2)

plt.tight_layout()
plt.subplots_adjust(left=0.08,
                    bottom=0.12,
                    right=0.97,
                    top=0.91,
                    wspace=0.1,
                    hspace=0.18)


plt.draw()
plt.show()