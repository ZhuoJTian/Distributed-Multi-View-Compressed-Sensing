import numpy as np
import matplotlib.pyplot as plt



avr_com_mse2=np.loadtxt('./ADMMOGD with different alpha(c=0.2g=0.05)/avr_com_mse.txt')
avr_loc_mse2=np.loadtxt('./ADMMOGD with different alpha(c=0.2g=0.05)/avr_loc_mse.txt')
avr_loss2=np.loadtxt('./ADMMOGD with different alpha(c=0.2g=0.05)/avr_loss.txt')
mse_total2=np.loadtxt('./ADMMOGD with different alpha(c=0.2g=0.05)/mse_total.txt')
cserr2=np.loadtxt('./ADMMOGD with different alpha(c=0.2g=0.05)/cserr.txt')

T=100
maxiter=500
gap=10

avr_com_mse2=1.0*avr_com_mse2/T
cserr2 = 1.0*cserr2/T
avr_loc_mse2 = 1.0*avr_loc_mse2/T
avr_loss2 = 1.0*avr_loss2/T
mse_total2 = 1

np.savetxt('./ADMMPD with different alpha(c=0.4g=0.1)/cserr.txt', cserr1)
np.savetxt('./ADMMPD with different alpha(c=0.4g=0.05)/cserr.txt', cserr2)
np.savetxt('./ADMMPD with different alpha(c=0.3g=0.05)/cserr.txt', cserr3)
np.savetxt('./ADMMPD with different alpha(c=0.2g=0.05)/cserr.txt', cserr4)
np.savetxt('./ADMMOGD with different alpha(c=0.2g=0.05)/cserr.txt', cserr5)