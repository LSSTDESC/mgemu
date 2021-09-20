from mgemu import emu
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

### LCDM parameters
h=0.67 # See README and the accompanying paper regarding the value of h.
Omh2=(h**2)*0.281
ns=0.971
s8=0.82
### Hu-Sawicki model parameters
fr0=1e-5
n=1
### Redshift
z=0.3
pkratio, k = emu(Omh2=Omh2, ns=ns, s8=s8, fR0=fr0, n=n, z=z)



plt.figure(1, figsize=(9, 6) )
fR0_arr= np.logspace(-6, -4, 10)
for i in range(10):
    fR0 = fR0_arr[i]
    pkratio, k = emu(Omh2=Omh2, ns=ns, s8=s8, fR0=fR0, n=n, z=z)
    plt.plot(k, pkratio)
plt.xscale('log')
plt.ylabel(r'$P_MG(k)/P_{LCDM}(k)$', fontsize=23)
plt.xlabel(r'$k$', fontsize=23)
# plt.xlim(0, 1.01)
plt.show()
