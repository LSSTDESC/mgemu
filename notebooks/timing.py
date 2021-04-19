from mgemu import emu, emu_fast
import time
import numpy as np


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

n_trials = 20

start_time = time.time()
for _ in range(n_trials):
    emu(Omh2=Omh2, ns=ns, s8=s8, fR0=fr0, n=n, z=z)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
for _ in range(n_trials):
    emu_fast(Omh2=Omh2, ns=ns, s8=s8, fR0=fr0, n=n, z=z)
print("--- %s seconds ---" % (time.time() - start_time))


print( 'difference: ' + str( np.max(np.abs( emu(Omh2=Omh2, ns=ns, s8=s8, fR0=fr0, n=n, z=z)[0] - emu_fast(Omh2=Omh2, ns=ns, s8=s8, fR0=fr0, n=n, z=z)[0] ))))
