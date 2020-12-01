import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
from matplotlib import gridspec
from matplotlib import cm
from scipy.integrate import quad, dblquad
from scipy.special import gamma
import os
import subprocess
from subprocess import Popen, PIPE
from scipy.integrate import odeint

#Define smoothing filter, in case we need to use it.
import scipy.signal
def SmoothPk(Pk_in):
    Pk_out = scipy.signal.savgol_filter(Pk_in, 51, 3)
    return Pk_out

#Define a-dependent functions for growth factor equation
def Omegatime(a, Om0):
 return Om0/(Om0+(1-Om0)*a*a*a)
#adot
def adot(a, Om0):
 return np.sqrt(Om0/a+(1-Om0)*a*a)
def H(a, Om0):
 return adot(a, Om0)/a
def adotprime(a, Om0):
 return (-Om0/a/a + 2*(1-Om0)*a)/np.sqrt(Om0/a+(1-Om0)*a*a)/2
#Define additional functions needed for
def mfr(a, Om0, fr0, nfr): #scalar field mass
 return (1/2997.72)*math.sqrt(1./(nfr+1)/fr0)*math.sqrt(math.pow(Om0+4*(1-Om0),-nfr-1))*math.sqrt(math.pow(Om0/a/a/a+4*(1-Om0),2+nfr))
def geff(a, Om0, fr0, nfr, k): #g_eff
 return k*k/(k*k+a*a*mfr(a, Om0, fr0, nfr)*mfr(a, Om0, fr0, nfr))/3
#Define 1/adot^3 integrand
def invadot3(a, Om0):
 return 1/adot(a, Om0)/adot(a, Om0)/adot(a, Om0)
#Define 1/adot^3 integral
def intToday(Om0, amin):
 return quad(invadot3,0,amin,args=(Om0))[0]
def Damin(Om0, amin):
 return 2.5*Om0*H(amin, Om0)*intToday(Om0, amin)
def DH1(a, Om0):
 return -3*Om0/(2*a*a*a*math.sqrt((a*a*a+Om0-Om0*a*a*a)/a))
def der0(Om0, amin):
 return 2.5*Om0*(DH1(amin, Om0)*intToday(Om0, amin)+H(amin, Om0)*invadot3(amin, Om0))
 
#Setup integration to get comoving volume for various z bin
#Comoving distance integrator
def integrand(a, Omm):
 return 1/math.sqrt(a*Omm+a*a*a*a*(1-Omm))
#get comoving volume
def Vcom(z, Omm):
 return (4*math.pi*20000)/(3*41252.96)*math.pow(2997.92458*quad(integrand, 1./(1+z), 1, args=(Omm))[0],3)#*math.pow(10,-9)
#get comoving differential volume in redshift bin with width +-0.05
def Vbin(z, Omm, bin):
 return Vcom(z+bin, Omm)-Vcom(z-bin, Omm)
Veff=np.vectorize(Vbin)

#Define function fro differential dN/dz to be integrated
def dNdz(z, alpha, z0, ntot):
 return (ntot*alpha/(z0*gamma(3.0/alpha)))*(z/z0)*(z/z0)*math.exp(-math.pow(z/z0,alpha))
def Nz(z, alpha, z0, ntot, bin):
 return quad(dNdz, z-bin, z+bin, args=(alpha, z0, ntot))[0]
#Nz = np.vectorize(Nz)
#for year10 sample
NY10=48*20000*3600
ay10= 0.90
z0y10 = 0.28
biny10 = 0.05
b1y10 = 0.95
#for year1 sample
NY1=18*20000*3600
ay1= 0.94
z0y1 = 0.26
biny1 = 0.1
b1y1 = 1.05

#Fiducial cosmology parameters
h=0.67 #+ 0.01
#och=0.1194 #+ 0.003
#ocb=0.022 #+ 2*0.0008
#ovh=0.00064
#As=math.exp(3.09104)/math.pow(10,10) #+ 5*math.pow(10,-10)
ns=0.967 #+ 0.025
omh = 0.142
#omLh = h*h - omh
#omL = omLh/h/h

Om = omh/h/h
omL = 1.0-Om
#Fiducial MG cosmology
fr0=math.pow(10.,-5.)
nfr=1.

#print (Vbin(0.13, Om, 0.1))
#print (Veff(0.13, Om, 0.1))

#print ('Shotnoise=', Veff(1.2, Om, biny10)/Nz(1.2, ay10, z0y10, NY10, biny10))


abserr = 1.0e-13
relerr = 1.0e-13
#Define growth factor differential equation system
def growth(y, a, Om0):
 D, w = y
 dyda = [w, -(adotprime(a, Om0) + 2*H(a, Om0))*w/adot(a, Om0) + 1.5*Omegatime(a, Om0)*H(a, Om0)*H(a, Om0)*D/adot(a, Om0)/adot(a, Om0)]
 return dyda
 
#y0 = [0.0019999999,0.999999999]
arange = np.logspace(math.log(0.002,10),math.log(1,10),1000)
#setting up boundary conditions
y0 = [Damin(Om, arange[0]),der0(Om, arange[0])]
sol = odeint(growth, y0, arange, args=(Om,), atol=abserr, rtol=relerr)
#Spline GR growth factor  solution and growth rate.
Dgr = interp1d(arange, sol[:,0], kind='cubic')
#and also growth rate
fgr = interp1d(arange, arange*sol[:,1]/sol[:,0], kind='cubic')


Nd=55
#Nd=36
colors= cm.rainbow(np.linspace(0,0.8, Nd))
Nsteps = 100
#Nt = 80


#random_seeds = np.loadtxt('/home/astrosun2/gvalogiannis/2LPTic_serial/random_2000.txt')
#random_seeds=random_seeds[1000:2000]
#print (random_seeds.shape)

timesteps = np.loadtxt('timestepsCOLA.txt')
irangey10 = np.array([43,45,48,53,57,61,65,70,75,82])
irangey1 = np.array([43,48,57,65,75])
for i in (irangey10):
 print (i, timesteps[i,1])
Nt = irangey10[irangey10.shape[0]-1]

Nseed=1999 #25
#for i in range(Nsteps):
#for i in range(Nt,Nt+1):
#for i in (irangey10):
for i in (irangey1):
 astep = timesteps[i,0]
 zstep = timesteps[i,1]
 #linear bias for galaxy sample, in our case Y10
 #b1=b1y10*Dgr(1)/Dgr(astep) #Bias for Y10
 b1=b1y1*Dgr(1)/Dgr(astep) #Bias for Y10
 print ('a,z,b_1=',astep, zstep, b1)
 for j in range(Nseed):
  #if j==1192 or j==1196 or j==1198:
  # continue
  #pklin= np.loadtxt('/home/astrosun2/gvalogiannis/CAMB/test_plindes_'+str(j)+'.txt')locals()["pkLdat_"+str(j)+"_"+str(i)+]
  locals()['pkLseed_'+str(j)+'_'+str(i)] = np.loadtxt('/home/astrosun2/gvalogiannis/colacode/tassev/Pemulator/PLCDMseed2k_'+str(j)+'_'+str(i)+'.txt')
  locals()['pkMGseed_'+str(j)+'_'+str(i)]= np.loadtxt('/home/astrosun2/gvalogiannis/colacode/tassev/Pemulator/PMGseed2k_'+str(j)+'_'+str(i)+'.txt')
  #Import GR and LCDM power spectra for matter
  locals()['pkLCDMseed_'+str(j)+'_'+str(i)]= locals()['pkLseed_'+str(j)+'_'+str(i)][:,1]
  locals()['pkMGseed'+str(j)+'_'+str(i)]= locals()['pkMGseed_'+str(j)+'_'+str(i)][:,1]
  kvec = locals()['pkMGseed_'+str(j)+'_'+str(i)][:,0]
  #multiply by bias squared to get galaxy Pks
  locals()['pkLCDMseed_'+str(j)+'_'+str(i)] *= b1*b1
  locals()['pkMGseed'+str(j)+'_'+str(i)] *= b1*b1
  #And then add shot noise contributions
  #locals()['pkMGseed'+str(j)+'_'+str(i)] += Veff(zstep, Om, biny10)/Nz(zstep, ay10, z0y10, NY10, biny10)
  #locals()['pkLCDMseed_'+str(j)+'_'+str(i)] += Veff(zstep, Om, biny10)/Nz(zstep, ay10, z0y10, NY10, biny10)
  locals()['pkMGseed'+str(j)+'_'+str(i)] += Veff(zstep, Om, biny1)/Nz(zstep, ay1, z0y1, NY1, biny1)
  locals()['pkLCDMseed_'+str(j)+'_'+str(i)] += Veff(zstep, Om, biny1)/Nz(zstep, ay1, z0y1, NY1, biny1)


 #locals()['ratioMGtoGRseed_'+str(i)]= locals()['ratioMGseed0_'+str(i)]
 locals()['pkMGcov_'+str(i)]= locals()['pkMGseed0_'+str(i)]
 #locals()['PkGR_'+str(i)]= locals()['pkLCDMseed_0_'+str(i)]
 for jj in range(1,Nseed):
  #locals()['PkGR_'+str(i)]=np.vstack((locals()['PkGR_'+str(i)],locals()['pkLCDMseed_'+str(jj)+'_'+str(i)]))
  locals()['pkMGcov_'+str(i)]=np.vstack((locals()['pkMGcov_'+str(i)],locals()['pkMGseed'+str(jj)+'_'+str(i)]))
  
 locals()['covPMGmat_'+str(i)]=np.cov(np.transpose(locals()['pkMGcov_'+str(i)]))
 #np.savetxt('covariance_step_'+str(i)+'.txt', locals()['covPMGmat_'+str(i)], fmt = '%1.8f')
 np.savetxt('covarianceY1_step_'+str(i)+'.txt', locals()['covPMGmat_'+str(i)], fmt = '%1.8f')


#ratiomean = np.mean(ratioMGtoGRseed_99, axis=0)
#ratiostd=np.std(ratioMGtoGRseed_99, axis=0, ddof=1)
#covmat=np.cov(np.transpose(ratioMGtoGRseed_99))
#corrmat=np.corrcoef(np.transpose(ratioMGtoGRseed_99))
covPMGmat=np.cov(np.transpose(locals()['pkMGcov_'+str(Nt)]))
corrPMGmat=np.corrcoef(np.transpose(locals()['pkMGcov_'+str(Nt)]))
#PkLmean = np.mean(PkGR_99, axis=0)
#Pkerr = np.std(PkGR_99, axis=0, ddof=1)
PkMGmean = np.mean(locals()['pkMGcov_'+str(Nt)], axis=0)
PkMGerr = np.std(locals()['pkMGcov_'+str(Nt)], axis=0, ddof=1)
#np.savetxt('covariance_99_2000s.txt', covmat, fmt = '%1.8f')
#np.savetxt('ratioavg_99_2000s.txt', (np.vstack((pkMGseed_0_97[:,0],ratiomean))).T, fmt = '%1.8f')
print (covPMGmat.shape)


def applyPlotStyle14():
 plt.tick_params(axis='both',which='major',length=5, left='on', right='on', width=1, direction='inout')
 plt.tick_params(axis='both',which='minor',length=3.2,left='on', right='on', width=1, direction='inout')
 plt.tick_params(which='both',width=1.3)
 plt.grid(True)
 plt.xscale("log")
 plt.yscale("log")
 plt.xlim(kvec[0], kvec[kvec.shape[0]-1])
 plt.ylim(kvec[0], kvec[kvec.shape[0]-1])
 #plt.ylabel(r"$\frac{P_{MG}}{P_{\Lambda CDM}}$",fontsize=18)
 plt.xlabel(r"$k (h/Mpc)$",fontsize=18)
 plt.ylabel(r"$k (h/Mpc)$",fontsize=18)
 #plt.ylabel(r"$r (Mpc/h)$",fontsize=18)
 plt.legend(loc="best", frameon=False, numpoints=1, prop={'size':12})
#colors= cm.rainbow(np.linspace(0,1, len(delenvi)))
#gs11=gridspec.GridSpec(3,2, wspace=0.0, hspace=0.0, height_ratios=[2,2,2])
#fig11=plt.figure(11,figsize=(0.95*8,0.95*11.))
#gs11=gridspec.GridSpec(1,2, wspace=0.0, hspace=0.0)#, height_ratios=[1,2])
#fig11=plt.figure(11,figsize=(0.85*12,0.4*12))

fig11=plt.figure('test')
ax=fig11.add_subplot(111)
applyPlotStyle14()
#ax2=fig11.add_subplot(gs11[0,1])
#applyPlotStyle14()
#ax3=fig11.add_subplot(gs11[1,0])
#applyPlotStyle13()
#ax4=fig11.add_subplot(gs11[1,1])
#applyPlotStyle14()
#rtry = np.linspace(20,180,20)
#rtry = rvec
ax.set_title('Galaxy-Galaxy covariance')
#pltcov=ax.pcolormesh(kvec,kvec,covPMGmat, cmap='plasma')
pltcov=ax.pcolormesh(kvec,kvec,corrPMGmat, cmap='plasma')
#pltcov=ax.pcolormesh(rtry,rtry,corrCohn, cmap='plasma')
fig11.colorbar(pltcov, ax=ax)
plt.tight_layout()
#plt.savefig('./cov_z02.png')


def applyPlotStyle2():
 plt.tick_params(axis='both',which='major',length=5, left='on', right='on', width=1, direction='inout')
 plt.tick_params(axis='both',which='minor',length=3.2,left='on', right='on', width=1, direction='inout')
 plt.tick_params(which='both',width=1.3)
 plt.grid(True)
 plt.xscale("log")
 plt.yscale("log")
 plt.xlim(kvec[0], kvec[kvec.shape[0]-1])
 #plt.ylim(kvec[0], kvec[kvec.shape[0]-1])
 #plt.ylabel(r"$\frac{P_{MG}}{P_{\Lambda CDM}}$",fontsize=18)
 plt.xlabel(r"$k (h/Mpc)$",fontsize=18)
 plt.ylabel(r"$P_{gg}(k)$",fontsize=18)
 plt.legend(loc="best", frameon=False, numpoints=1, prop={'size':12})

fig2=plt.figure('Pk')
ax=fig2.add_subplot(111)
applyPlotStyle2()
ax.errorbar(kvec,PkMGmean, yerr=0*PkMGerr, color='b', marker='o', markersize=2, linestyle=None)
ax.fill_between(kvec,PkMGmean-PkMGerr, PkMGmean+PkMGerr, color='b', alpha=0.2)
plt.tight_layout()
#plt.savefig('./pk_z02.png')

plt.show()
