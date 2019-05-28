#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

#hz as given by flat-LCDM model
def hz(x, a, b, c, d):
    return b*np.sqrt(a*(1.+x)**3. + (1.-a)*(1.+x)**(3.*(1.+c+d))*np.exp(-3.*d*x/(1.+x)))

#fitting function for sighz
def sighz_fit(x, a, b, c):
    return a + b*x + c*x**2.# + d*x**3.

#sighz for SKA-IM (z<0.5 for band 2, z>0.5 for band 1)    
#def sighz(x):
    #if x <= 0.5:
        #return 0.032677499870005927 - 0.1182449987366246*x + 0.18874999752200228*x**2.
    #if x > 0.5:
        #return 0.01057407547541934 - 0.0099023519792246981*x +  0.0051646657148614241*x**2.

#sighz for euclid        
#def sighz(x):
    #return 0.00815734690799534 - 0.008442219335642114*x + 0.0036065050823616235*x**2.
    
#sighz for SKA-GS
#def sighz(x):
    #return 0.12939999987243384 - 0.5447714275005697*x + 0.734285712524884*x**2.

#sighz for DESI    
#def sighz(x):
    #return 0.08250239666491332 - 0.1476293690019363*x + 0.0746253739811405*x**2.
    
#sighz for Meerkat L-band
def sighz(x):
    return 0.10073969407222942 -0.3831244911430175*x + 0.5455102060641105*x**2.

#input fiducial Cosmology (P18 best-fit for TT,TE,EE+lowE+lensing)
sigom = 0.0084
om = 0.3166# + sigom*np.random.randn()

w0 = -1.00
wa = +0.00

#sigh = 0.54 
#h = 67.36# + sigh*np.random.randn()

# h0 best-fit from Riess et al 2019 after LMC Cepheids inclusion
sigh = 1.42
h = 74.03# + sigh*np.random.randon()

# z-array
zmin=0.10
zmax=0.50
nz=5
z_arr=zmin+(zmax-zmin)*np.arange(nz)/(nz-1.0)

## reading sighz values 
#z,sighz = np.loadtxt('sighz_im_meerkat_Lband.dat', unpack='true')

## fitting sighz
#popt1, pcov1 = curve_fit(sighz_fit, z_arr, sighz)
#print(tuple(popt1))

#for i in range(nz):
    #print(sighz_fit(z_arr[i],*popt1),sighz[i])

# hz values according to the fiducial Cosmology and the given z values
hz_arr=np.array([hz(z,om,h,w0,wa) for z in z_arr])

sighz_arr = np.array([sighz(z)*hz(z,om,h,w0,wa) for z in z_arr])

hztype_arr=np.array([1 for z in z_arr])

##displaying results
for i in range(nz):
    print(z_arr[i],hz_arr[i],sighz_arr[i],hztype_arr[i])

### saving the simulated hz results in a text file
np.savetxt('hz_im_meerkat_bandL_'+str(nz)+'pts_r19.dat',np.transpose([z_arr, hz_arr, sighz_arr, hztype_arr]))