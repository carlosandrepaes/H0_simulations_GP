from gapp import gp, dgp, covariance
import pickle
from numpy import array,concatenate,loadtxt,savetxt,zeros
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
 
nz1 = 30
nz2 = 10

if __name__=="__main__":
    #load data
    #(Z,h,Sigma,hid) = loadtxt('hz_im_band1and2_'+str(nz1)+'+'+str(nz2)+'pts_p18.dat',unpack='True')
    (Z,h,Sigma,hid) = loadtxt('hz_im_band1_'+str(nz1)+'pts_p18.dat',unpack='True')

    ##Gaussian process
    #g = dgp.DGaussianProcess(Z,h,Sigma,cXstar=(0.0,2.0,200),theta=[0.5,0.5])
    g = gp.GaussianProcess(Z,h,Sigma,covfunction=covariance.SquaredExponential,cXstar=(0.0,3.0,200))
    #g = gp.GaussianProcess(Z,h,Sigma,covfunction=covariance.Matern92,cXstar=(0.0,3.0,200))
    (rec,theta) = g.gp()
    #(drec,theta) = g.dgp(thetatrain='False')
    #(d2rec,theta) = g.d2gp()
    
    #saving the reconstructed hz and its derivatives
    #savetxt('rech_hz_im_band1and2_'+str(nz1)+'+'+str(nz2)+'pts_mat92_p18.dat',rec)
    #savetxt('rech_hz_euclid_'+str(nz1)+'pts_p18.dat',rec)
    #savetxt("dh.txt",drec)
    #savetxt("d2h.txt",d2rec)
    
    #calculate covariances between h, h' and h'' at points Zstar.
    #fcov = g.f_covariances(fclist=[0,1,2])
    #f = open('hcovariances.dmp','wb')
    #pickle.dump(fcov,f)
    #f.close()
    
    print(rec[0,1], rec[0,2], rec[0,2]/rec[0,1])
    
    ########################################################################
    ## plotting the reconstructed hz curve
    ##latex rendering text fonts
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    ## Create figure size in inches
    #fig, ax = plt.subplots(figsize = (9., 7.))

    ## Define axes
    #ax.set_xlabel(r"$z$", fontsize=22)
    #ax.set_ylabel(r"$H(z)$", fontsize=22)
    #plt.xlim(0., 3.05)
    #plt.ylim(0., 400)
    #for t in ax.get_xticklabels(): t.set_fontsize(20)
    #for t in ax.get_yticklabels(): t.set_fontsize(20)
    
    #plt.errorbar(Z, h, yerr=Sigma*7., fmt='o', color='black')
    #ax.fill_between(rec[:,0], rec[:,1]+2.*rec[:,2], rec[:,1]-2.*rec[:,2], facecolor='#F08080', alpha=0.80, interpolate=True)
    #ax.fill_between(rec[:,0], rec[:,1]+3.*rec[:,2], rec[:,1]-3.*rec[:,2], facecolor='#F08080', alpha=0.50, interpolate=True)
    
    #plt.legend((r"$H(z)$ rec ($2\sigma$)", "$H(z)$ rec ($3\sigma$)", "$H(z)$ data"), fontsize='18', loc='upper left')
    #plt.show()

    ##saving the plot
    #fig.savefig('hz_im_band1_'+str(nz1)+'pts_p18_reconst.png')
    ##fig.savefig('hz_im_band1and2_'+str(nz1)+'+'+str(nz2)+'pts_p18_reconst.png')
    
    ########################################################################
    #now plotting the reconstructed h0 values with comparison to R19 and P18 
    ##latex rendering text fonts
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    ## Create figure size in inches
    #fig, ax = plt.subplots(figsize = (9., 7.))

    ## Define axes
    #ax.set_xlabel(r"$H_0 \; (\mathrm{km} \; \mathrm{s}^{-1} \; \mathrm{Mpc}^{-1}$)", fontsize=22)
    ##ax.set_ylabel(r"measurement", fontsize=22)
    #plt.ylim(0.5,8.0)
    #plt.xlim(50.,80.)
    #for t in ax.get_xticklabels(): t.set_fontsize(20)
    #for t in ax.get_yticklabels(): t.set_fontsize(20)

    ## plotting
    #plt.axvspan(67.36-0.54, 67.36+0.54, color='#C0C0C0', alpha=0.50)
    #plt.axvspan(67.36-2.*0.54, 67.36+2.*0.54, color='#C0C0C0', alpha=0.30)
    #plt.axvspan(74.03-1.42, 74.03+1.42, color='#F08080', alpha=0.50)
    #plt.axvspan(74.03-2.*1.42, 74.03+2.*1.42, color='#F08080', alpha=0.30)
    ##plt.axvspan(rec[0,1]-rec[0,2], rec[0,1]+rec[0,2], color='#87CEFA', alpha=0.30)
    
    #plt.errorbar(67.36, 1, xerr=0.54, fmt='o', color='#FF0000')
    #plt.errorbar(74.03, 2, xerr=1.42, fmt='o', color='#0000FF')
    
    ##SKA IM band 1 and 2
    #z1a,h1a,sig1a=loadtxt('rech_hz_im_band1and2_'+str(nz1+0)+'+'+str(nz2)+'pts_p18.dat',unpack='True')
    #z1b,h1b,sig1b=loadtxt('rech_hz_im_band1and2_'+str(nz1+0)+'+'+str(nz2)+'pts_r19.dat',unpack='True')
    #z2a,h2a,sig2a=loadtxt('rech_hz_im_band1and2_'+str(nz1+5)+'+'+str(nz2)+'pts_p18.dat',unpack='True')
    #z2b,h2b,sig2b=loadtxt('rech_hz_im_band1and2_'+str(nz1+5)+'+'+str(nz2)+'pts_r19.dat',unpack='True')
    #z3a,h3a,sig3a=loadtxt('rech_hz_im_band1and2_'+str(nz1+10)+'+'+str(nz2)+'pts_p18.dat',unpack='True')
    #z3b,h3b,sig3b=loadtxt('rech_hz_im_band1and2_'+str(nz1+10)+'+'+str(nz2)+'pts_r19.dat',unpack='True')
    #z4a,h4a,sig4a=loadtxt('rech_hz_im_band1and2_'+str(nz1+15)+'+'+str(nz2)+'pts_p18.dat',unpack='True')
    #z4b,h4b,sig4b=loadtxt('rech_hz_im_band1and2_'+str(nz1+15)+'+'+str(nz2)+'pts_r19.dat',unpack='True')
    #z5a,h5a,sig5a=loadtxt('rech_hz_im_band1and2_'+str(nz1+20)+'+'+str(nz2)+'pts_p18.dat',unpack='True')
    #z5b,h5b,sig5b=loadtxt('rech_hz_im_band1and2_'+str(nz1+20)+'+'+str(nz2)+'pts_r19.dat',unpack='True')

    ##setting a starting value for y-axis
    #y_val = 1.0 
    ##print(h30a[0],sigh30a[0],h30b[0],sigh30b[0])
    
    ##SKA IM band 1 and 2
    #plt.errorbar(h1a[0], y_val+0.0, xerr=2.*sig1a[0], fmt='o', color='black')
    #plt.errorbar(h1b[0], y_val+0.5, xerr=2.*sig1b[0], fmt='x', color='black')
    #plt.errorbar(h2a[0], y_val+1.0, xerr=2.*sig2a[0], fmt='o', color='#0000FF')
    #plt.errorbar(h2b[0], y_val+1.5, xerr=2.*sig2b[0], fmt='x', color='#0000FF')
    #plt.errorbar(h3a[0], y_val+2.0, xerr=2.*sig3a[0], fmt='o', color='#FF00FF')
    #plt.errorbar(h3b[0], y_val+2.5, xerr=2.*sig3b[0], fmt='x', color='#FF00FF')
    #plt.errorbar(h4a[0], y_val+3.0, xerr=2.*sig4a[0], fmt='o', color='#00FFFF')
    #plt.errorbar(h4b[0], y_val+3.5, xerr=2.*sig4b[0], fmt='x', color='#00FFFF')
    #plt.errorbar(h5a[0], y_val+4.0, xerr=2.*sig5a[0], fmt='o', color='#228B22')
    #plt.errorbar(h5b[0], y_val+4.5, xerr=2.*sig5b[0], fmt='x', color='#228B22')
    
    #print(25, (sig1a[0]/h1a[0])*100., (h1b[0]-h1a[0])/np.sqrt(sig1a[0]**2. + sig1b[0]**2.))
    #print(30, (sig2a[0]/h2a[0])*100., (h2b[0]-h2a[0])/np.sqrt(sig2a[0]**2. + sig2b[0]**2.))
    #print(35, (sig3a[0]/h3a[0])*100., (h3b[0]-h3a[0])/np.sqrt(sig3a[0]**2. + sig3b[0]**2.))
    #print(40, (sig4a[0]/h4a[0])*100., (h4b[0]-h4a[0])/np.sqrt(sig4a[0]**2. + sig4b[0]**2.))
    #print(50, (sig5a[0]/h5a[0])*100., (h5b[0]-h5a[0])/np.sqrt(sig5a[0]**2. + sig5b[0]**2.))

    #plt.title(r'$H_0$ reconstructed', fontsize='20')
    #plt.legend((r"P18 ($1\sigma$)", "P18 ($2\sigma$)", "R19 ($1\sigma$)", "R19 ($2\sigma$)", "15 pts, P18", "15 pts, R19", "20 pts, P18", "20 pts, R19", "25 pts, P18", "25 pts, R19", "30 pts, P18", "30 pts, R19", "35 pts, P18", "35 pts, R19"), fontsize='17', loc='upper left')
    #plt.show()

    #### saving the plot
    #fig.savefig('h0_reconst_ska_im_band1and2-a.png')
    
    ##SKA IM band 1
    #z1a,h1a,sig1a=loadtxt('rech_hz_im_band1_'+str(nz1+0)+'pts_p18.dat',unpack='True')
    #z1b,h1b,sig1b=loadtxt('rech_hz_im_band1_'+str(nz1+0)+'pts_r19.dat',unpack='True')
    #z2a,h2a,sig2a=loadtxt('rech_hz_im_band1_'+str(nz1+5)+'pts_p18.dat',unpack='True')
    #z2b,h2b,sig2b=loadtxt('rech_hz_im_band1_'+str(nz1+5)+'pts_r19.dat',unpack='True')
    #z3a,h3a,sig3a=loadtxt('rech_hz_im_band1_'+str(nz1+10)+'pts_p18.dat',unpack='True')
    #z3b,h3b,sig3b=loadtxt('rech_hz_im_band1_'+str(nz1+10)+'pts_r19.dat',unpack='True')
    #z4a,h4a,sig4a=loadtxt('rech_hz_im_band1_'+str(nz1+15)+'pts_p18.dat',unpack='True')
    #z4b,h4b,sig4b=loadtxt('rech_hz_im_band1_'+str(nz1+15)+'pts_r19.dat',unpack='True')
    #z5a,h5a,sig5a=loadtxt('rech_hz_im_band1_'+str(nz1+20)+'pts_p18.dat',unpack='True')
    #z5b,h5b,sig5b=loadtxt('rech_hz_im_band1_'+str(nz1+20)+'pts_r19.dat',unpack='True')

    ##setting a starting value for y-axis
    #y_val = 1.0 
    ##print(h30a[0],sigh30a[0],h30b[0],sigh30b[0])
    
    ##SKA IM band 1 and 2
    #plt.errorbar(h1a[0], y_val+0.0, xerr=2.*sig1a[0], fmt='o', color='black')
    #plt.errorbar(h1b[0], y_val+0.5, xerr=2.*sig1b[0], fmt='x', color='black')
    #plt.errorbar(h2a[0], y_val+1.0, xerr=2.*sig2a[0], fmt='o', color='#0000FF')
    #plt.errorbar(h2b[0], y_val+1.5, xerr=2.*sig2b[0], fmt='x', color='#0000FF')
    #plt.errorbar(h3a[0], y_val+2.0, xerr=2.*sig3a[0], fmt='o', color='#FF00FF')
    #plt.errorbar(h3b[0], y_val+2.5, xerr=2.*sig3b[0], fmt='x', color='#FF00FF')
    #plt.errorbar(h4a[0], y_val+3.0, xerr=2.*sig4a[0], fmt='o', color='#00FFFF')
    #plt.errorbar(h4b[0], y_val+3.5, xerr=2.*sig4b[0], fmt='x', color='#00FFFF')
    #plt.errorbar(h5a[0], y_val+4.0, xerr=2.*sig5a[0], fmt='o', color='#228B22')
    #plt.errorbar(h5b[0], y_val+4.5, xerr=2.*sig5b[0], fmt='x', color='#228B22')
    
    #print(25, (sig1a[0]/h1a[0])*100., (h1b[0]-h1a[0])/np.sqrt(sig1a[0]**2. + sig1b[0]**2.))
    #print(30, (sig2a[0]/h2a[0])*100., (h2b[0]-h2a[0])/np.sqrt(sig2a[0]**2. + sig2b[0]**2.))
    #print(35, (sig3a[0]/h3a[0])*100., (h3b[0]-h3a[0])/np.sqrt(sig3a[0]**2. + sig3b[0]**2.))
    #print(40, (sig4a[0]/h4a[0])*100., (h4b[0]-h4a[0])/np.sqrt(sig4a[0]**2. + sig4b[0]**2.))
    #print(50, (sig5a[0]/h5a[0])*100., (h5b[0]-h5a[0])/np.sqrt(sig5a[0]**2. + sig5b[0]**2.))

    #plt.title(r'$H_0$ reconstructed', fontsize='20')
    #plt.legend((r"P18 ($1\sigma$)", "P18 ($2\sigma$)", "R19 ($1\sigma$)", "R19 ($2\sigma$)", "10 pts, P18", "10 pts, R19", "15 pts, P18", "15 pts, R19", "20 pts, P18", "20 pts, R19", "25 pts, P18", "25 pts, R19", "30 pts, P18", "30 pts, R19"), fontsize='17', loc='upper left')
    #plt.show()

    #### saving the plot
    #fig.savefig('h0_reconst_ska_im_band1-a.png')
    
    ##Euclid
    #z1a,h1a,sig1a=loadtxt('rech_euclid_'+str(nz1+0)+'pts_p18.dat',unpack='True')
    #z1b,h1b,sig1b=loadtxt('rech_euclid_'+str(nz1+0)+'pts_r19.dat',unpack='True')
    #z2a,h2a,sig2a=loadtxt('rech_euclid_'+str(nz1+5)+'pts_p18.dat',unpack='True')
    #z2b,h2b,sig2b=loadtxt('rech_euclid_'+str(nz1+5)+'pts_r19.dat',unpack='True')
    #z3a,h3a,sig3a=loadtxt('rech_euclid_'+str(nz1+10)+'pts_p18.dat',unpack='True')
    #z3b,h3b,sig3b=loadtxt('rech_euclid_'+str(nz1+10)+'pts_r19.dat',unpack='True')
    #z4a,h4a,sig4a=loadtxt('rech_euclid_'+str(nz1+15)+'pts_p18.dat',unpack='True')
    #z4b,h4b,sig4b=loadtxt('rech_euclid_'+str(nz1+15)+'pts_r19.dat',unpack='True')
    #z5a,h5a,sig5a=loadtxt('rech_euclid_'+str(nz1+20)+'pts_p18.dat',unpack='True')
    #z5b,h5b,sig5b=loadtxt('rech_euclid_'+str(nz1+20)+'pts_r19.dat',unpack='True')
    #z6a,h6a,sig6a=loadtxt('rech_euclid_'+str(nz1+25)+'pts_p18.dat',unpack='True')
    #z6b,h6b,sig6b=loadtxt('rech_euclid_'+str(nz1+25)+'pts_r19.dat',unpack='True')
    #z7a,h7a,sig7a=loadtxt('rech_euclid_'+str(nz1+30)+'pts_p18.dat',unpack='True')
    #z7b,h7b,sig7b=loadtxt('rech_euclid_'+str(nz1+30)+'pts_r19.dat',unpack='True')
    
    ##setting a starting value for y-axis
    #y_val = 1.0 
    ##print(h30a[0],sigh30a[0],h30b[0],sigh30b[0])
    
    #print(10, (sig1a[0]/h1a[0])*100., (h1b[0]-h1a[0])/np.sqrt(sig1a[0]**2. + sig1b[0]**2.))
    #print(15, (sig2a[0]/h2a[0])*100., (h2b[0]-h2a[0])/np.sqrt(sig2a[0]**2. + sig2b[0]**2.))
    #print(20, (sig3a[0]/h3a[0])*100., (h3b[0]-h3a[0])/np.sqrt(sig3a[0]**2. + sig3b[0]**2.))
    #print(25, (sig4a[0]/h4a[0])*100., (h4b[0]-h4a[0])/np.sqrt(sig4a[0]**2. + sig4b[0]**2.))
    #print(30, (sig5a[0]/h5a[0])*100., (h5b[0]-h5a[0])/np.sqrt(sig5a[0]**2. + sig5b[0]**2.))
    #print(30, (sig6a[0]/h6a[0])*100., (h6b[0]-h6a[0])/np.sqrt(sig6a[0]**2. + sig6b[0]**2.))
    #print(30, (sig7a[0]/h7a[0])*100., (h7b[0]-h7a[0])/np.sqrt(sig7a[0]**2. + sig7b[0]**2.))
    
    #plt.errorbar(h1a[0], y_val+0.0, xerr=2.*sig1a[0], fmt='o', color='black')
    #plt.errorbar(h1b[0], y_val+0.5, xerr=2.*sig1b[0], fmt='x', color='black')
    #plt.errorbar(h2a[0], y_val+1.0, xerr=2.*sig2a[0], fmt='o', color='#0000FF')
    #plt.errorbar(h2b[0], y_val+1.5, xerr=2.*sig2b[0], fmt='x', color='#0000FF')
    #plt.errorbar(h3a[0], y_val+2.0, xerr=2.*sig3a[0], fmt='o', color='#FF00FF')
    #plt.errorbar(h3b[0], y_val+2.5, xerr=2.*sig3b[0], fmt='x', color='#FF00FF')
    #plt.errorbar(h4a[0], y_val+3.0, xerr=2.*sig4a[0], fmt='o', color='#00FFFF')
    #plt.errorbar(h4b[0], y_val+3.5, xerr=2.*sig4b[0], fmt='x', color='#00FFFF')
    #plt.errorbar(h5a[0], y_val+4.0, xerr=2.*sig5a[0], fmt='o', color='#228B22')
    #plt.errorbar(h5b[0], y_val+4.5, xerr=2.*sig5b[0], fmt='x', color='#228B22')
    #plt.errorbar(h6a[0], y_val+5.0, xerr=2.*sig6a[0], fmt='o', color='#8A2BE2')
    #plt.errorbar(h6b[0], y_val+5.5, xerr=2.*sig6b[0], fmt='x', color='#8A2BE2')
    #plt.errorbar(h7a[0], y_val+6.0, xerr=2.*sig7a[0], fmt='o', color='#FFD700')
    #plt.errorbar(h7b[0], y_val+6.5, xerr=2.*sig7b[0], fmt='x', color='#FFD700')
  
    #plt.title(r'$H_0$ reconstructed', fontsize='20')
    #plt.legend((r"P18 ($1\sigma$)", "P18 ($2\sigma$)", "R19 ($1\sigma$)", "R19 ($2\sigma$)", "10 pts, P18", "10 pts, R19", "15 pts, P18", "15 pts, R19", "20 pts, P18", "20 pts, R19", "25 pts, P18", "25 pts, R19", "30 pts, P18", "30 pts, R19",  "35 pts, P18", "35 pts, R19", "40 pts, P18", "40 pts, R19"), fontsize='17', loc='upper left')
    #plt.show()

    #### saving the plot
    #fig.savefig('h0_reconst_euclid-a.png')
    
    ##DESI
    #z1a,h1a,sig1a=loadtxt('rech_desi_'+str(nz1+0)+'pts_p18.dat',unpack='True')
    #z1b,h1b,sig1b=loadtxt('rech_desi_'+str(nz1+0)+'pts_r19.dat',unpack='True')
    #z2a,h2a,sig2a=loadtxt('rech_desi_'+str(nz1+5)+'pts_p18.dat',unpack='True')
    #z2b,h2b,sig2b=loadtxt('rech_desi_'+str(nz1+5)+'pts_r19.dat',unpack='True')
    #z3a,h3a,sig3a=loadtxt('rech_desi_'+str(nz1+10)+'pts_p18.dat',unpack='True')
    #z3b,h3b,sig3b=loadtxt('rech_desi_'+str(nz1+10)+'pts_r19.dat',unpack='True')
    ##z4a,h4a,sig4a=loadtxt('rech_euclid_'+str(nz1+15)+'pts_p18.dat',unpack='True')
    ##z4b,h4b,sig4b=loadtxt('rech_euclid_'+str(nz1+15)+'pts_r19.dat',unpack='True')
    #z5a,h5a,sig5a=loadtxt('rech_desi_'+str(nz1+20)+'pts_p18.dat',unpack='True')
    #z5b,h5b,sig5b=loadtxt('rech_desi_'+str(nz1+20)+'pts_r19.dat',unpack='True')
    ##z6a,h6a,sig6a=loadtxt('rech_euclid_'+str(nz1+25)+'pts_p18.dat',unpack='True')
    ##z6b,h6b,sig6b=loadtxt('rech_euclid_'+str(nz1+25)+'pts_r19.dat',unpack='True')
    ##z7a,h7a,sig7a=loadtxt('rech_desi_'+str(nz1+30)+'pts_p18.dat',unpack='True')
    ##z7b,h7b,sig7b=loadtxt('rech_desi_'+str(nz1+30)+'pts_r19.dat',unpack='True')
    
    ##setting a starting value for y-axis
    #y_val = 1.0 
    ##print(h30a[0],sigh30a[0],h30b[0],sigh30b[0])
    
    #print(10, (sig1a[0]/h1a[0])*100., (h1b[0]-h1a[0])/np.sqrt(sig1a[0]**2. + sig1b[0]**2.))
    #print(15, (sig2a[0]/h2a[0])*100., (h2b[0]-h2a[0])/np.sqrt(sig2a[0]**2. + sig2b[0]**2.))
    #print(20, (sig3a[0]/h3a[0])*100., (h3b[0]-h3a[0])/np.sqrt(sig3a[0]**2. + sig3b[0]**2.))
    ##print(25, (sig4a[0]/h4a[0])*100., (h4b[0]-h4a[0])/np.sqrt(sig4a[0]**2. + sig4b[0]**2.))
    #print(30, (sig5a[0]/h5a[0])*100., (h5b[0]-h5a[0])/np.sqrt(sig5a[0]**2. + sig5b[0]**2.))
    ##print(30, (sig6a[0]/h6a[0])*100., (h6b[0]-h6a[0])/np.sqrt(sig6a[0]**2. + sig6b[0]**2.))
    ##print(30, (sig7a[0]/h7a[0])*100., (h7b[0]-h7a[0])/np.sqrt(sig7a[0]**2. + sig7b[0]**2.))
    
    ####################################################################
    #npts1,perc1,th1=loadtxt('th0_vs_npts_ska_im_band1.dat',unpack='True')
    #npts2,perc2,th2=loadtxt('th0_vs_npts_ska_im_band2.dat',unpack='True')
    ##npts3,perc3,th3=loadtxt('th0_vs_npts_ska_im_band1and2-a.dat',unpack='True')
    #npts4,perc4,th4=loadtxt('th0_vs_npts_ska_im_band1and2-b.dat',unpack='True')
    ##npts5,perc5,th5=loadtxt('th0_vs_npts_ska_gs_band2.dat',unpack='True')
    #npts6,perc6,th6=loadtxt('th0_vs_npts_euclid.dat',unpack='True')
    ##npts7,perc7,th7=loadtxt('th0_vs_npts_meerkat_im_bandL.dat',unpack='True')
    
    ##latex rendering text fonts
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    ## Create figure size in inches
    #fig, ax = plt.subplots(figsize = (9., 7.))

    ## Define axes
    #ax.set_ylabel(r"$T_{H_0}$ ($\sigma$)", fontsize=22)
    ##ax.set_ylabel(r"$\sigma_{H_0}/H_0$ (\%)", fontsize=22)
    #ax.set_xlabel(r"num pts", fontsize=22)
    ##plt.ylim(0.5,7.5)
    ##plt.ylim(0.5,7.5)
    #plt.xlim(3.,53.)
    #for t in ax.get_xticklabels(): t.set_fontsize(20)
    #for t in ax.get_yticklabels(): t.set_fontsize(20)
    
    #plt.plot(npts1, perc1, '-x', color='black')
    #plt.plot(npts2, perc2, '-*', color='red')
    #plt.plot(npts4, perc4, '-v', color='blue')
    #plt.plot(npts6, perc6, '-o', color='#FF00FF')
    ##plt.plot(npts7, perc7, '-+', color='#00FF00')
    ##plt.plot(npts1, th1, '-x', color='black')
    ##plt.plot(npts2, th2, '-*', color='red')
    ##plt.plot(npts4, th4, '-v', color='blue')
    ##plt.plot(npts6, th6, '-o', color='#FF00FF')
    #plt.axhline(y=0.802, color='#8A2BE2')
    #plt.axhline(y=1.981, color='#228B22')
    ##plt.axhline(y=4.4, color='#228B22')

    #plt.legend((r"SKA IM B1", "SKA IM B2", "SKA IM B1+B2", "Euclid", "P18", "R19"), loc='center right', fontsize='16')  
    ##plt.legend((r"SKA IM B1", "SKA IM B2", "SKA IM B1+B2", "Euclid", "P18 vs. R19"), loc='center right', fontsize='18')  
    #plt.show()

    ##saving the plot
    ##fig.savefig('th0.png')
    #fig.savefig('perc.png')
    
    ####################################################################
    #now plotting the reconstructed h0 values with comparison to R19 and P18 

    h,y,hlow,hup=loadtxt('h0_data.dat',unpack='True')
    #print(h,herrlow,herrupp)

    #latex rendering text fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Create figure size in inches
    fig, ax = plt.subplots(figsize = (9., 7.))

    # Define axes
    ax.set_xlabel(r"$H_0 \; (\mathrm{km} \; \mathrm{s}^{-1} \; \mathrm{Mpc}^{-1}$)", fontsize=22)
    #ax.set_ylabel(r"measurement", fontsize=22)
    plt.ylim(0.8,5.2)
    plt.xlim(55.,85.)
    for t in ax.get_xticklabels(): t.set_fontsize(20)
    for t in ax.get_yticklabels(): t.set_fontsize(20)

    # plotting
    plt.axvspan(67.36-0.54, 67.36+0.54, color='#C0C0C0', alpha=0.50)
    plt.axvspan(67.36-2.*0.54, 67.36+2.*0.54, color='#C0C0C0', alpha=0.30)
    plt.axvspan(74.03-1.42, 74.03+1.42, color='#F08080', alpha=0.50)
    plt.axvspan(74.03-2.*1.42, 74.03+2.*1.42, color='#F08080', alpha=0.30)
    plt.errorbar(h, y, xerr=[hlow,hup], fmt='x', color='black')
    #plt.errorbar(68.0, 1.0, xerr=4.15, fmt='o', color='black')
    #plt.errorbar(71.3, 1.5, xerr=4.25, fmt='x', color='blue')
    #plt.errorbar(67.42, 2.0, xerr=4.75, fmt='v', color='red')
    #plt.errorbar(67.2, 2.5, xerr=1.05, fmt='*', color='cyan')
    #plt.errorbar(69.13, 3.0, xerr=2.34, fmt='>', color='magenta')
    #plt.errorbar(70., 3.5, xerr=10., fmt='<', color='LimeGreen')
    #plt.errorbar(71., 4.0, xerr=3.5, fmt='+', color='brown')
    #plt.errorbar(73.04, 4.5, xerr=0.931, fmt='.', color='green')
    #plt.errorbar(66.46, 5.0, xerr=0.847, fmt='^', color='green')
    plt.show()
    
     #saving the plot
    #fig.savefig('th0.png')
    fig.savefig('h0_meas1.png')
