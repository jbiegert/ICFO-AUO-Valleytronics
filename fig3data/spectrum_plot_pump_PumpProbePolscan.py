#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:52:10 2020
Takes spectrometer data and plots them.
@author: ityulnev
"""



import sys
from pathlib import Path
import os
import glob
import numpy as np
from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
from scipy import constants
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from PIL import ImageFilter
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pylab import * #for figtext
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, hilbert, chirp
from scipy.ndimage import gaussian_filter
from BaselineRemoval import BaselineRemoval
import re
sys.path.append(r'C:\Users\Igor\Documents\PhD\Data\230726')



#%% Custom functions
def smooth0(y, box_pts, loop=1): 
    box = np.ones(box_pts)/box_pts
    y_smooth = y
    for i in range(loop):
        y_smooth = np.convolve(y_smooth, box, mode='same')
    y_smooth[0:2*box_pts]=y[2*box_pts]
    y_smooth[-2*box_pts:]=y[-2*box_pts]
    return y_smooth

def smooth(arr, span):
    if span==0:
        return arr
    
    arr = savgol_filter(arr, span * 2 + 1, 2)
    arr = savgol_filter(arr, span * 2 + 1, 2)
    return arr

def mynorm(arr):
    return arr/max(arr)

def myfft(Et,t):
    """Physical FFT implementation, can take zero centered un-/even array
    with coefficients as [a-2,a-1,a0,a1,a2] or [a-2,a-1,a0,a1]"""
    N = len(Et)
    T = (max(t)- min(t))
    dt = T/N
    Ef = fftshift(fft(ifftshift(Et))) * (T/N) # multiplied by dt
    f = np.arange(-math.floor(N/2),math.ceil(N/2),1)/(T)
    return(Ef,f)

def myifft(Ef,f):
    """Phyiscal iFFT implementation, can take zero centered un-/even array
    with coefficients as [a-2,a-1,a0,a1,a2] or [a-2,a-1,a0,a1]"""
    N = len(Ef)
    F = max(f)-min(f)
    dt =  1/(F) 
    Et = fftshift(ifft(ifftshift(Ef))) * F # divided by dt
    t = (np.arange(0,N,1)-(math.ceil(N/2)-1))*(dt)
    return(Et,t)

# def symmetrize_frame(data_frame):
#     """Take data_frame with rows [t,E,It] and symmetrize t around zero.
#     Et and Ir are zero padded accordingly"""
#     tt = data_frame.iloc[:,0]
    
#     tt_extra = tt.where(tt>(-tt.min()))
#     tt_extra.dropna(axis="rows", how="any", inplace=True)
#     # tt_new = pd.concat([-tt_extra[::-1], tt], join='inner',ignore_index=True)
#     data = {'t':-tt_extra[::-1] ,'Et':[0]*len(tt_extra) ,'It':[0]*len(tt_extra)} 
#     pad_frame = pd.DataFrame(data) 

#     new_frame = pd.concat([pad_frame, data_frame], join='inner',ignore_index=True)
#     return(new_frame)

# def zeropad_frame(data_frame,N):
#     data = {'t':[0]*N,'Et':[0]*N ,'It':[0]*N} 
#     pad_frame = pd.DataFrame(data) 
#     new_frame = pd.concat([pad_frame, data_frame, pad_frame], join='inner',ignore_index=True)
#     return(new_frame)

# class mySupergauss(object): 
#     def __init__(self,x,x_offset=0,y_fwhm=100e-15,order=1):
#         """Calculate a Guassian function of a given order on the x axis"""
#         self.y_fwhm = y_fwhm
#         self.order = order
#         self.x_offset = x_offset
#         self.y = np.exp(-((((x-self.x_offset)/(self.y_fwhm/(2*np.sqrt(np.log(2)))))**2))**self.order);

def MakeHeaderToAxis(name): #for FROG files
    """Read HEADER"""
    with open(name) as myfile:
        header=myfile.readlines()[0:5]
    data0=np.asfarray(header,float)
    Ntpoints = int(data0[0])
    tstep = data0[2]
    Nwpoints = int(data0[1])
    wstep = data0[3]
    wcenter = data0[4]
    
    t0= np.arange(0.0, Ntpoints*tstep, tstep)
    w0= np.linspace(wcenter-0.5*Nwpoints*wstep, wcenter+0.5*Nwpoints*wstep, Nwpoints)
    
    return t0/2,w0,Ntpoints,Nwpoints,wcenter
           #time sampling
           
           
def Norm_NoiseAv(signalXY):
    signalXY_norm = []
    ylength=len(signalXY[:,0])
    noiseAv_init = sum(signalXY[:,0])/ylength
    for ii in np.arange(0,len(signalXY[0,:])):
        noiseAv_roll = sum(signalXY[:,ii])/ylength
        signalXY_norm.append( (signalXY[:,ii])*noiseAv_init/noiseAv_roll)
    return signalXY_norm


        
    
def cut_stripe(Ixy,axis,dimension,lmin,lmax):
    if dimension==0:
        Ixy_cut = Ixy[np.where((lmin<=axis) & (axis<=lmax)),:]
        axis_cut = axis[np.where((lmin<=axis) & (axis<=lmax))]
    if dimension==1:
        Ixy_cut = Ixy[:,np.where((lmin<=axis) & (axis<=lmax))]
        axis_cut = axis[np.where((lmin<=axis) & (axis<=lmax))]
    return Ixy_cut[:,0,:], axis_cut
           
def amplify_region(Ixy,axis,dimension,lmin,lmax,amp_factor):
    if dimension==0:
        Ixy[np.where((lmin<=axis) & (axis<=lmax)),:]*amp_factor
    if dimension==1:
        Ixy[:,np.where((lmin<=axis) & (axis<=lmax))]*amp_factor
    return Ixy[:,:]



def WvltoFreq(xwvl,ywvl,axis):
    xfreq = constants.c/xwvl
    if axis==0:
        yfreq=[]
        for ii in np.arange(0,len(ywvl[:,0]),1):
            yfreq.append(np.flip(ywvl[ii,:]*xwvl**2/constants.c))

    return np.flip(xfreq),yfreq

def BaselineLoop(inArray,box_pts):
    outArray=[]
    for ii in np.arange(0,len(inArray[0,:])):
        smoothline = inArray[:,ii]
        baseline = baseline_als(smoothline,p=1e-1,lam=100,niter=10)
        outArray.append(smoothline-baseline)
        # outArray.append(smoothline)
    return np.array(outArray).transpose()


def SmoothLoop(inArray,box_pts):
    outArray=[]
    for ii in np.arange(0,len(inArray[:,0])):
        outArray.append(smooth(inArray[ii,:],box_pts))
    return np.array(outArray)

#https://stackoverflow.com/questions/29156532/python-baseline-correction-library
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


#%% My custom jet plus cmap, where lowest level is colored white
cdict = {'red': ((0.0, 1, 1),
                 (0.02, 0, 0),
                 (0.05, 0, 0),
                 (0.11, 0, 0),
                 (0.66, 1, 1),
                 (0.89, 1, 1),
                 (1, 0.5, 0.5)),
         'green': ((0.0, 1, 1),
                   (0.02, 0, 0),
                   (0.05, 0, 0),
                   (0.11, 0, 0),
                   (0.375, 1, 1),
                   (0.64, 1, 1),
                   (0.91, 0, 0),
                   (1, 0, 0)),
         'blue': ((0.0, 1, 1),
                  (0.02, 0.5, 0.5),                  
                  (0.05, 0.7, 0.7),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65, 0, 0),
                  (1, 0, 0))}

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
# my_cmap = cm.Blues
# gcmap = cm.jet   # global colormap
gcmap = cm.gist_rainbow   # global colormap
# gcmap = cm.copper   # global colormap




#%% Build filepath, 
genpath = Path(r"C:\Users\Igor\Documents\Phd\Data\230726\scans\pp_rotscans")
myfilenames = sorted(genpath.glob(r"*MoS2_03_PPpumpdelayROTscan*SPECTRA.txt"))
mynames=[]
mynames_sorted = []
for ii in myfilenames:   
    mynames.append(str(ii))
    print(str(ii))
# mynames_sorted = sorted(mynames, key = lambda x: re.search(r'pppos*([-]?\d*.\d*)', x).group(1))
# mynames_sorted = sorted(mynames, key = lambda x: re.search(r'pppos*(\d*.\d*)', x).group(1))
mynames_sorted = sorted(mynames, key = lambda x: int(1000*float(re.search(r'pppos([-]?\d*.\d*)', x).group(1))))

# mynames_sorted = sorted(mynames, key = lambda x: re.search(r'(\d*.\d*)_SPECTRA', x).group(1))

# mynames_sorted = sorted(mynames, key = lambda x: re.search(r'(\d+.\d+)_SPECTRA', x).group(1))

# print(re.search(r'pppos([-]?\d*.\d*)', str(myfilenames[0])).group(1))
# mynames_sorted = sorted(mynames, key = lambda x: float(re.findall(r'wedgepos(\d+.\d+)', x)[-1])) #sorted for angles
# MoS2_05_polrotscan_3umPol352QWP133_1600HWP35QWP11_0_360000um_step2deg_wedgepos10.96mm_int200.0ms_avg4_runtime327.29s_SPECTRA

# print(re.findall(r'pppos(\d+.\d+)',mynames[-1]))

mywavelengths = sorted(genpath.glob(r"*MoS2_03_PPpumpdelayROTscan*WAVELENGTHS.txt"))
mywedgepos = sorted(genpath.glob(r"*MoS2_03_PPpumpdelayROTscan*POSITIONS.txt"))
probe_pos = []
#%%
Af_arr = []
ff_arr = []
probe_avpeaks = []

AH3 = []
AH4 = []
AH5 = []
AH6 = []
AH7 = []
AH8 = []
AH_noise = []
AH_nsum = []
# mycolormap01=cm.Paired
mycolormap01=cm.jet

# mycolormap=mpl.colors.LinearSegmentedColormap.from_list("jet", Mcolors)
my1d_colors01 = mycolormap01(np.linspace(1,0,len(mynames_sorted)))
harm_number = [4,8] #choose which harmoics to plot
# harm_number = [7,8] #choose which harmoics to plot

ll=0
QWPpos = []
probe_pos = []
plt.rcParams.update({'font.size': 20})
fig2 = plt.figure(figsize=(8,7))
ax2 = fig2.add_subplot(111, polar=False)
   
    # mm = mynames_sorted[0]
for mm in mynames_sorted: 
    #% Load files into dict
    dict_keys = []
    file_dict = {}
    # os.path.isfile(mynames[0])
    
    
    print(mm) 
    shortname = Path(mm).parts[-1].replace(".txt","")
    # probe_pos.append(float(re.search(r'([-]?\d*.\d*)\D+$', shortname).group(1)))
    # probe_pos.append(float(re.search(r'([-]?\d*)\D+$', shortname).group(1)))
    # probe_pos.append(float(re.findall(r'(\d+.\d+)_SPECTRA', mm)[-1]))
    probe_pos.append(float(re.findall(r'pppos([-]?\d*.\d*)', mm)[-1]))

    

    dict_keys.append(shortname)
    myfile = np.loadtxt(mm)
    file_dict[shortname] = myfile
    
    savename = shortname
    print(savename)
       
    #% Colormesh
    angles = np.loadtxt(mywedgepos[0])#*1000/166*360
    spectra = file_dict[dict_keys[0]]
    smoothSpectra = SmoothLoop(spectra,5)
    wavelengths = np.loadtxt(mywavelengths[0])
    
    #% Lineouts
    # harm_number = [8] #choose which harmoics to plot
    # ind_harm = 4 #FOR FURTHER ANALYSIS  
    
    n_myrange = 10 # delta is +/- myrange nm
    n_rangeind0 = np.logical_and(wavelengths>=(415-n_myrange),wavelengths<=(415+n_myrange)) 
    
    A_noise = np.zeros((len(angles),1))
    A_sum = np.zeros((len(angles),1)) 
    for ii in range(0,len(angles)):
        A_noise[ii] = np.std(spectra[ii,n_rangeind0])
        A_sum[ii] = np.mean(spectra[ii,n_rangeind0])
    AH_noise.append(A_noise)
    AH_nsum.append(A_sum)

    mypos = ([265,290,320,356,400,460,530,647,800,1067])
    myharm = ([0,3200,1600,1067,800,630,530,465,400,356,320,290,267])
    # print('\nANALYSIS for H{}: {}nm\n\n'.format(ind_harm,myharm[ind_harm]) )
    
    # select harmonics for plot legend
    leg_str = [ myharm[i] for i in harm_number]
    leg_str = list(map(str,leg_str))
    
 
    mydata_cut = spectra
    myt0 = angles
    myw0 = wavelengths
    Apeak = np.zeros((len(myt0),len(myharm)))
    for pos in myharm:
        myrange = 10 # delta is +/- myrange nm
        rangeind0 = np.logical_and(myw0>=(pos-myrange),myw0<=(pos+myrange)) 
        # rangeind_l = wvl.where(400-myrange)
        for ii in range(0,len(myt0)):
            Apeak[ii,myharm.index(pos)] = sum(mydata_cut[ii,rangeind0])
    
        
    Apeak_norm = np.zeros((len(myt0),len(myharm)))
    for ii in range(0,len(Apeak[0,:])):
        normi = (np.average(Apeak[:,ii])) if (np.average(Apeak[:,ii])) > 0 else 1
        Apeak_norm[:,ii] = smooth(Apeak[:,ii]/normi,1)
    for ii in harm_number:
        myt0[0]=0
        ax2.plot((myt0),Apeak_norm[:,ii]/max(Apeak_norm[:,ii]),'o-',linewidth=3, markersize=2.5,color=my1d_colors01[ll],label=str(probe_pos[ll]))
        # ax2.plot(myt0,Apeak[:,ii]/max(Apeak[:,ii]),'o-',linewidth=3, markersize=2.5,color=my1d_colors01[ll],label=str(probe_pos[ll]))
    AH8.append(Apeak[:,8])
    AH4.append(Apeak[:,4])
    
    
    ll +=1
# ax2.set_yticks([0,0.25,0.5,0.75,1])
# ax2.set_ylim(0,1)
# ax2.legend(["H3","H4","H5","H6"],loc="lower right")
# ax2.legend(loc="lower right")
# ax2.set_ylim([0,2])
plt.tight_layout()
# plt.savefig(("MoS2_scan01_WedgeVsPol2"+"av_Lineouts_H4"+".png"),dpi=(250))
# plt.savefig(("MoS2_scan02_WedgeVsPol"+"av_Lineouts_H8"+".png"),dpi=(250))
#%%
plt.rcParams.update({'font.size': 24})
fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.pcolormesh(wavelengths,myt0,smoothSpectra,shading='None')# shading='gouraud'
ax4.set_ylabel("wavelength (nm)")
ax4.set_xlabel(r"Wedge delay (mm)")
#%%
plt.rcParams.update({'font.size': 24})
fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.plot(wavelengths,smoothSpectra[5,:],"-",color="k",linewidth=2,alpha=1,label="30°")
ax4.plot(wavelengths,smoothSpectra[21,:],"-",color="b",linewidth=2,alpha=1,label="120°")

# ax4.plot(wavelengths,spectra[21,:],"-",color="b",linewidth=2,alpha=1,label="120°")


ax4.set_xlabel("wavelength (nm)")
ax4.set_ylabel("Intensity (counts)")
plt.legend()
plt.tight_layout()
# plt.savefig(("MoS2_scans521_"+"spectrum2"+".png"),dpi=(250))

#%%
plt.rcParams.update({'font.size': 24})
fig3, ax3 = plt.subplots(figsize=(10,6))
myt0[0]=0

probe_pos_act = -1*2*linspace(-0.08,0.08,20)/constants.c*1e-3*1e15

probe_pos_mm = linspace(-0.08,0.08,20)

AH8_arr = np.array(AH8)
AH8_arrnorm = np.zeros((len(AH8_arr[:,0]),len(angles))) 
for ii in range(0,len(AH8_arr[:,0])):
    AH8_arrnorm[ii,:] = mynorm(smooth(AH8_arr[ii,:],5))


H8_ellip = np.sum(AH8_arr[:,3:7],1)/np.sum(AH8_arr[:,19:23],1)
H8_min = np.sum(AH8_arr[:,3:7],1)/5
H8_max = np.sum(AH8_arr[:,19:23],1)/5

H8_ellipn = AH8_arrnorm[:,6]/AH8_arrnorm[:,24]


A0_noise = np.sum(np.array(AH_noise)[:,3:7],1)/5

min_err = 0.06 #from 3% std.dev. of probe 800nm 
max_err = 0.06
# err_tot = np.sqrt(min_err**2+max_err**2)
err_min = min_err/np.sqrt(2)#data averaged twice**2!
err_Hmax = np.sqrt((err_min*H8_max)**2+(A0_noise)**2)
err_Hmin = np.sqrt((err_min*H8_min)**2+(A0_noise)**2)

err_tot = np.sqrt((err_min)**2+(err_min)**2)
# err_tot = np.sqrt((err_Hmin/H8_min)**2+(err_Hmax/H8_max)**2)
# err_tot2 = (np.sqrt((A0_noise)**2+(err_min*H8_min)**2+(err_min*H8_max)**2))


H8_ellip_min = (H8_min-H8_min*err_min)/(H8_max+H8_max*err_min)
H8_ellip_max = (H8_min+H8_min*err_min)/(H8_max-H8_max*err_min)

H8_err_full = (H8_ellip_max-H8_ellip_min)/2


ax3.errorbar(probe_pos_mm,H8_ellip,yerr=err_tot*H8_ellip) #ftm = "--o" ,markersize=8
ax3.errorbar(probe_pos_mm,H8_ellipn,yerr=err_tot*H8_ellipn) #ftm = "--o" ,markersize=8
ax3.plot(probe_pos_mm,H8_ellipn,'o',color="r",markersize=5,label="ellipn") #ftm = "--o" ,markersize=8
ax3.plot(probe_pos_mm,H8_ellip,'o',color="b",markersize=5,label="ellip") #ftm = "--o" ,markersize=8

# ax3.errorbar(probe_pos_mm,H8_ellip,yerr=err_min*2) #ftm = "--o" ,markersize=8

# ax3.scatter(probe_pos_mm,H8_ellip_max)
# ax3.scatter(probe_pos_mm,H8_ellip_min)

# ax3.set_ylim(0,0.2)
# plt.savefig(("MoS2_scan02_WedgeVsPol"+"2D_H8"+".png"),dpi=(250))
ax3.set_ylabel("Probe H2 ellipticity")
ax3.set_xlabel("pump-probe delay (fs)")
plt.tight_layout()
# plt.savefig(("MoS2_scan03_PPVsPol"+"av_Lineouts_H8_ellip"+".png"),dpi=(250))


scan1_stack = np.vstack((H8_ellip,H8_err_full,probe_pos_mm))
# np.savetxt("scan03__H8_polscan_Smooth5.txt",scan1_stack)


#Baseline Removal
# baseline = baseline_als(Apeak_norm[:,harm_number][:,0],p=1e-1,lam=100,niter=10)
# ax2.plot(myt0,baseline,".-",c='k',linewidth=2)
# data0 = (Apeak_norm[:,harm_number][:,0]-baseline)
# ax2.plot(myt0,data0,".-",linewidth=2)
#Lineouts 
# ax1.plot(myt0,10*Apeak_norm[:,ind_harm-1]+myharm[ind_harm-1],".-",label='Trefoil',linewidth=2)
# ax1.plot(myt0,10*Apeak_norm[:,ind_harm]+myharm[ind_harm],".-",label='Trefoil',linewidth=2)
# ax1.plot(myt0,10*Apeak_norm[:,ind_harm+1]+myharm[ind_harm+1],".-",label='Trefoil',linewidth=2)
#%%
plt.rcParams.update({'font.size': 20})
fig4 = plt.figure(figsize=(8,7))
ax4 = fig4.add_subplot(111, polar=False)

Amax1 = smooth(np.array(AH8)[7,:],1)
Amax2 = smooth(np.array(AH8)[8,:],1)
Amax = mynorm(np.average((Amax1,Amax2),0))

Aprepump1 = smooth(np.array(AH8)[19,:],1)
Aprepump2 = smooth(np.array(AH8)[18,:],1)
Aprepump = mynorm(np.average((Aprepump1,Aprepump2),0))

Apostpump1 = smooth(np.array(AH8)[0,:],1)
Apostpump2 = smooth(np.array(AH8)[1,:],1)
Apostpump = mynorm(np.average((Apostpump1,Apostpump2),0))

# Apostpump = np.array(AH8)[0,:]

# Adiff = Amax-Aprepump
# Adiff_abs = np.array(AH8)[8,:]-np.array(AH8)[19,:] - np.mean( np.array(AH8)[8,:]-np.array(AH8)[19,:])

# Apre_noise = np.array(AH_noise)[19,:]
# Amax_noise = np.array(AH_noise)[8,:]


# Aunpump_noise = np.sqrt(Apre_noise**2 + (err_min*Aprepump)**2)[0,:]
Apump_noise = np.sqrt((err_min)**2+ (err_min)**2)

Amax_360 = np.hstack((Amax,Amax[1:-1])) 
Aprepump_360 = np.hstack((Aprepump,Aprepump[1:-1]))
Apostpump_360 = np.hstack((Apostpump,Apostpump[1:-1]))
myt0_360 = np.hstack((myt0,myt0[1:-1]+180))
Apump_noise_360 = Apump_noise*Amax_360/Aprepump_360

# ax4.errorbar(np.radians(myt0),Aprepump/max(Aprepump) ,yerr = 0.08*Aprepump/max(Aprepump),linewidth=3, markersize=2.5,label="unpumped")
# ax4.errorbar(np.radians(myt0),Amax/max(Amax),yerr = 0.08*Amax/max(Amax),linewidth=3, markersize=2.5,label="pumped")
# ax4.errorbar(np.radians(myt0),smooth(Aprepump,4),yerr = 2*Aunpump_noise,linewidth=1, label="unpumped",fmt ='o-',markersize=5,color ="k")
# ax4.errorbar(np.radians(myt0),Aprepump ,yerr = 0.0*Aprepump,linewidth=1, label="unpumped",fmt ='o-',markersize=5,color ="blue")
# ax4.errorbar(np.radians(myt0),smooth(Amax,4),yerr = 2*Apump_noise*Amax,linewidth=1,label="pumped",fmt ='o-',markersize=5,color="red")
ax4.errorbar(np.radians(myt0_360),smooth(Amax_360/Aprepump_360,2),yerr = Apump_noise_360,linewidth=1,label="norm. ratio",fmt ='o-',markersize=5,color="blue")
ax4.errorbar(np.radians(myt0_360),smooth(Apostpump_360/Aprepump_360,2),yerr = Apump_noise_360,linewidth=1,label="norm. ratio",fmt ='s-',markersize=5,color="darkblue")
ax4.errorbar(np.radians(myt0),smooth(mynorm(Amax1)/mynorm(Aprepump1),2),yerr = Apump_noise,linewidth=1,label="norm. ratio",fmt ='o-',markersize=5,color="red")
ax4.errorbar(np.radians(myt0),smooth(mynorm(Amax2)/mynorm(Aprepump1),2),yerr = Apump_noise,linewidth=1,label="norm. ratio",fmt ='o-',markersize=5,color="k")
ax4.errorbar(np.radians(myt0),smooth(mynorm(Apostpump)/mynorm(Aprepump),2),yerr = Apump_noise,linewidth=1,label="norm. ratio",fmt ='o-',markersize=5,color="gray")



# ax4.errorbar(myt0_360,smooth(Amax_360/Aprepump_360,2)-1,yerr = 0*err_tot*Amax_360/Aprepump_360,linewidth=1,label="norm. ratio",fmt ='o-',markersize=5,color="blue")
# ax4.errorbar(np.radians(myt0),smooth(Adiff,3)*5,yerr = 0.00*Amax,linewidth=1,label="norm. diff. x5",fmt ='o-',markersize=5,color="red")
# ax4.errorbar(np.radians(myt0),smooth(Adiff,3)*4,yerr = 0.00*Amax,linewidth=1,label="norm. difference",fmt ='o-',markersize=5,color="red")
# ax4.errorbar(np.radians(myt0),smooth(Adiff_abs,3)/8000,yerr = 0.00*Amax,linewidth=1,label="norm. difference",fmt ='o-',markersize=5,color="red")


# ax4.set_thetamin(0)
# ax4.set_thetamax(180)
# ax4.set_xticks(np.radians([0,30,60,90,120,150,180]))
# ax4.set_yticks(np.radians([0,0.5]))
# ax4.set_yticks(ax4.get_yticks()[::2])

# ax4.set_yticklabels([])
# ax4.set_ylim([0,0.5])
ax4.set_xlabel("rel. intensity change (norm.)")
# plt.grid(b=True, which='both', color='0.65', linestyle='-')
plt.legend()


plt.tight_layout()

stack01 = np.vstack((myt0_360,Amax_360,Aprepump_360,Apostpump_360,Apump_noise_360))
stack02 = np.vstack((myt0,Amax1,Amax2,Aprepump1,Aprepump2))


# np.savetxt("Polscan_H8_Smooth3_AnglePumpUnpump.txt",stack01)
# np.savetxt("Polscan_H8_Smooth3_AnglePump12Unpump12.txt",stack02)

# plt.savefig(("MoS2_scan03_PPVsPol"+"norm_abs"+".png"),dpi=(250))


#%% Build filepath, 

myspectra_pump = sorted(genpath.glob((r"*MoS2_03_pol*SPECTRA.txt")))
mywavelengths_pump = sorted(genpath.glob((r"*MoS2_03_pol*WAVELENGTHS.txt")))
mywedgepos_pump = sorted(genpath.glob((r"*MoS2_03_pol*POSITIONS.txt")))


# N_av = float(re.search(r'avg([-]?\d*.\d*)_', pump_name).group(1))
# T_integ = float(re.search(r'int([-]?\d*.\d*)ms', pump_name).group(1))
# Pol_angle = float(re.search(r'pol([-]?\d*.\d*)deg', pump_name).group(1))

pump_spec_int = []
pump_spec_int_H4 = []

pump_pos = []
pump_name = []
Pol_angle= []
N_files = len(myspectra_pump)
for ii in range(0,N_files):
    pump_spec = np.loadtxt((myspectra_pump[ii]))#*1000/166*360
    pump_pos.append(np.loadtxt((mywedgepos_pump[ii])))#*1000/166*360
    pump_wvl = np.loadtxt((mywavelengths_pump[ii]))
    pump_name = Path(myspectra_pump[ii]).parts[-1].replace(".txt","")
    print(pump_name)

    Pol_angle.append(float(re.search(r'pol([-]?\d*.\d*)deg', pump_name).group(1)))

    pump_spec_smth = SmoothLoop(pump_spec,10)
    
    rangeind0 = np.logical_and(pump_wvl>=(myharm[8]-myrange),pump_wvl<=(myharm[8]+myrange)) 
    rangeind_H4 = np.logical_and(pump_wvl>=(myharm[4]-myrange),pump_wvl<=(myharm[4]+myrange)) 

    pump_spec_int.append(np.sum(pump_spec_smth[:,rangeind0],1))
    pump_spec_int_H4.append(np.sum(pump_spec_smth[:,rangeind_H4],1))

pump_pos_act = 2*np.array(pump_pos)/constants.c*1e-3*1e15*(-1)
pump_spec_int = np.array(pump_spec_int)
pump_spec_int_H4 = np.array(pump_spec_int_H4)




plt.rcParams.update({'font.size': 24})
fig4, ax4 = plt.subplots(figsize=(10,6))
# ax4.pcolormesh(pump_wvl,pump_pos,np.log(pump_spec),shading='None')# shading='gouraud'
ax4.scatter(pump_pos_act[0,:],pump_spec_int[0,:]/max(pump_spec_int[0,:]))
ax4.scatter(pump_pos_act[1,:],pump_spec_int[1,:]/max(pump_spec_int[1,:]))

H4_line = (pump_spec_int_H4[0,:]/max(pump_spec_int_H4[0,:]))**2
H4_line2 = (pump_spec_int_H4[1,:]/max(pump_spec_int_H4[1,:]))**2


# ax4.scatter(pump_pos_act[0,:],mynorm(pump_spec_int[0,:]/pump_spec_int_H4[0,:]**2),color="k")
# ax4.scatter(pump_pos_act[0,:],smooth(mynorm(pump_spec_int[0,:]/pump_spec_int_H4[0,:]**2),5),color="b")

# ax4.scatter(pump_pos_act[0,:],mynorm(pump_spec_int[1,:]/pump_spec_int_H4[1,:]**2),color="gray")
# ax4.scatter(pump_pos_act[0,:],smooth(mynorm(pump_spec_int[1,:]/pump_spec_int_H4[1,:]**2),5),color="r")

H8_H4_0 = mynorm(pump_spec_int[0,:]/pump_spec_int_H4[0,:]**2)
H8_H4_1 = mynorm(pump_spec_int[1,:]/pump_spec_int_H4[1,:]**2)

H8_H8_11 = smooth(mynorm(pump_spec_int[1,:]/pump_spec_int[0,:]),1)


ax4.scatter(pump_pos_act[0,:],H8_H4_0,color="g")
ax4.scatter(pump_pos_act[0,:],H8_H4_1,color="k")

ax4.plot(pump_pos_act[0,:],H8_H4_1/H8_H4_0,color="r")
ax4.plot(pump_pos_act[0,:],H8_H8_11,color="b")

# ax4.plot(pump_pos_act[0,:],smooth(pump_spec_int[0,:]/max(pump_spec_int[0,:])/H4_line,4))

# ax4.plot(pump_pos_act[1,:],smooth(pump_spec_int[1,:]/pump_spec_int[0,:],1)*5,"s--",color="k")
# ax4.plot(pump_pos_act[1,:],smooth(pump_spec_int[1,:]/pump_spec_int[0,:],4)*5)

# ax4.plot(pump_pos_act[1,:],smooth(pump_spec_int_H4[1,:]/pump_spec_int_H4[0,:],2))


ax4.set_xlabel("Pump-probe delay (fs)")
ax4.set_ylabel(r"Harmonic intensity (counts)")
# ax4.set_ylim(0,100)
# stats = ("N_av "+str(N_av)+", T_int "+str(T_integ))
# ax4.legend(stats) 
plt.tight_layout()



# plt.savefig(("MoS2_scan03_PPdelayVSPOL"+"av_Lineouts_H8_1d"+".png"),dpi=(250))


#%%
plt.rcParams.update({'font.size': 24})
fig5, ax5 = plt.subplots(nrows=2, ncols=1,figsize=(10,6),sharex=True)
# fig5.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

# ax5[0].scatter(pump_pos_act[0,:],(pump_spec_int[0,:]/max(pump_spec_int[0,:])),color="k",label="Pol$_{max}$")
# ax5[0].scatter(pump_pos_act[1,:],(pump_spec_int[1,:]/max(pump_spec_int[1,:])),color="b",label="Pol$_{min}$")

# ax5[0].plot(pump_pos_act[1,:],H8_H4_1/H8_H4_0,"o",color="gray",alpha=1)
# ax5[0].plot(pump_pos_act[1,:],smooth(H8_H4_0,3),color="g",label="Max")
# ax5[0].plot(pump_pos_act[1,:],smooth(H8_H4_1,3),color="b",label="Min")
ax5[0].plot(pump_pos_act[1,:],smooth(H8_H4_1/H8_H4_0,1),"o:",color="gray",label="Min/Max")
ax5[0].plot(pump_pos_act[1,:],smooth(H8_H4_1/H8_H4_0,3),color="k",linewidth=2)

# ax5[0].plot(pump_pos_act[0,:],H8_H8_11,"o",color="k")
# ax5[0].plot(pump_pos_act[0,:],smooth(H8_H8_11,3),color="b")

ax5[0].set_ylabel("H8/H4")


# ax5[0].set_ylabel("H2 intensity (norm.)")
ax5[0].legend(fontsize = 14)


ax5[1].errorbar(-1*probe_pos_act,(H8_ellip/min(H8_ellip))-1,yerr=2*err_tot*H8_ellip/min(H8_ellip),fmt='s',color="r",capsize=3) #ftm = "--o" ,markersize=8

ax5[1].set_xlabel("Pump-probe delay (fs)")
ax5[1].set_xlim([-450,1000])
ax5[0].grid(True)
ax5[1].grid(True)
ax5[1].set_ylabel("rel. P$_{min}$/P$_{max}$")


plt.tight_layout()


# plt.savefig(("MoS2_scan03_PPdelayVSPOL"+"av_Lineouts_H8_all3"+".png"),dpi=(250))
