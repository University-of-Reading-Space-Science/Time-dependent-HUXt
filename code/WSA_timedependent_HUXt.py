# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:14:03 2024

@author: mathewjowens
"""


import os
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
import numpy as np
import astropy.units as u
import pandas as pd
import imageio
from moviepy.editor import ImageSequenceClip
import glob
from scipy import interpolate
import h5py


import helio_time as htime
import helio_coords as hcoords
import mplot as mplot
import system as system

import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA



#set plot defaults
system.plot_defaults()

wsa_run = 'AGONG' 
#wsa_run = 'MO_API'
#wsa_run = 'GONGZ'



ndays = 1  # download coronal solutions up to this many days prior to the forecast date

firstdate = datetime.datetime(2020,1,1,0)
finaldate = datetime.datetime(2020,12,31,23)

deacc = True #Whether to deaccelerate the WSA maps froms 215 to 21.5 rS
input_res_days = 0.1 #interpolated WSA time resolution (days) for the time-dependent runs.
bufferdays_td = 4 #number of days before the forecasttime to start the time-dependent runs


load_maps_now = True
single_run_now = True
single_anim_now = False
plot_map_frames = True
enhanced_steady_timedependent_anim_now = True

save_figs_now = False

#set plot defaults
system.plot_defaults()

fontsize = 12
daysec = 24*60*60

#set up directories
cwd = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(cwd)
figdir = os.path.join(root,'output')
reduced_data_dir = os.path.join(root,'data')

if load_maps_now:
    Rstr = '000'
    wsa_run = 'AGONG' #'MO_API'
    datarootdir = os.path.join( os.getenv('DBOX'),'Data')
    datadir = os.path.join(datarootdir,'WSA_' + wsa_run)

# <codecell> load in the OMNI data
from sunpy.net import Fido
from sunpy.net import attrs
from sunpy.timeseries import TimeSeries

startdate = firstdate + datetime.timedelta(days=4) #datetime.datetime(2023,1,3,0)
stopdate = finaldate#datetime.datetime(2024,1,3,0)

# Download the 1hr OMNI data from CDAweb
trange = attrs.Time(startdate, stopdate)
dataset = attrs.cdaweb.Dataset('OMNI2_H0_MRG1HR')
result = Fido.search(trange, dataset)
downloaded_files = Fido.fetch(result)

# Import the OMNI data
omni = TimeSeries(downloaded_files, concatenate=True)
data = omni.to_dataframe()

# Set invalid data points to NaN
id_bad = data['V'] == 9999.0
data.loc[id_bad, 'V'] = np.NaN

# Create a datetime column
data['datetime'] = data.index
data['mjd'] = htime.datetime2mjd(data['datetime'].to_numpy())

# <codecell> first produce a summary plot of the Owens 2005 results
data_str = """
1995    108.55  109.53  112.40  108.57  112.40  110.43  115.60  
1996    86.99   87.35   84.35   83.53   82.25   82.04   84.13   
1997    75.65   73.91   74.33   74.71   73.36   77.02   75.47  
1998    96.58   92.45   90.17   90.49   95.34   99.77   100.43  
1999    89.33   90.08   90.08   93.37   95.70   95.18   94.35   
2000    94.07   93.36   91.85   93.85   94.88   97.52   96.24  
2001    102.07  101.73  99.89   101.64  103.70  100.82  102.64  
2002    84.10   85.73   82.92   85.06   85.64   86.27   88.42   
"""

# Split the data into lines and then split each line into individual values
lines = data_str.strip().split('\n')
data_values = [line.split() for line in lines]

# Convert the values into a NumPy array
wsa_mse = np.array(data_values, dtype=float)
xvals = np.arange(1,8,1)

#normalise the data
wsa_mse_norm = wsa_mse.copy()
for n in range(0, len(wsa_mse[:,0])):
    thismean = np.nanmean(wsa_mse[n,1:])
    wsa_mse_norm[n,1:] = wsa_mse_norm[n,1:]/thismean


plot_notch=False


fig = plt.figure(figsize = (10,10))

ax = plt.subplot(2,2,1)
for n in range(0, len(wsa_mse[:,0])):
    plt.plot(xvals, wsa_mse[n,1:], label = str(int(wsa_mse[n,0])))
ax.legend(loc = 'upper left')
ax.set_ylabel(r'Mean-square error, MSE $[km^2 s^{-2}]$')
ax.text(0.92, 0.03, '(a)', transform=plt.gca().transAxes, fontsize=fontsize)

ax = plt.subplot(2,2,2)
box = ax.boxplot(wsa_mse[:,1:],
               notch=plot_notch, patch_artist=True,showfliers=False,whis=1.5)
ax.text(0.92, 0.03, '(b)', transform=plt.gca().transAxes, fontsize=fontsize)

ax = plt.subplot(2,2,3)
for n in range(0, len(wsa_mse[:,0])):
    plt.plot(xvals, wsa_mse_norm[n,1:], label = str(int(wsa_mse[n,0])))
#ax.legend(loc = 'upper left')
ax.set_ylabel('Normalised MSE')
ax.set_xlabel('Forecast lead time [days]')
ax.text(0.92, 0.03, '(c)', transform=plt.gca().transAxes, fontsize=fontsize)

ax = plt.subplot(2,2,4)
box = ax.boxplot(wsa_mse_norm[:,1:],
               notch=plot_notch, patch_artist=True,showfliers=False,whis=1.5)
ax.set_xlabel('Forecast lead time [days]')
ax.text(0.92, 0.03, '(d)', transform=plt.gca().transAxes, fontsize=fontsize)


if save_figs_now:
    fig.savefig(os.path.join(figdir,'wsa2005summary.pdf'))
    
    
# <codecell> plot a few example maps

start_date = datetime.datetime(2020,12, 12, 0)

fig = plt.figure(figsize=(10,6))

gs = gridspec.GridSpec(2,3)


for i in range(0,3):
    
    plot_date = start_date + datetime.timedelta(days=i)
    cr, cr_lon_init = Hin.datetime2huxtinputs(plot_date)
    
    
    
    year = str(plot_date.year)
    month = str(plot_date.month)
    day = str(plot_date.day)
    
    #get the MJD
    mjd = htime.datetime2mjd(plot_date)
    
     
    #get Earth lat
    Ecoords = hcoords.carringtonlatlong_earth(mjd)
    E_lat = np.pi/2 - Ecoords[0][0]
    
    
    
    if load_maps_now:
        #load data from the WSA archive
    
        #create the expected filename
        if wsa_run == 'MO_API':
            filename = 'models%2Fenlil%2F' + year +'%2F' + month +'%2F' + day + '%2F0%2Fwsa.gong.fits'
        elif wsa_run == 'AGONG':
            filename = ('vel_' + year + plot_date.strftime('%m') + plot_date.strftime('%d') 
            + '1200R' + Rstr + '_agong.fits')
        elif wsa_run == 'GONGZ':
            pattern = ('vel_' + year + plot_date.strftime('%m') + plot_date.strftime('%d') 
            + '*R000_gongz.fits')
            matching_files = glob.glob(os.path.join(datadir, pattern))
            filename = matching_files[0]
            
        filepath = os.path.join(datadir, filename)
        
        print(filepath)
        
    else:
        #load the three maps provided in the github repo
        filename = ('vel_' + year + plot_date.strftime('%m') + plot_date.strftime('%d') 
        + '1200R000_agong.fits')
        
        filepath = os.path.join(reduced_data_dir, filename)

    
    if os.path.exists(filepath):
        
        
        #load and plot the speed map
        #=========================================
        ax = fig.add_subplot(gs[0, i])
        
        
        vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
            = Hin.get_WSA_maps(filepath)
            
        # deaccelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
        if deacc:
            vr_map_deacc = vr_map.copy()
            for nlat in range(1, len(vr_lats)):
                vr_map_deacc[nlat, :], lon_temp = Hin.map_v_inwards(vr_map[nlat, :], 215 * u.solRad,
                                                                    vr_longs, 21.5* u.solRad)
            vr_map = vr_map_deacc
            
        E_lats = E_lat * (vr_longs.value * 0 + 1)
        
        fig, ax, axins, pc = mplot.plotspeedmap(vr_map, vr_longs, vr_lats, 
                           fig = fig, ax = ax, plot_colourbar = False, plot_sinelat = True)
            
        ax.plot(vr_longs*180/np.pi, np.sin(E_lats), 'k--',label = 'Earth')
        ax.plot(vr_longs*180/np.pi, E_lats*0, 'k')
        ax.plot(Ecoords[0][1]*180/np.pi, np.pi/2 - Ecoords[0][0], 'ro')
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.axes.xaxis.set_ticklabels([])
        
        ax.set_title(plot_date.strftime("%Y-%m-%d"))
        
        if  i == 2:
            #add the colorbar
            axins = inset_axes(ax,
                                width="100%",  # width = 50% of parent_bbox width
                                height="100%",  # height : 5%
                                loc='upper right',
                                bbox_to_anchor=(1.05, 0.08, 0.12, 0.8),
                                bbox_transform=ax.transAxes,
                                borderpad=0,)
            cb = fig.colorbar(pc, cax = axins, orientation = 'vertical',  pad = -0.1)
            cb.ax.xaxis.set_ticks_position("top")
            ax.text(1.05,0.95,r'$V_{SW}$ [km/s]' , 
                    fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
        if i >0:
            ax.axes.yaxis.set_ticklabels([])
        else:
            ax.set_ylabel('Latitude [deg]')
                
        
        
        #extract and plot the values at earth lat
        #=========================================
        ax = fig.add_subplot(gs[1, i])
        
        v_in = Hin.get_WSA_long_profile(filepath, lat=E_lat*u.rad)
        
        if deacc:
            # deaccelerate them?
            v_in, lon_temp = Hin.map_v_inwards(v_in, 215 * u.solRad, vr_longs,  21.5 * u.solRad)
        
        ax.plot(vr_longs*180/np.pi, v_in,'k')
        ax.axes.xaxis.set_ticks([0,90,180,270,360])
        
        if i==1:
            ax.set_xlabel('Carrington Longitude [deg]')
        if i ==0:
            ax.set_ylabel(r'V [km s$^{-1}$]')
        else:
            ax.axes.yaxis.set_ticklabels([])
        ax.set_ylim((200, 650))
        ax.set_xlim((0, 360))
        
        ax.plot([Ecoords[0][1]*180/np.pi, Ecoords[0][1]*180/np.pi], [200,650],  'r')

        plt.tight_layout()    
        #ax.legend()
     
        
if save_figs_now:
    fig.savefig(os.path.join(figdir,'Example3maps.pdf'))       
        


# <codecell> plot a few example maps - vertical

start_date = datetime.datetime(2020,12, 12, 0)
nplot = 3
heeq = True



fig = plt.figure(figsize=(6,10))
gs = gridspec.GridSpec(nplot,2)


for i in range(0,nplot):
    plot_date = start_date + datetime.timedelta(days=i)
    cr, cr_lon_init = Hin.datetime2huxtinputs(plot_date)
    
    
    
    year = str(plot_date.year)
    month = str(plot_date.month)
    day = str(plot_date.day)
    
    if load_maps_now:
        #load data from the WSA archive
    
        #create the expected filename
        if wsa_run == 'MO_API':
            filename = 'models%2Fenlil%2F' + year +'%2F' + month +'%2F' + day + '%2F0%2Fwsa.gong.fits'
        elif wsa_run == 'AGONG':
            filename = ('vel_' + year + plot_date.strftime('%m') + plot_date.strftime('%d') 
            + '1200R' + Rstr + '_agong.fits')
        elif wsa_run == 'GONGZ':
            pattern = ('vel_' + year + plot_date.strftime('%m') + plot_date.strftime('%d') 
            + '*R000_gongz.fits')
            matching_files = glob.glob(os.path.join(datadir, pattern))
            filename = matching_files[0]
            
        filepath = os.path.join(datadir, filename)
        
        print
    else:
        #load the three maps provided in the github repo
        filename = ('vel_' + year + plot_date.strftime('%m') + plot_date.strftime('%d') 
        + '1200R000_agong.fits')
        
        filepath = os.path.join(reduced_data_dir, filename)
    
    #get the MJD
    mjd = htime.datetime2mjd(plot_date)
    
     
    #get Earth lat
    Ecoords = hcoords.carringtonlatlong_earth(mjd)
    E_lat = np.pi/2 - Ecoords[0][0]
    
    
    if os.path.exists(filepath):
        
        
        #load and plot the speed map
        #=========================================
        ax = fig.add_subplot(gs[i,0])
        
        
        vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
            = Hin.get_WSA_maps(filepath)
            
        # deaccelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
        if deacc:
            vr_map_deacc = vr_map.copy()
            for nlat in range(1, len(vr_lats)):
                vr_map_deacc[nlat, :], lon_temp = Hin.map_v_inwards(vr_map[nlat, :], 215 * u.solRad,
                                                                    vr_longs, 21.5* u.solRad)
            vr_map = vr_map_deacc
            
        ylims = [-1, 1]
        yticks = [-1,-0.5,0,0.5,1]
        
        if heeq:
            # rotate the maps so they are in the HEEQ frame
            vr_map_HEEQ = np.empty(vr_map.shape)
            for nlat in range(0, len(vr_lats)):
                interp = interpolate.interp1d(vr_longs, vr_map[nlat, :], kind="nearest",
                                              fill_value="extrapolate")
                vr_map_HEEQ[nlat, :] = interp(H._zerototwopi_(vr_longs - Ecoords[0][1]*u.rad + np.pi*u.rad))
            vr_longs_HEEQ = vr_longs - np.pi*u.rad

          
        if heeq:
            pc = ax.pcolor(vr_longs_HEEQ.value*180/np.pi, np.sin(vr_lats.value), vr_map_HEEQ, 
                    shading='auto',vmin=250, vmax=650)
            ax.set_xlim([-180,180])
            ax.set_xlabel('HEEQ Longitude [deg]')
            ax.plot(0, np.pi/2 - Ecoords[0][0], 'ro')
            ax.axes.xaxis.set_ticks([-180,-90,0,90,180])
        else:
            pc = ax.pcolor(vr_longs.value*180/np.pi, np.sin(vr_lats.value), vr_map.value, 
                    shading='auto',vmin=250, vmax=650)
            ax.set_xlim([0,360])
            ax.set_xlabel('Carrington Longitude [deg]')
            ax.plot(cr_lon_init, np.pi/2 - Ecoords[0][0], 'ro')
            ax.axes.xaxis.set_ticks([0,90,180,270,360])
        
        ax.set_ylabel('Sine latitude [deg]')
        ax.set_ylim(ylims); 
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.axes.xaxis.set_ticklabels([])
        
        ax.set_title(plot_date.strftime("%Y-%m-%d"))
        
        if  i == 2:
            #add the colorbar
            axins = inset_axes(ax,
                                width="100%",  # width = 50% of parent_bbox width
                                height="100%",  # height : 5%
                                loc='upper right',
                                bbox_to_anchor=(1.05, 0.08, 0.12, 0.8),
                                bbox_transform=ax.transAxes,
                                borderpad=0,)
            cb = fig.colorbar(pc, cax = axins, orientation = 'vertical',  pad = -0.1)
            cb.ax.xaxis.set_ticks_position("top")
            ax.text(1.05,0.95,r'$V_{SW}$ [km/s]' , 
                    fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
        if i >0:
            ax.axes.yaxis.set_ticklabels([])
        else:
            ax.set_ylabel('Latitude [deg]')
                
        
        
        #extract and plot the values at earth lat
        #=========================================
        ax = fig.add_subplot(gs[i, 1])
        
        v_in = Hin.get_WSA_long_profile(filepath, lat=E_lat*u.rad)
        
        if deacc:
            # deaccelerate them?
            v_in, lon_temp = Hin.map_v_inwards(v_in, 215 * u.solRad, vr_longs,  21.5 * u.solRad)
            
        if heeq:
            interp = interpolate.interp1d(vr_longs, v_in, kind="nearest",
                                          fill_value="extrapolate")
            v_in_HEEQ = interp(H._zerototwopi_(vr_longs - Ecoords[0][1]*u.rad + np.pi*u.rad))
        
        
        
        

        if heeq:
            ax.plot(vr_longs_HEEQ*180/np.pi, v_in_HEEQ,'k')
            ax.axes.xaxis.set_ticks([-180,-90,0,90,180])
            ax.set_xlim((-180, 180))
        else:
            ax.plot(vr_longs*180/np.pi, v_in,'k')
            ax.axes.xaxis.set_ticks([0,90,180,270,360])
            ax.set_xlim((0, 360))
            
            
            
            
        if i==1:
            ax.set_xlabel('Carrington Longitude [deg]')
        if i ==0:
            ax.set_ylabel(r'V [km s$^{-1}$]')
        else:
            ax.axes.yaxis.set_ticklabels([])
        ax.set_ylim((200, 650))

        
        ax.plot([Ecoords[0][1]*180/np.pi, Ecoords[0][1]*180/np.pi], [200,650],  'r')

        plt.tight_layout()    
        #ax.legend()
     
        
if save_figs_now:
    fig.savefig(os.path.join(figdir,'Example3maps_vert.pdf'))       
        


# <codecell> load all the data and extract the Earth lat values

h5filename = 'WSA_Earth_lat_properties_ADAPT_' + Rstr + '.h5'
h5filepath = os.path.join(reduced_data_dir, h5filename)



if load_maps_now:
    
    thisdate = firstdate

    count = 1
    vlong_list = []
    brlong_list = []
    mjd_list = []
    while thisdate <= finaldate:
        
        year = str(thisdate.year)
        month = str(thisdate.month)
        day = str(thisdate.day)
        
        #create the expected filename
        if wsa_run == 'MO_API':
            filename = 'models%2Fenlil%2F' + year +'%2F' + month +'%2F' + day + '%2F0%2Fwsa.gong.fits'
        elif wsa_run == 'AGONG':
            filename = ('vel_' + year + thisdate.strftime('%m') + thisdate.strftime('%d') 
            + '1200R' + Rstr + '_agong.fits')
        elif wsa_run == 'GONGZ':
            pattern = ('vel_' + year + thisdate.strftime('%m') + thisdate.strftime('%d') 
            + '*R000_gongz.fits')
            matching_files = glob.glob(os.path.join(datadir, pattern))
            if len(matching_files) > 0:
                filename = matching_files[0]
            else:
                filename = pattern
        filepath = os.path.join(datadir, filename)
        
        #get the MJD
        mjd = htime.datetime2mjd(thisdate)
        mjd_list.append(mjd)
        
        #get Earth lat
        Ecoords = hcoords.carringtonlatlong_earth(mjd)
        E_lat = np.pi/2 - Ecoords[0][0]
        
        
        
        if os.path.exists(filepath):
            vr_map, vr_longs, vr_lats, br_map, br_longs, br_lats, cr_fits \
                = Hin.get_WSA_maps(filepath)
                
            # deaccelerate the WSA map from 1-AU calibrated speeds to expected 21.5 rS values
            if deacc:
                vr_map_deacc = vr_map.copy()
                for nlat in range(1, len(vr_lats)):
                    vr_map_deacc[nlat, :], lon_temp = Hin.map_v_inwards(vr_map[nlat, :], 215 * u.solRad,
                                                                        vr_longs, 21.5* u.solRad)
                vr_map = vr_map_deacc
                
            E_lats = E_lat * (vr_longs.value * 0 + 1)
            
            if plot_map_frames:
                #plot it
                fig, ax, axins, pc = mplot.plotspeedmap(vr_map, vr_longs, vr_lats)
                
                ax.plot(vr_longs*180/np.pi,E_lats*180/np.pi,'k--',label = 'Earth')
                ax.plot(vr_longs*180/np.pi,E_lats*0,'k')
                ax.legend()
                ax.text(0.95,-0.13, thisdate.strftime("%Y-%m-%d"),  
                         fontsize = 14, transform=ax.transAxes)
                
                #save map image
                formatted_number = "{:03d}".format(count)
                map_image = os.path.join(datadir,'frames', 'frame' + formatted_number + '.png')
                plt.savefig(map_image)
                
                plt.close(fig)
                
            #get the Earth lat slice
            v_in = Hin.get_WSA_long_profile(filepath, lat=E_lat*u.rad)
            if deacc:
                # deaccelerate them?
                v_in, lon_temp = Hin.map_v_inwards(v_in, 215 * u.solRad, vr_longs,  21.5 * u.solRad)
            br_in = Hin.get_WSA_br_long_profile(filepath, lat=E_lat*u.rad)
             
            #store the data
            vlong_list.append(v_in)
            brlong_list.append(br_in)
            
        else:
            print(filepath + '; not found')
        
        
        
        #advance the date
        thisdate = thisdate + datetime.timedelta(days=ndays)
        count = count + 1
      
     
      
    #convert to arrays    
    vlongs_1d = np.array(vlong_list).T
    brlongs_1d = np.array(brlong_list).T
    mjds_1d = np.array(mjd_list)
    
    #save these reduced data as a h5 file
    with h5py.File(h5filepath, 'w') as h5f:
        h5f.create_dataset('vlongs_1d', data=vlongs_1d)
        h5f.create_dataset('brlongs_1d', data=brlongs_1d)
        h5f.create_dataset('mjds_1d', data=mjds_1d)
        #h5f.create_dataset('time_1d', data=time_1d)
    
        
else:
    #load the reduced data from the h5 file
    print('reading reduced WSA data from h5 file: '+ h5filepath)
    
    with h5py.File(h5filepath, 'r') as h5f:
        vlongs_1d = h5f['vlongs_1d'][:]
        brlongs_1d = h5f['brlongs_1d'][:]
        mjds_1d = h5f['mjds_1d'][:]
        
        
time_1d = htime.mjd2datetime(mjds_1d)
n_longs = len(vlongs_1d[:,0])


#increase the time resolution of the vlongs for the time-dependent runs
mjds = np.arange(mjds_1d[0], mjds_1d[-1], input_res_days)
time = htime.mjd2datetime(mjds)
vlongs = np.ones((n_longs, len(mjds)))
brlongs = np.ones((n_longs, len(mjds)))
for n in range(0, n_longs):
    vlongs[n,:] = np.interp(mjds, mjds_1d, vlongs_1d[n,:])
    brlongs[n,:] = np.interp(mjds, mjds_1d, brlongs_1d[n,:])


#plot the CarrLon - time Vin
fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,1,1)
pc = ax.pcolor(time_1d, vr_longs.value*180/np.pi, vlongs_1d, 
            shading='auto',vmin=250, vmax=650)
ax.set_ylabel('Carrington Longitude [deg]')
ax.axes.yaxis.set_ticks([0,90,180,270,360])
ax.text(1.05,1.05,r'$V_{SW}$ [km/s]' , 
        fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
cbar = plt.colorbar(pc, ax=ax)
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Date (2020)')

# ax = plt.subplot(2,1,2)
# pc = ax.pcolor(time, br_longs.value*180/np.pi, brlongs, 
#             shading='auto')
# ax.set_ylabel('Carrington Longitude [deg]')
# ax.axes.yaxis.set_ticks([0,90,180,270,360])
# ax.text(0.15,1.05,r'$Br [nT]' , 
#         fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
# cbar = plt.colorbar(pc, ax=ax)

if save_figs_now:
    fig.savefig(os.path.join(figdir,'WSAsummary.pdf'))


# <codecell> run HUXt with the time-dependent boundary condition
if single_run_now:
    runstart = firstdate#datetime.datetime(2023,1,1)
    smjd = htime.datetime2mjd(runstart)
    
    runend = finaldate#datetime.datetime(2023,12,31)
    fmjd = htime.datetime2mjd(runend)
    
    simtime = (runend-runstart).days * u.day
      
    
    #set up the model, with (optional) time-dependent bpol boundary conditions
    model_td = Hin.set_time_dependent_boundary(vlongs, mjds, runstart, simtime, 
                                            r_min=21.5 *u.solRad, r_max=250*u.solRad,
                                            #bgrid_Carr = brlongs.T, 
                                            dt_scale=1, latitude=0*u.deg,
                                            frame = 'synodic')
    
    #trace a bunch of field lines from a range of evenly spaced Carrington longitudes
    dlon = (20*u.deg).to(u.rad).value
    lon_grid = np.arange(dlon/2, 2*np.pi-dlon/2 + 0.0001, dlon)*u.rad
    
    #give the streakline footpoints (in Carr long) to the solve method
    #model_td.solve([], streak_carr = lon_grid)
    model_td.solve([])
    
    HA.plot(model_td, 12*u.day)
    
    
    #get the Earth timeseries
    #get conditions at Earth
    td_Earth_ts = HA.get_observer_timeseries(model_td, observer='Earth')
    #convert the model time to MJD
    td_tim_mjd = model_td.time_init.mjd + model_td.time_out.value/daysec
    
    if single_anim_now:

        HA.animate(model_td, tag='HUXt_WSA_Jan2023_time_dependent', duration = 60, fps = 20) # This takes about two minutes.





# <codecell> Enhanced steady-state and time-dependent comaprison movie



if enhanced_steady_timedependent_anim_now:
    
    runstart = firstdate
    runstart = datetime.datetime(2020,8,1)
    smjd = htime.datetime2mjd(runstart)
    
    runend = finaldate
    runend = datetime.datetime(2020,9,1)
    fmjd = htime.datetime2mjd(runend)
    
    
    
    #generate side-by-side images of steady-state and time-dependent runs
    #=====================================================================
    #delete existing images
    # List all files in the directory
    # Directory containing the PNG files
    png_dir = os.path.join(datadir,'frames')
    files = os.listdir(png_dir)
    
    # Iterate through the files and delete .png files
    for file in files:
        if file.endswith('.png'):
            os.remove(os.path.join(png_dir, file))
        
    #generate one steady state run and from that produce mutliple frams
    count = 1
    fracs = np.arange(0,1,1/24)
    #fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    plot_mjd = smjd
    
    while plot_mjd <= fmjd:
        plot_date = htime.mjd2datetime(plot_mjd).item()
        
        #time from start of the TD run
        dmjd = plot_mjd - model_td.time_init.mjd
        
        
        print('Processing ' + plot_date.strftime('%Y-%m-%d'))
        
        
        #run the steady-state model
        #===========================
        #get the CR num and cr_lon_init that this corresponds to
        cr, cr_lon_init = Hin.datetime2huxtinputs(plot_date)
        #find the map with this date
        id_t_1d = np.argmin(np.abs(time_1d - plot_date))
        #set up a HUXt run with this boundary condition
        model_ss = H.HUXt(v_boundary=vlongs_1d[:, id_t_1d]*u.km/u.s, cr_num=cr, cr_lon_init=cr_lon_init,
                       simtime=1*u.day, dt_scale=1, r_min = 21.5*u.solRad,
                       r_max = 250*u.solRad, frame='synodic')
        model_ss.solve([])
        
        
        for plot_day_frac in fracs:

                
            thismjd = plot_mjd + plot_day_frac
            Earth_r = (hcoords.earth_R(thismjd)*u.km).to(u.solRad).value
            
            fig = plt.figure(figsize=(12, 5))
            #fig.subplots_adjust(left=0.01, bottom=0.17, right=0.99, top=0.99)
            
            
            #plot the SS model ecliptic place
            #================================
            ax = plt.subplot(131,  projection='polar')
            HA.plot(model_ss, plot_day_frac*u.day, 
                    fighandle=fig, axhandle=ax, minimalplot=True,  trace_earth_connection=True)
            
   
            ax.set_title('Steady state')
            ax.plot(0, Earth_r,'ko')
            
            
            #plot the TD model ecliptic place
            #================================
            ax = plt.subplot(132,  projection='polar')
            
            HA.plot(model_td, (dmjd + plot_day_frac)*u.day, 
                    fighandle=fig, axhandle=ax, minimalplot=True, trace_earth_connection=True) 

            ax.set_title('Time dependent')
            ax.plot(0, Earth_r,'ko')
            
            hr = int(plot_day_frac*24)
            if hr < 10:
                hr_str = '0' + str(hr) + ':00'
            else:
                hr_str = str(hr) + ':00'
            ax.text(0.34,1.3,plot_date.strftime('%Y-%m-%d') + ' ' + hr_str, 
                    fontsize = 14,
                    transform=ax.transAxes, backgroundcolor = 'w')
            
            #add the colorbar
            #=================
            cmap = plt.get_cmap('viridis')
            norm = mcolors.Normalize(vmin=200, vmax=800)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # You can pass an empty list here

            
            #add the colorbar
            axins = inset_axes(ax,
                                width="100%",  # width = 50% of parent_bbox width
                                height="100%",  # height : 5%
                                loc='upper right',
                                bbox_to_anchor=(-0.70, -0.18, 1.2, 0.08),
                                bbox_transform=ax.transAxes,
                                borderpad=0,)
            cb = fig.colorbar(sm, cax = axins, orientation = 'horizontal',  pad = -0.1)
            cb.ax.xaxis.set_ticks_position("bottom")
            ax.text(-1.05,-0.16,r'$V_{SW}$ [km/s]' , 
                    fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
            
            #plot the speed along the E-S line
            #=================================
            
            ax = plt.subplot(133)
            ax.set_position([0.68, 0.17, 0.25, 0.6])
            
            model_ts_mjd = model_ss.time_init.mjd + model_ss.time_out.value/daysec
            id_t = np.argmin(np.abs(model_ts_mjd - thismjd))
            vr_ss = model_ss.v_grid[id_t, :, 0]
            
            model_ts_mjd = model_td.time_init.mjd + model_td.time_out.value/daysec
            id_t = np.argmin(np.abs(model_ts_mjd - thismjd))
            vr_td = model_td.v_grid[id_t, :, 0]
            
            r = model_td.r.to(u.solRad)
            
            ax.plot(r, vr_ss, 'k', label ='Steady state')
            ax.plot(r, vr_td, 'r' , label ='Time dependent')
            ax.set_ylim([250,750])
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
            ax.set_ylabel(r'$V_{SW}$ [km/s]')
            ax.get_xaxis().set_ticks([21.5, 100, 215])
            
            #get the OMNI speed for this time
            mask = (data['mjd'] >= thismjd - 0.1) & (data['mjd'] < thismjd + 0.1)
            vavg = np.nanmean(data.loc[mask,'V'])
            vmax = np.nanmax(data.loc[mask,'V'])
            vmin = np.nanmin(data.loc[mask,'V'])
            ax.plot(Earth_r, vavg, 'bo', label='Observed')
            ax.plot([Earth_r,Earth_r],[vmin,vmax],'b')
            ax.axes.yaxis.set_ticks([300, 400, 500, 600, 700])
            ax.legend(loc = 'upper left')
            ax.set_xlabel(r'Heliocentric distance [$r_S$]')
            ax.set_title('Sun-Earth line')
            
            
            
            
            #save the plot
            formatted_number = "{:04d}".format(count)
            fig_image = os.path.join(datadir,'frames', 'huxt' + formatted_number + '.png')
            plt.savefig(fig_image)
            
            plt.close('all')
            
            count = count + 1
        
        plot_mjd = plot_mjd + 1.0
    
    
    # List the PNG files
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png')])
    
    # Read PNG files into a list of images
    images = [imageio.imread(os.path.join(png_dir, f)) for f in png_files]
    
    # Create a video clip from the images
    clip = ImageSequenceClip(images, fps=24)  # Change fps as needed
    
    # Write the video clip to an MP4 file
    output_file = os.path.join(datadir,'frames',"Enhanced_SS_TD_comparison.mp4")
    clip.write_videofile(output_file, codec='libx264', fps=20) 


# <codecell> plot some example frames

#start_date = datetime.datetime(2023,1, 16)

start_date = datetime.datetime(2020,9, 6, 0)

fig = plt.figure(figsize=(10,12))

gs = gridspec.GridSpec(3, 3)

if single_run_now:
    for i in range(0,3):
        ax = fig.add_subplot(gs[i, 0], projection='polar')
        
        plot_date = start_date + datetime.timedelta(days=i)
        plotmjd = htime.datetime2mjd(plot_date)
        Earth_r = (hcoords.earth_R(plotmjd)*u.km).to(u.solRad).value
        
        #plot the steady-state solution
        #==============================
        cr, cr_lon_init = Hin.datetime2huxtinputs(plot_date)
        #find the map with this date
        id_t_1d = np.argmin(np.abs(time_1d - plot_date.replace(hour = 0)))
        #set up a HUXt run with this boundary condition
        model_ss = H.HUXt(v_boundary=vlongs_1d[:, id_t_1d]*u.km/u.s, cr_num=cr, cr_lon_init=cr_lon_init,
                       simtime=1*u.day, dt_scale=1, 
                       r_min = 21.5*u.solRad, r_max=250*u.solRad, frame='synodic')
        model_ss.solve([])
        HA.plot(model_ss, 0*u.day, 
                fighandle=fig, axhandle=ax, minimalplot=True,  trace_earth_connection=True)
        ax.set_ylabel(plot_date.strftime('%Y-%m-%d'), fontsize = 14)
        if i == 0:
            ax.set_title('Steady state')
            
        ax.plot(0, Earth_r,'ko')
        
        #plot the time-dependent solution
        #==============================
        ax = fig.add_subplot(gs[i, 1], projection='polar')
        run_days = plotmjd - model_td.time_init.mjd
        HA.plot(model_td, run_days*u.day, 
                fighandle=fig, axhandle=ax, minimalplot=True, trace_earth_connection=True) 
        if i == 0:
            ax.set_title('Time dependent')
        ax.plot(0, Earth_r,'ko')
        
        
        if i ==2:
            #add the colorbar
            #=================
            cmap = plt.get_cmap('viridis')
            norm = mcolors.Normalize(vmin=200, vmax=800)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # You can pass an empty list here

            
            #add the colorbar
            axins = inset_axes(ax,
                                width="100%",  # width = 50% of parent_bbox width
                                height="100%",  # height : 5%
                                loc='upper right',
                                bbox_to_anchor=(-0.70, -0.18, 1.2, 0.08),
                                bbox_transform=ax.transAxes,
                                borderpad=0,)
            cb = fig.colorbar(sm, cax = axins, orientation = 'horizontal',  pad = -0.1)
            cb.ax.xaxis.set_ticks_position("bottom")
            ax.text(-1.09,-0.16,r'$V_{SW}$ [km/s]' , 
                    fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
        
        #plot the speed along the E-S line
        #=================================
        
        ax = fig.add_subplot(gs[i, 2:])
        
        vr_ss = model_ss.v_grid[0, :, 0]
        
        model_ts_mjd = model_td.time_init.mjd + model_td.time_out.value/daysec
        id_t = np.argmin(np.abs(model_ts_mjd - plotmjd))
        vr_td = model_td.v_grid[id_t, :, 0]
        
        r = model_td.r.to(u.solRad)
        
        ax.plot(r, vr_ss, 'k', label ='Steady state')
        ax.plot(r, vr_td, 'r' , label ='Time dependent')
        ax.set_ylim([250,700])
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(r'$V_{SW}$ [km/s]')
        ax.get_xaxis().set_ticks([21.5, 100, 215])
        
        #get the OMNI speed for this time
        mask = (data['mjd'] >= plotmjd - 0.2) & (data['mjd'] < plotmjd + 0.2)
        vavg = np.nanmean(data.loc[mask,'V'])
        vmax = np.nanmax(data.loc[mask,'V'])
        vmin = np.nanmin(data.loc[mask,'V'])
        ax.plot(Earth_r, vavg, 'bo', label='Observed')
        ax.plot([Earth_r,Earth_r],[vmin,vmax],'b')
        
        #add some custom points to track, so show outward propagation
        # if i == 0:
        #     ax.plot(88,455,'rd')
        # elif i == 1:
        #     ax.plot(138.5,458,'rd')
        # elif i == 2:
        #     ax.plot(193,455,'rd')
        
        if i ==0:
            ax.legend(bbox_to_anchor=(0.05, 1.1, 1.2, 0.08),
            bbox_transform=ax.transAxes, loc = 'upper left', framealpha = 1)
        
        #if i <2:
            #ax.get_xaxis().set_ticklabels([])
            
        if i ==2:
            ax.set_xlabel(r'Heliocentric distance [$r_S$]')
            
        

    if save_figs_now:   
         fig.savefig(os.path.join(figdir,'Example3days.pdf'))
    


# <codecell> Hovemoller summary of the Sun-Earth line
runstart = firstdate#datetime.datetime(2023,1,1)
smjd = htime.datetime2mjd(runstart)

runend = finaldate#datetime.datetime(2023,12,31)
fmjd = htime.datetime2mjd(runend)

fracs = np.arange(0,1,1/24)
#fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

plot_mjd = smjd

vSE_ss = []
vSE_td = []
mjds = []
vE = []

while plot_mjd <= fmjd - 1:
    plot_date = htime.mjd2datetime(plot_mjd).item()
    dmjd = plot_mjd - smjd
    
    
    #run the steady-state model
    #===========================
    #get the CR num and cr_lon_init that this corresponds to
    cr, cr_lon_init = Hin.datetime2huxtinputs(plot_date)
    #find the map with this date
    id_t_1d = np.argmin(np.abs(time_1d - plot_date))
    #set up a HUXt run with this boundary condition
    model_ss = H.HUXt(v_boundary=vlongs_1d[:, id_t_1d]*u.km/u.s, cr_num=cr, cr_lon_init=cr_lon_init,
                   simtime=1*u.day, dt_scale=1, r_min = 21.5*u.solRad,
                   r_max = 250*u.solRad, frame='synodic')
    model_ss.solve([])
    
    
    for plot_day_frac in fracs:

            
        thismjd = plot_mjd + plot_day_frac
        Earth_r = (hcoords.earth_R(thismjd)*u.km).to(u.solRad).value
        mjds.append(thismjd)
       
        #get speed along S-E line
        model_ts_mjd = model_ss.time_init.mjd + model_ss.time_out.value/daysec
        id_t = np.argmin(np.abs(model_ts_mjd - thismjd))
        vr_ss = model_ss.v_grid[id_t, :, 0]
        vSE_ss.append(vr_ss)
        
        model_ts_mjd = model_td.time_init.mjd + model_td.time_out.value/daysec
        id_t = np.argmin(np.abs(model_ts_mjd - thismjd))
        vr_td = model_td.v_grid[id_t, :, 0]
        vSE_td.append(vr_td)
        
        #get the OMNI speed for this time
        mask = (data['mjd'] >= thismjd - 0.1) & (data['mjd'] < thismjd + 0.1)
        vavg = np.nanmean(data.loc[mask,'V'])
        vmax = np.nanmax(data.loc[mask,'V'])
        vmin = np.nanmin(data.loc[mask,'V'])
        vE.append(vavg)
        
    plot_mjd = plot_mjd + 1.0
        
   

#plot the CarrLon - time Vin
fig = plt.figure(figsize = (10,5))

r = model_td.r.value
datetimes = htime.mjd2datetime(np.array(mjds))
vSE_ss_array = np.array(vSE_ss)
vSE_td_array = np.array(vSE_td)

ax = plt.subplot(2,1,1)
pc = ax.pcolor( datetimes, r, vSE_ss_array.T, 
            shading='auto', vmin=250, vmax=650)
ax.set_ylabel(r'Heliocentric distance [$r_S$]')
ax.text(1.05,1.05,r'$V_{SW}$ [km/s]' , 
        fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
cbar = plt.colorbar(pc, ax=ax)
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlim([datetime.datetime(2020,12,1,0), datetime.datetime(2020,12,31,0)])

ax = plt.subplot(2,1,2)
pc = ax.pcolor( datetimes, r, vSE_td_array.T, 
            shading='auto', vmin=250, vmax=650)
ax.set_ylabel(r'Heliocentric distance [$r_S$]')
ax.text(1.05,1.05,r'$V_{SW}$ [km/s]' , 
        fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
cbar = plt.colorbar(pc, ax=ax)
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Date (2020)')
ax.set_xlim([datetime.datetime(2020,12,1,0), datetime.datetime(2020,12,31,0)])


# <codecell> put together 1 to 7-day advance forecasts for steady state models

# Initialize lists for for1d to for7d
for_lists = [ [] for _ in range(1, 8) ]

# Initialize lists for tim1d to tim7d
tim_lists = [ [] for _ in range(1, 8) ]

startdate = firstdate + datetime.timedelta(days=4) #datetime.datetime(2023,1,3,0)
stopdate = finaldate#datetime.datetime(2024,1,3,0)

forecasttime = startdate
while forecasttime <=stopdate:

    #get the CR num and cr_lon_init that this corresponds to
    cr, cr_lon_init = Hin.datetime2huxtinputs(forecasttime)
    
    #find the map with this date
    id_t_1d = np.argmin(np.abs(time_1d - forecasttime))
    #set up a HUXt run with this boundary condition
    model = H.HUXt(v_boundary=vlongs_1d[:, id_t_1d]*u.km/u.s, cr_num=cr, cr_lon_init=cr_lon_init,
                   simtime=7*u.day, dt_scale=4, r_min = 21.5*u.solRad, lon_out=0.0*u.rad)
    model.solve([])
    

    #convert the model time to MJD
    tim_mjd = model.time_init.mjd + model.time_out.value/daysec
    f_mjd = htime.datetime2mjd(forecasttime)
    
    #get conditions at Earth
    Earth_ts = HA.get_observer_timeseries(model, observer='Earth', suppress_warning = True)
    
    #now hack out days of data for the various forecasts
    for d_advanced in range(0,7):
        # mask = (model.time_out.value >= daysec*d_advanced) & (model.time_out.value < daysec*(d_advanced+1))
        # tim_lists[d_advanced].extend(tim_mjd[mask])
        # for_lists[d_advanced].extend(Earth_ts.loc[mask,'vsw'])
        
        mask = (tim_mjd >= f_mjd + d_advanced) & (tim_mjd < f_mjd + d_advanced + 1)
        tim_lists[d_advanced].extend(tim_mjd[mask])
        for_lists[d_advanced].extend(Earth_ts.loc[mask,'vsw'])
        
    #advance the date
    forecasttime = forecasttime + datetime.timedelta(days=1)
     

# <codecell> Now compare to 1-hour omni data


#find the maximum period of overlap
smjd = tim_lists[6][0]
fmjd = tim_lists[0][-1]

mask_omni = (data['mjd'] >= smjd) & (data['mjd'] <= fmjd)
data = data[mask_omni]

mae_ss = []

for d_advanced in range(0,7):
    #interpolate the forecast onto the omni time step
    data['for'+str(d_advanced)] = np.interp(data['mjd'], tim_lists[d_advanced], 
                                            for_lists[d_advanced])
    
    mae_ss.append(np.nanmean(np.abs(data['for'+str(d_advanced)] - data['V'])))
    
print(mae_ss)

# plt.figure()
# plt.plot(htime.mjd2datetime(np.array(tim_lists[0])), for_lists[0])
# plt.plot(data['datetime'], data['V'])


# <codecell> do a synodic time-dependent run

# #set up the model, with (optional) time-dependent bpol boundary conditions
# model = Hin.set_time_dependent_boundary(vlongs.T, mjds, runstart, simtime, 
#                                         r_min=r_min, r_max=250*u.solRad,
#                                         frame='synodic', lon_start=0 * u.rad,
#                                         lon_stop=0.1 * u.rad,
#                                         dt_scale=10, latitude=0*u.deg)



# #give the streakline footpoints (in Carr long) to the solve method
# model.solve([])

# #get the Earth timeseries
# #get conditions at Earth
# td_Earth_ts = HA.get_observer_timeseries(model, observer='Earth')
# #convert the model time to MJD
# td_tim_mjd = model.time_init.mjd + model.time_out.value/daysec


    
# data['fortd'] = np.interp(data['mjd'], td_tim_mjd, td_Earth_ts['vsw'])  
# mae = np.nanmean(np.abs(data['fortd'] - data['V']))
# print(mae)

# plt.figure()
# plt.plot(htime.mjd2datetime(np.array(td_tim_mjd)), td_Earth_ts['vsw'])
# plt.plot(data['datetime'], data['V'])



# <codecell> put together 1 to 7-day advance forecasts for time-dependent solar wind

# Initialize lists for for1d to for7d
for_lists_td = [ [] for _ in range(1, 8) ]

# Initialize lists for tim1d to tim7d
tim_lists_td = [ [] for _ in range(1, 8) ]

startdate = firstdate + datetime.timedelta(days = bufferdays_td)#datetime.datetime(2023,1,3,0)
stopdate = finaldate#datetime.datetime(2024,1,3,0)

forecasttime = startdate
while forecasttime <=stopdate:
    
    runstart = forecasttime - datetime.timedelta(days = bufferdays_td)

    #get the CR num and cr_lon_init that this corresponds to
    cr, cr_lon_init = Hin.datetime2huxtinputs(runstart)
    
    #find the map with this date
    id_t_start = np.argmin(np.abs(time - runstart))
    id_t_stop = np.argmin(np.abs(time - forecasttime))
    f_mjd = htime.datetime2mjd(forecasttime)
    
    #create a input carr_v with the last few days
    vlongs_slice =[]
    mjds_slice = []
    for n in range(id_t_start, id_t_stop):
        vlongs_slice.append(vlongs[:,n])
        mjds_slice.append(mjds[n])
    
    #then project forward using the current value
    for i in range(0,7):
        vlongs_slice.append(vlongs[:,id_t_stop])
        mjds_slice.append(f_mjd + i)
        
    vlongs_slice = np.array(vlongs_slice).T
    mjds_slice = np.array(mjds_slice)

    #set up a HUXt run with this boundary condition
    model = Hin.set_time_dependent_boundary(vgrid_Carr=vlongs_slice, time_grid=mjds_slice,
                                            starttime=runstart, 
                                            simtime=(7+bufferdays_td)*u.day, 
                                            r_min=21.5*u.solRad,
                                            frame='synodic', lon_out=0 * u.rad,
                                            dt_scale=4, latitude=0*u.deg)
    model.solve([])
    

    #convert the model time to MJD
    tim_mjd_td = model.time_init.mjd + model.time_out.value/daysec
    f_mjd = htime.datetime2mjd(forecasttime)
    
    #get conditions at Earth
    Earth_ts = HA.get_observer_timeseries(model, observer='Earth', suppress_warning = True)
    
    #now hack out days of data for the various forecasts
    for d_advanced in range(0,7):
        mask = (tim_mjd_td >= f_mjd + d_advanced) & (tim_mjd_td < f_mjd + d_advanced + 1)
        tim_lists_td[d_advanced].extend(tim_mjd_td[mask])
        for_lists_td[d_advanced].extend(Earth_ts.loc[mask,'vsw'])
        
    #advance the date
    forecasttime = forecasttime + datetime.timedelta(days=1)

print('=====================')    
mae_td=[]
for d_advanced in range(0,7):
    #interpolate the forecast onto the omni time step
    data['for_td'+str(d_advanced)] = np.interp(data['mjd'], tim_lists_td[d_advanced], 
                                            for_lists_td[d_advanced])
    
    mae_td.append(np.nanmean(np.abs(data['for_td'+str(d_advanced)] - data['V'])))
    
print(mae_td)

# plt.figure()
# plt.plot(htime.mjd2datetime(np.array(tim_lists_td[0])), for_lists_td[0])
# plt.plot(data['datetime'], data['V'])


fig = plt.figure()
plt.plot(np.arange(1,8,1), mae_ss, 'k',label =  'Steady state')
plt.plot(np.arange(1,8,1), mae_td, 'r',label =  'Time dependent')
plt.legend()
plt.xlabel('Forecast lead time [days]')
plt.ylabel('MAE [km/s]')

if save_figs_now:
    fig.savefig(os.path.join(figdir,'MAE_leadtime.pdf'))

# <codecell> Example time series plot

smjd = htime.datetime2mjd(datetime.datetime(2020,12,1))
fmjd = smjd + 30

dadv = 0

mask = (data['mjd'] >= smjd) & (data['mjd'] < fmjd)

plt.figure(figsize=(12, 6))

plt.plot(data.loc[mask,'datetime'], data.loc[mask,'V'], 'b', label = 'Observed')
plt.plot(data.loc[mask,'datetime'], data.loc[mask,'for'+str(dadv)], 'k', label = 'WSA/HUXt, steady state')
plt.plot(data.loc[mask,'datetime'], data.loc[mask,'for_td'+str(dadv)], 'r', label ='WSA/HUXt, time dependent')
plt.ylabel(r'V$_{SW}$ [km/s]')

plt.legend()
# <codecell> plot of forecast consistency

colors = ['k', 'r', 'k', 'r', 'k', 'r', 'k', 'r', 'k', 'r', 'k', 'r', 'k', 'r']
fig = plt.figure()
ax = plt.subplot(111)

dv_ss = []
dv_td = []
for n in range(1,7):


    mask = (data['for'+str(n)].notna()) & (data['for'+str(n-1)].notna())
    dv = np.abs(data.loc[mask,'for'+str(n)] - data.loc[mask,'for'+str(n-1)])
    meanv = np.nanmean(data.loc[mask,'for'+str(n)])
    dv_ss.append(dv)
    
    dv = np.abs(data.loc[mask,'for_td'+str(n)] - data.loc[mask,'for_td'+str(n-1)])
    meanv = np.nanmean(data.loc[mask,'for_td'+str(n)])
    dv_td.append(dv)
    
    
    
box = ax.boxplot([dv_ss[0], dv_td[0], dv_ss[1], dv_td[1] , 
                  dv_ss[2], dv_td[2], dv_ss[3], dv_td[3], 
                  dv_ss[4], dv_td[4], dv_ss[5], dv_td[5]],
           notch=True, patch_artist=True,showfliers=False,whis=1.5)
for patch, median, color in zip(box['boxes'], box['medians'], colors):
    patch.set(facecolor = color)
    patch.set(color = color)
    median.set(color='w') 
ax.get_xaxis().set_ticks([1.5, 3.5, 5.5, 7.5, 9.5, 11.5])
        
ax.set_ylabel(r'Day-to-day forecast jump, $|\Delta V|$ [km/s]', fontsize = 12)
ax.set_xlabel(r'Forecast leadtime [days]', fontsize = 12)

steady_patch = mpatches.Patch(color='black', label='Steady State')
time_dependent_patch = mpatches.Patch(color='red', label='Time Dependent')

# Add legend
plt.legend(handles=[steady_patch, time_dependent_patch])

if save_figs_now:
    fig.savefig(os.path.join(figdir,'Jumpiness.pdf'))

# <codecell> Comparison of parker angle at Earth
nr = 1
dt = 1 #averaging interval for OMNI data, in days



def save_list_to_file(lst, filename):
    with open(filename, 'w') as f:
        for item in lst:
            f.write(f"{item}\n")

# Initialize lists. Comment this out if runs are split into muliple parts
Bang_ss_list = []
Bang_td_list = []
Bang_omni_list = []
V_ss_list = []
V_td_list = []
V_omni_list = []
Bang_dates = []

startdate = firstdate + datetime.timedelta(days=4) #datetime.datetime(2023,1,3,0)
stopdate = finaldate#datetime.datetime(2024,1,3,0)

forecasttime = startdate
while forecasttime <=stopdate:
    
    print(forecasttime.strftime('%Y-%m-%d'))
    plotmjd = htime.datetime2mjd(forecasttime)
    Earth_r = (hcoords.earth_R(plotmjd)*u.km).to(u.solRad).value
    
    Bang_dates.append(forecasttime)

    # #get the CR num and cr_lon_init that this corresponds to
    cr, cr_lon_init = Hin.datetime2huxtinputs(forecasttime)
    plotmjd = htime.datetime2mjd(forecasttime)
    
    print('running steady state model')
    
    #find the map with this date
    id_t = np.argmin(np.abs(time_1d - forecasttime))
    #set up a steady state HUXt run with this boundary condition
    model_ss = H.HUXt(v_boundary=vlongs_1d[:, id_t]*u.km/u.s, cr_num=cr, cr_lon_init=cr_lon_init,
                    simtime=1*u.day, dt_scale=1, r_min = 21.5*u.solRad, r_max=250*u.solRad,
                    frame='synodic')
    model_ss.solve([])
    
    print('tracing field (steady state)') 
    
    #work out the B angle at Earth
    plotlon, plotr, optimal_lon, optimal_t = HA.find_Earth_connected_field_line(model_ss, 0*u.day)
    #find the nearest point to Earth
    id_r = np.argmin(np.abs(plotr - Earth_r))
    #convert the r, theta to x, y
    x_in = plotr[id_r-nr] * np.cos(plotlon[id_r-nr])
    y_in = plotr[id_r-nr] * np.sin(plotlon[id_r-nr])
    x_out = plotr[id_r] * np.cos(plotlon[id_r])
    y_out = plotr[id_r] * np.sin(plotlon[id_r])
    #find angle to the radial
    angle_rad = np.arctan2(y_out - y_in, x_out - x_in)
    Bang_ss_list.append(angle_rad)
    save_list_to_file(Bang_ss_list, os.path.join(figdir,'Bang_ss.dat'))
    
    #get the solar wind speed here
    id_r = np.argmin(np.abs(model_ss.r.value - Earth_r))
    V = model_ss.v_grid[0, id_r, 0].value
    V_ss_list.append(V)
    save_list_to_file(V_ss_list, os.path.join(figdir,'V_ss.dat'))
    
    
    print('tracing field (time dependent)') 
    
    #now get the time depdendent value
    run_days = plotmjd - model_td.time_init.mjd
    
    #work out the B angle at Earth
    plotlon, plotr, optimal_lon, optimal_t = HA.find_Earth_connected_field_line(model_td, run_days*u.day)
    #find the nearest point to Earth
    id_r = np.argmin(np.abs(plotr - Earth_r))
    #convert the r, theta to x, y
    x_in = plotr[id_r-nr] * np.cos(plotlon[id_r-nr])
    y_in = plotr[id_r-nr] * np.sin(plotlon[id_r-nr])
    x_out = plotr[id_r] * np.cos(plotlon[id_r])
    y_out = plotr[id_r] * np.sin(plotlon[id_r])
    #find angle to the radial
    angle_rad = np.arctan2(y_out - y_in, x_out - x_in)
    Bang_td_list.append(angle_rad)
    save_list_to_file(Bang_td_list, os.path.join(figdir,'Bang_td.dat'))
    
    #get the solar wind speed here
    id_r = np.argmin(np.abs(model_td.r.value - Earth_r))
    id_t = np.argmin(np.abs(model_td.time_out.value - run_days*daysec))
    V = model_td.v_grid[id_t, id_r, 0].value
    V_td_list.append(V)
    save_list_to_file(V_td_list, os.path.join(figdir,'V_td.dat'))
    
    
    
    
    #get the OMNI angle for this time
    mask = (data['mjd'] >= plotmjd) &  (data['mjd'] < plotmjd + dt)
    
    Bx = np.nanmean(data.loc[mask,'BX_GSE'])
    By = np.nanmean(data.loc[mask,'BY_GSE'])
    Bz = np.nanmean(data.loc[mask,'BZ_GSE'])
    V = np.nanmean(data.loc[mask,'V'])
    
    angle_rad = np.arctan2(By,Bx)
    if angle_rad < 0:
        angle_rad = np.pi +angle_rad

    Bang_omni_list.append(angle_rad)
    V_omni_list.append(V)
    
    save_list_to_file(Bang_omni_list, os.path.join(figdir,'Bang_omni.dat'))
    save_list_to_file(V_omni_list, os.path.join(figdir,'V_omni.dat'))
    
    #advance the date
    forecasttime = forecasttime + datetime.timedelta(days=1)
    


# <codecell> plot the magnetic field angle - use short runs (~ month)


Vthresh = 400

dang = 10
angbin_edges = np.arange(0,190,dang)  
  
angbin_centres = (angbin_edges[1:] + angbin_edges[:-1])/2

alpha = 0.5    



Bang_omni = np.array(Bang_omni_list)
Bang_ss = np.array(Bang_ss_list)
Bang_td = np.array(Bang_td_list)
V_omni = np.array(V_omni_list)
V_ss = np.array(V_ss_list)
V_td = np.array(V_td_list)
    



fig = plt.figure(figsize=(5,8))


ax = plt.subplot(311)   

omnihist_values, bin_edges = np.histogram(180 - Bang_omni*180/np.pi, bins=angbin_edges)
sshist_values, bin_edges = np.histogram(180 - Bang_ss*180/np.pi, bins=angbin_edges)
tdhist_values, bin_edges = np.histogram(180 - Bang_td*180/np.pi, bins=angbin_edges)

  
ax.bar(angbin_centres, sshist_values, width=dang, facecolor = 'k', 
       edgecolor = 'k', label = 'Steady-state')
ax.bar(angbin_centres, tdhist_values, width=dang, facecolor = 'r', 
       edgecolor = 'r', label = 'Time-dependent', alpha = alpha)
ax.bar(angbin_centres, omnihist_values, width=dang, facecolor = 'none', 
       edgecolor = 'b', label = 'Observed', alpha = alpha, linewidth =4)

# ax.hist(180 - np.array(Bang_omni_list)*180/np.pi, label = 'Observed')      
# ax.hist(180 - np.array(Bang_td_list)*180/np.pi, label = 'Time-dependent')      
# ax.hist(180 - np.array(Bang_ss_list)*180/np.pi, label = 'Steady-state')   
ax.legend()  
ax.axes.xaxis.set_ticks([0,45, 90, 135, 180])
#ax.get_xaxis().set_ticklabels([])
#ax.set_xlabel('Magnetic field angle to radial [deg]')
ax.set_ylabel('Count')
ax.text(0.75, 0.40, '(a) All data', 
        transform=plt.gca().transAxes, fontsize=fontsize)



ax = plt.subplot(312)   

mask_ss = V_ss > Vthresh
mask_td = V_td > Vthresh
mask_omni = V_omni > Vthresh

omnihist_values, bin_edges = np.histogram(180 - Bang_omni[mask_omni]*180/np.pi, bins=angbin_edges)
sshist_values, bin_edges = np.histogram(180 - Bang_ss[mask_ss]*180/np.pi, bins=angbin_edges)
tdhist_values, bin_edges = np.histogram(180 - Bang_td[mask_td]*180/np.pi, bins=angbin_edges)

  
ax.bar(angbin_centres, sshist_values, width=dang, facecolor = 'k', 
       edgecolor = 'k', label = 'Steady-state')
ax.bar(angbin_centres, tdhist_values, width=dang, facecolor = 'r', 
       edgecolor = 'r', label = 'Time-dependent', alpha = alpha)
ax.bar(angbin_centres, omnihist_values, width=dang, facecolor = 'none', 
       edgecolor = 'b', label = 'Observed', alpha = alpha, linewidth =4)

# ax.hist(180 - np.array(Bang_omni_list)*180/np.pi, label = 'Observed')      
# ax.hist(180 - np.array(Bang_td_list)*180/np.pi, label = 'Time-dependent')      
# ax.hist(180 - np.array(Bang_ss_list)*180/np.pi, label = 'Steady-state')   

ax.axes.xaxis.set_ticks([0,45, 90, 135, 180])
#ax.get_xaxis().set_ticklabels([])
#ax.set_xlabel('Magnetic field angle to radial [deg]')
ax.set_ylabel('Count')
ax.text(0.42, 0.88, '(b) Fast wind (> ' + str(Vthresh) + r' km s$^{-1}$)', 
        transform=plt.gca().transAxes, fontsize=fontsize)


ax = plt.subplot(313)   


mask_ss = V_ss <= Vthresh
mask_td = V_td <= Vthresh
mask_omni = V_omni <= Vthresh

omnihist_values, bin_edges = np.histogram(180 - Bang_omni[mask_omni]*180/np.pi, bins=angbin_edges)
sshist_values, bin_edges = np.histogram(180 - Bang_ss[mask_ss]*180/np.pi, bins=angbin_edges)
tdhist_values, bin_edges = np.histogram(180 - Bang_td[mask_td]*180/np.pi, bins=angbin_edges)

  
ax.bar(angbin_centres, sshist_values, width=dang, facecolor = 'k', 
       edgecolor = 'k', label = 'Steady-state')
ax.bar(angbin_centres, tdhist_values, width=dang, facecolor = 'r', 
       edgecolor = 'r', label = 'Time-dependent', alpha = alpha)
ax.bar(angbin_centres, omnihist_values, width=dang, facecolor = 'none', 
       edgecolor = 'b', label = 'Observed', alpha = alpha, linewidth =4)

# ax.hist(180 - np.array(Bang_omni_list)*180/np.pi, label = 'Observed')      
# ax.hist(180 - np.array(Bang_td_list)*180/np.pi, label = 'Time-dependent')      
# ax.hist(180 - np.array(Bang_ss_list)*180/np.pi, label = 'Steady-state')   

ax.axes.xaxis.set_ticks([0,45, 90, 135, 180])
ax.set_xlabel('Magnetic field angle to radial [deg]')
ax.set_ylabel('Count')
ax.text(0.42, 0.88, '(c) Slow wind (< ' + str(Vthresh) + r' km s$^{-1}$)', 
        transform=plt.gca().transAxes, fontsize=fontsize)




if save_figs_now:
    fig.savefig(os.path.join(figdir,'Bangle.pdf'))
    
    
    
    

# dV = 20
# Vbin_edges = np.arange(200,800,dV)  
  
# Vbin_centres = (Vbin_edges[1:] + Vbin_edges[:-1])/2


# fig = plt.figure(figsize=(5,5))


# ax = plt.subplot(111)   

# omnihist_values, bin_edges = np.histogram(V_omni, bins=Vbin_edges)
# sshist_values, bin_edges = np.histogram(V_ss, bins=Vbin_edges)
# tdhist_values, bin_edges = np.histogram(V_td, bins=Vbin_edges)
  
# ax.bar(Vbin_centres, sshist_values, width=dV, facecolor = 'k', 
#        edgecolor = 'k', label = 'Steady-state')
# ax.bar(Vbin_centres, tdhist_values, width=dV, facecolor = 'r', 
#        edgecolor = 'r', label = 'Time-dependent', alpha = alpha)
# ax.bar(Vbin_centres, omnihist_values, width=dV, facecolor = 'none', 
#        edgecolor = 'b', label = 'Observed', alpha = alpha, linewidth =4)
# ax.legend()

# <codecell> Investigate the effect on CME arrival times





