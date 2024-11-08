#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:51:32 2024

a script to load and process all the WSA ADAPT maps and run time-dependent HUXt
to compute MAE with OMNI. Does not produce all teh additional plots in the 
SciRep paper. For that, run WSA_time_dependent_HUXt.py

@author: vy902033
"""





import os
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import astropy.units as u
import pandas as pd
import imageio
from moviepy.editor import ImageSequenceClip
import h5py


import helio_time as htime
import helio_coords as hcoords
import mplot as mplot
import system as system

import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA



ndays = 1  # download coronal solutions up to this many days prior to the forecast date

firstdate = datetime.datetime(2020,1,1,0)
finaldate = datetime.datetime(2020,12,31,0)

deacc = True #Whether to deaccelerate the WSA maps froms 215 to 21.5 rS
input_res_days = 0.1

daysec = 24*60*60

max_forecast_lead = 40


bufferdays_td = 4 #number of days before the forecasttime to start the time-dependent runs

load_maps_now = False #whether to start from the raw data - not provided on git.
plot_map_frames = False #whether to save each WSA map as an image. Can be used to create a gif

save_figs_now = False

#set plot defaults
system.plot_defaults()

fontsize = 12


#set up directories
cwd = os.path.abspath(os.path.dirname(__file__))
root = os.path.dirname(cwd)
figdir = os.path.join(root,'output')
reduced_data_dir = os.path.join(root,'data')

if load_maps_now:
    wsa_run = 'AGONG' #'MO_API'
    datarootdir = os.path.join( os.getenv('DBOX'),'Data')


    
# <codecell> load all WSA maps and extract Earth lat
mae_ss_list = []
mae_td_list = []   
for R in range(0,12):
    Rstr = str(R).zfill(3)
    
    print('Running ADAPT ' + Rstr)
    h5filename = 'WSA_Earth_lat_properties_ADAPT_' + Rstr + '.h5'
    h5filepath = os.path.join(reduced_data_dir, h5filename)
    
    if load_maps_now:
    
        datadir = os.path.join(datarootdir,'WSA_' + wsa_run)
        
        
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
                    fig, ax, axins = mplot.plotspeedmap(vr_map, vr_longs, vr_lats)
                    
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
    
    #compute teh associateed datetime
    time_1d = htime.mjd2datetime(mjds_1d)
    n_longs = len(vlongs_1d[:,0])
        
    
    
    #increase the time resolution of the vlongs
    mjds = np.arange(mjds_1d[0], mjds_1d[-1], input_res_days)
    time = htime.mjd2datetime(mjds)
    vlongs = np.ones((n_longs, len(mjds)))
    brlongs = np.ones((n_longs, len(mjds)))
    for n in range(0, n_longs):
        vlongs[n,:] = np.interp(mjds, mjds_1d, vlongs_1d[n,:])
        brlongs[n,:] = np.interp(mjds, mjds_1d, brlongs_1d[n,:])
    
    
    # #plot the CarrLon - time Vin
    # fig = plt.figure(figsize = (10,10))
    # ax = plt.subplot(2,1,1)
    # pc = ax.pcolor(time, vr_longs.value*180/np.pi, vlongs, 
    #             shading='auto',vmin=250, vmax=650)
    # ax.set_ylabel('Carrington Longitude [deg]')
    # ax.axes.yaxis.set_ticks([0,90,180,270,360])
    # ax.text(0.15,1.05,r'$V_{SW}$ [km/s]' , 
    #         fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
    # cbar = plt.colorbar(pc, ax=ax)
    
    # ax = plt.subplot(2,1,2)
    # pc = ax.pcolor(time, br_longs.value*180/np.pi, brlongs, 
    #             shading='auto')
    # ax.set_ylabel('Carrington Longitude [deg]')
    # ax.axes.yaxis.set_ticks([0,90,180,270,360])
    # ax.text(0.15,1.05,r'$Br [nT]' , 
    #         fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
    # cbar = plt.colorbar(pc, ax=ax)
    
    # if save_figs_now:
    #     fig.savefig(os.path.join(figdir,'WSAsummary.pdf'))
    
    
    
    
    
    # <codecell> put together 1 to 7-day advance forecasts for steady state models
    
    # Initialize lists for for1d to for7d
    for_lists = [ [] for _ in range(1, max_forecast_lead + 1) ]
    
    # Initialize lists for tim1d to tim7d
    tim_lists = [ [] for _ in range(1, max_forecast_lead + 1) ]
    
    startdate = firstdate + datetime.timedelta(days=bufferdays_td) #datetime.datetime(2023,1,3,0)
    stopdate = finaldate#datetime.datetime(2024,1,3,0)
    
    forecasttime = startdate
    while forecasttime <=stopdate:
        
        f_mjd = htime.datetime2mjd(forecasttime)
        
        #get the CR num and cr_lon_init that this corresponds to
        cr, cr_lon_init = Hin.datetime2huxtinputs(forecasttime)
        
        #find the map with this date
        id_t = np.argmin(np.abs(time - forecasttime))
        #set up a HUXt run with this boundary condition
        model = H.HUXt(v_boundary=vlongs[:, id_t]*u.km/u.s, cr_num=cr, cr_lon_init=cr_lon_init,
                       simtime=max_forecast_lead*u.day, dt_scale=4, 
                       r_min = 21.5*u.solRad, lon_out=0.0*u.rad)
        model.solve([])
        
    
        #convert the model time to MJD
        tim_mjd = model.time_init.mjd + model.time_out.value/daysec
        
        #get conditions at Earth
        Earth_ts = HA.get_observer_timeseries(model, observer='Earth', suppress_warning = True)
        
        #now hack out days of data for the various forecasts
        for d_advanced in range(0,max_forecast_lead):
            # mask = (model.time_out.value >= daysec*d_advanced) & (model.time_out.value < daysec*(d_advanced+1))
            # tim_lists[d_advanced].extend(tim_mjd[mask])
            # for_lists[d_advanced].extend(Earth_ts.loc[mask,'vsw'])
            
            mask = (tim_mjd >= f_mjd + d_advanced) & (tim_mjd < f_mjd + d_advanced + 1)
            tim_lists[d_advanced].extend(tim_mjd[mask])
            for_lists[d_advanced].extend(Earth_ts.loc[mask,'vsw'])
            
        #advance the date
        forecasttime = forecasttime + datetime.timedelta(days=1)
         
    
    # <codecell> Now compare to 1-hour omni data
    
    from sunpy.net import Fido
    from sunpy.net import attrs
    from sunpy.timeseries import TimeSeries
    
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
    
    
    
    
    
    
    #find the maximum period of overlap
    smjd = tim_lists[max_forecast_lead-1][0]
    fmjd = tim_lists[0][-1]
    
    mask_omni = (data['mjd'] >= smjd) & (data['mjd'] <= fmjd)
    data = data[mask_omni]
    
    mae_ss = []
    
    for d_advanced in range(0,max_forecast_lead):
        #interpolate the forecast onto the omni time step
        data['for'+str(d_advanced)] = np.interp(data['mjd'], tim_lists[d_advanced], 
                                                for_lists[d_advanced])
        
        mae_ss.append(np.nanmean(np.abs(data['for'+str(d_advanced)] - data['V'])))
        
    print(mae_ss)
    
    # plt.figure()
    # plt.plot(htime.mjd2datetime(np.array(tim_lists[0])), for_lists[0])
    # plt.plot(data['datetime'], data['V'])
    
    
    mae_ss_list.append(mae_ss)
    
    
    
    # <codecell> put together 1 to 7-day advance forecasts for time-dependent solar wind
    
    # Initialize lists for for1d to for7d
    for_lists_td = [ [] for _ in range(1, max_forecast_lead+1) ]
    
    # Initialize lists for tim1d to tim7d
    tim_lists_td = [ [] for _ in range(1, max_forecast_lead+1) ]
    
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
        for i in range(0,max_forecast_lead):
            vlongs_slice.append(vlongs[:,id_t_stop])
            mjds_slice.append(f_mjd + i)
            
        vlongs_slice = np.array(vlongs_slice).T
        mjds_slice = np.array(mjds_slice)
        
        #set up a HUXt run with this boundary condition
        model = Hin.set_time_dependent_boundary(vgrid_Carr=vlongs_slice, time_grid=mjds_slice,
                                                starttime=runstart, 
                                                simtime=(max_forecast_lead+bufferdays_td)*u.day, 
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
        for d_advanced in range(0,max_forecast_lead):
            mask = (tim_mjd_td >= f_mjd + d_advanced) & (tim_mjd_td < f_mjd + d_advanced + 1)
            tim_lists_td[d_advanced].extend(tim_mjd_td[mask])
            for_lists_td[d_advanced].extend(Earth_ts.loc[mask,'vsw'])
            
        #advance the date
        forecasttime = forecasttime + datetime.timedelta(days=1)
    
    print('=====================')    
    mae_td=[]
    for d_advanced in range(0,max_forecast_lead):
        #interpolate the forecast onto the omni time step
        data['for_td'+str(d_advanced)] = np.interp(data['mjd'], tim_lists_td[d_advanced], 
                                                for_lists_td[d_advanced])
        
        mae_td.append(np.nanmean(np.abs(data['for_td'+str(d_advanced)] - data['V'])))
        
    print(mae_td)
    
    mae_td_list.append(mae_td)
    
   
   
    
mae_ss = np.array(mae_ss_list)
mae_td = np.array(mae_td_list)
np.savetxt(os.path.join(figdir,'mae_ss_agong2020.dat'), mae_ss, fmt='%d', delimiter='\t')
np.savetxt(os.path.join(figdir,'mae_td_agong2020.dat'), mae_td, fmt='%d', delimiter='\t')    
    
# <codecell> final summary plot

mae_ss = np.loadtxt(os.path.join(figdir,'mae_ss_agong2020.dat'))
mae_td = np.loadtxt(os.path.join(figdir,'mae_td_agong2020.dat'))

# Compute statistics for mae_ss and mae_td across lead times
median_ss = np.median(mae_ss, axis=0)
median_td = np.median(mae_td, axis=0)
q1_ss, q3_ss = np.percentile(mae_ss, [25, 75], axis=0)
q1_td, q3_td = np.percentile(mae_td, [25, 75], axis=0)
min_ss = np.min(mae_ss, axis=0)
max_ss = np.max(mae_ss, axis=0)
min_td = np.min(mae_td, axis=0)
max_td = np.max(mae_td, axis=0)
lead_times = np.arange(1,max_forecast_lead+1)

# # Plotting
# plt.figure(figsize=(8, 6))
# ax = plt.subplot(1,2,1)

# # Plot median with solid line and interquartile range (IQR) as shaded area for mae_ss
# ax.plot(lead_times, median_ss, color='blue', label='Steady-state', linewidth=2)
# ax.fill_between(lead_times, q1_ss, q3_ss, color='blue', alpha=0.3)

# # Plot median with solid line and interquartile range (IQR) as shaded area for mae_td
# ax.plot(lead_times, median_td, color='red', label='Time dependent', linewidth=2)
# ax.fill_between(lead_times, q1_td, q3_td, color='red', alpha=0.3)

# # Plot total span (min to max) as light shaded area for mae_ss and mae_td
# ax.fill_between(lead_times, min_ss, max_ss, color='blue', alpha=0.1)
# ax.fill_between(lead_times, min_td, max_td, color='red', alpha=0.1)

# # Customize plot
# ax.set_ylabel(r'Forecast V MAE [km/s]')
# ax.set_xlabel(r'Forecast leadtime [days]')
# ax.legend()
# plt.grid(True)
# #plt.yscale('log')


# ax = plt.subplot(1,2,2)

# # Plot median with solid line and interquartile range (IQR) as shaded area for mae_ss
# ax.plot(lead_times, median_ss, color='blue', label='Steady-state', linewidth=2)
# ax.fill_between(lead_times, q1_ss, q3_ss, color='blue', alpha=0.3)

# # Plot median with solid line and interquartile range (IQR) as shaded area for mae_td
# ax.plot(lead_times, median_td, color='red', label='Time dependent', linewidth=2)
# ax.fill_between(lead_times, q1_td, q3_td, color='red', alpha=0.3)

# # Plot total span (min to max) as light shaded area for mae_ss and mae_td
# ax.fill_between(lead_times, min_ss, max_ss, color='blue', alpha=0.1)
# ax.fill_between(lead_times, min_td, max_td, color='red', alpha=0.1)

# # Customize plot
# ax.set_xlabel(r'Forecast leadtime [days]')
# plt.grid(True)

# ax.set_xlim((0,7))
# ax.set_ylim((70,90))










from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Plotting
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(top=0.65)
ax = plt.subplot(1,1,1)

# Plot median with solid line and interquartile range (IQR) as shaded area for mae_ss
ax.plot(lead_times, median_ss, color='blue', label='Steady-state', linewidth=2)
ax.fill_between(lead_times, q1_ss, q3_ss, color='blue', alpha=0.3)

# Plot median with solid line and interquartile range (IQR) as shaded area for mae_td
ax.plot(lead_times, median_td, color='red', label='Time dependent', linewidth=2)
ax.fill_between(lead_times, q1_td, q3_td, color='red', alpha=0.3)

# Plot total span (min to max) as light shaded area for mae_ss and mae_td
ax.fill_between(lead_times, min_ss, max_ss, color='blue', alpha=0.1)
ax.fill_between(lead_times, min_td, max_td, color='red', alpha=0.1)

# Customize plot
ax.set_ylabel(r'Forecast MAE [km/s]')
ax.set_xlabel(r'Forecast leadtime [days]')
ax.legend(loc = 'lower right')
ax.set_ylim(60, 200)
ax.set_xlim(0, 40)
#plt.grid(True)

# Create zoomed inset axes in the top-left corner
axins = zoomed_inset_axes(ax, 4, loc='upper left', bbox_transform=ax.transAxes,
                          bbox_to_anchor=(0.25, 1.65), borderpad=3)
axins.plot(lead_times, median_ss, color='blue', label='Steady-state', linewidth=2)
axins.fill_between(lead_times, q1_ss, q3_ss, color='blue', alpha=0.3)
axins.plot(lead_times, median_td, color='red', label='Time dependent', linewidth=2)
axins.fill_between(lead_times, q1_td, q3_td, color='red', alpha=0.3)
axins.fill_between(lead_times, min_ss, max_ss, color='blue', alpha=0.1)
axins.fill_between(lead_times, min_td, max_td, color='red', alpha=0.1)
axins.set_xlim(1, 7)
axins.set_ylim(70, 90)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
axins.yaxis.tick_right()  # Move ticks to the right
axins.yaxis.set_label_position('right')  # Move label to the right
axins.yaxis.set_ticks([70,80,90]) 
axins.xaxis.tick_top()  # Move ticks to the top
axins.xaxis.set_ticks([1,2,3,4,5,6,7])  # Move ticks to the top
axins.xaxis.set_label_position('top')  # Move label to the top
axins.set_ylabel(r'MAE [km/s]')
axins.set_xlabel(r'Forecast leadtime [days]')


if save_figs_now:
    fig.savefig(os.path.join(figdir,'wsa_agong_mae.pdf'))