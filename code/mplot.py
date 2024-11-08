# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:46:02 2023

custom plot rountines

@author: mathewjowens
"""
import datetime as datetime
import os as os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import scipy as sp
import scipy.odr as odr
import scipy.stats as stats







#compute the percentiles
def getconfidintervals(endata,confid_intervals):
    tmax = len(endata[0,:])
    n = len(confid_intervals)*2 + 1
    confid_ts = np.ones((tmax,n))*np.nan
    for t in range(0,tmax):
        dist = endata[:,t][~np.isnan(endata[:,t])]
        #median
        confid_ts[t,0] = np.percentile(dist,50)
        for nconfid in range(0,len(confid_intervals)):
            confid_ts[t,2*nconfid+1] = np.percentile(dist,confid_intervals[nconfid])
            confid_ts[t,2*nconfid+2] = np.percentile(dist,100-confid_intervals[nconfid])
    return confid_ts


#plot the percentiles
def plotconfidbands(tdata, endata, confid_intervals = [5,10,32], plot_legend=False):
    
    n = len(confid_intervals)*2 + 1
    #get confid intervals
    confid_ts = getconfidintervals(endata,confid_intervals)
    
    #change the line colours to use the inferno colormap
    # nc = len(confid_intervals) + 1
    # color = plt.cm.cool(np.linspace(0, 1,nc))
    # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(confid_intervals))
    nplot = 1 
    nconfid = 0
    while (nplot < n):
        rgba = mpl.cm.cool(norm(nconfid))
        labeltxt = (str(confid_intervals[nconfid]) + '-' 
                    + str(100-confid_intervals[nconfid]) + 'th')
        plt.fill_between(tdata, confid_ts[:,nplot+1], confid_ts[:,nplot],
                         label= labeltxt, color = rgba, zorder = 0 ) 
        nconfid = nconfid + 1
        nplot = nplot + 2
    #plot the median
    plt.plot(tdata, confid_ts[:,0],'w', label = 'Median', zorder = 0)    
    if plot_legend:
        plt.legend(facecolor='silver')    
        
        
def odr_lin_regress(xvals, yvals, xerr = None, yerr = None, p0 = [1., 2.]):
    """
    Orthogonal distance regression
    
    Parameters
    ----------
    x : xdata
    y : ydata
    xerr : xerror, of same dimensions of x,y
        DESCRIPTION. The default is None.
    yerr :  xerror, of same dimensions of x,y
        DESCRIPTION. The default is None.
    p0 : Initial guess at fit coefficients. Set to [1.] to force c = 0
        DESCRIPTION. The default is [1., 2.].

    Returns
    -------
    p : linear fit coefficients

    """
    
    # Remove NaN values from the data
    valid_indices = ~np.isnan(xvals) & ~np.isnan(yvals)
    x = xvals[valid_indices]
    y = yvals[valid_indices]
    
    def odr_line(p, x):
        y = p[0]*x + p[1]
        return y
    
    linear = odr.Model(odr_line)
    if xerr is None:
        mydata = odr.Data(x, y)#, wd=1./xerr, we=1./yerr)
    else:
        mydata = odr.Data(x, y, wd=1./xerr, we=1./yerr)
    myodr = odr.ODR(mydata, linear, beta0=p0)
    output = myodr.run()
    return output.beta


def ols_odr_example(n = 50, xsigma = 2., ysigma = 1.) : 
    """
    generates random data and demonstrates the ordinary least squares and
    orthogonal distance regression fits
    """      
    # Example data
    x = np.linspace(0, 10, n)
    xerr = np.abs(np.random.normal(0, xsigma, n))
    x = np.random.normal(x, xerr, n)
    
    y = np.linspace(0, 20, n)
    yerr = np.abs(np.random.normal(0, ysigma, n))
    y = np.random.normal(y, yerr)

    #ols fit
    p_ols, cov = np.polyfit(x, y, 1, cov=True)  
    

    
    def line(p, x):
        y = p[0]*x + p[1]
        return y

    #plot it
    plt.figure(figsize = (12,6))
    
    ax = plt.subplot(1,2,1)
    p_odr = odr_lin_regress(x, y, xerr = xerr, yerr=yerr)
    
    ax.errorbar(x, y, xerr, yerr, 'ko', alpha=0.7)
    ax.plot(x, line(p_ols, x), label='OLS', lw=3, color='b')
    ax.plot(x, line(p_odr, x), label='ODR', lw=3, color='r')
    
    # plot the true line:
    X = np.linspace(-10, 10, 10)
    ax.plot(X, 2*X, label='True', lw=3, ls='--', color = 'k')
    ax.legend(loc=(1, .5))
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    ax.set_title('Using x and y errors')
    
    
    ax = plt.subplot(1,2,2)
    p_odr = odr_lin_regress(x, y)
    
    ax.plot(x, y, 'ko', alpha=0.7)
    ax.plot(x, line(p_ols, x), label='OLS', lw=3, color='b')
    ax.plot(x, line(p_odr, x), label='ODR', lw=3, color='r')
    
    # plot the true line:
    X = np.linspace(-10, 10, 10)
    ax.plot(X, 2*X, label='True', lw=3, ls='--', color = 'k')
    ax.legend(loc=(1, .5))
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)
    
    ax.set_title('Ignoring x and y errors')
    
    
    
    plt.tight_layout()
    plt.show()



def lin_regress_bootstrap(x_vals, y_vals, num_bootstraps = 1000, plotnow = False):
    # Perform the linear regression and bootstrapping
    bootstrap_params = []
    
    
    # Remove NaN values from the data
    valid_indices = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    x_values = x_vals[valid_indices]
    y_values = y_vals[valid_indices]
    
    for _ in range(num_bootstraps):
        # Generate a bootstrap sample
        indices = np.random.choice(len(x_values), size=len(x_values), replace=True)
        x_bootstrap = x_values[indices]
        y_bootstrap = y_values[indices]
    
        # Fit the linear regression model to the bootstrap sample
        params, _ = np.polyfit(x_bootstrap, y_bootstrap, deg=1, cov=True)
        bootstrap_params.append(params)
    
    # Calculate the mean and standard deviation of the bootstrap parameters
    bootstrap_params = np.array(bootstrap_params)
    mean_params = np.mean(bootstrap_params, axis=0)
    std_params = np.std(bootstrap_params, axis=0)
    
    # Extract the fitting parameters and their confidence intervals
    fit_slope, fit_intercept = mean_params
    conf_slope = [fit_slope - 2 * std_params[0], fit_slope + 2 * std_params[0]]
    conf_intercept = [fit_intercept - 2 * std_params[1], fit_intercept + 2 * std_params[1]]
    
    if plotnow:
        # Output the fitting parameters and their confidence intervals
        print("Fitted slope (a): {:.2f}".format(fit_slope))
        print("Fitted intercept (b): {:.2f}".format(fit_intercept))
        print("Confidence interval for slope (a): {:.2f} to {:.2f}".format(conf_slope[0], conf_slope[1]))
        print("Confidence interval for intercept (b): {:.2f} to {:.2f}".format(conf_intercept[0], conf_intercept[1]))
        
        # Plot the data and the fitted line with confidence intervals
        plt.scatter(x_values, y_values, label='Data')
        plt.plot(x_values, fit_slope * x_values + fit_intercept, 'r-', label='Fitted Line')
        plt.fill_between(x_values, (conf_slope[0] * x_values + conf_intercept[0]), 
                         (conf_slope[1] * x_values + conf_intercept[1]), 
                         color='blue', alpha=0.2, label='Confidence Interval')

    return fit_slope, fit_intercept, conf_slope, conf_intercept


# # Example data

# n = 50
# xsigma = 2.
# ysigma = 1.
# x = np.linspace(0, 10, n)
# xerr = np.abs(np.random.normal(0, xsigma, n))
# x = np.random.normal(x, xerr, n)

# y = np.linspace(0, 20, n)
# yerr = np.abs(np.random.normal(0, ysigma, n))
# y = np.random.normal(y, yerr)


# num_bootstraps = 1000
# plotnow = True


def odr_lin_regress_bootstrap(x, y, xerr= None, yerr= None, 
                              num_bootstraps = 1000, plotnow = False):
    # Perform the linear via orthongal distance regression and bootstrapping
    #for uncertainty
    bootstrap_params = []
    
    
    # Remove NaN values from the data
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    x_values = x[valid_indices]
    y_values = y[valid_indices]
    
    if xerr is None:
        xerr_values = None
        yerr_values = None
    else:
        xerr_values = xerr[valid_indices]
        yerr_values = yerr[valid_indices]
    
    for _ in range(num_bootstraps):
        # Generate a bootstrap sample
        indices = np.random.choice(len(x_values), size=len(x_values), replace=True)
        x_bootstrap = x_values[indices]
        y_bootstrap = y_values[indices]
            
        if xerr is None:
            # Fit the linear regression model to the bootstrap sample
            params = odr_lin_regress(x_bootstrap, y_bootstrap, xerr = None, yerr = None, p0 = [1., 2.])
        else:
            x_err_bootstrap = xerr_values[indices]
            y_err_bootstrap = yerr_values[indices]
            # Fit the linear regression model to the bootstrap sample
            params = odr_lin_regress(x_bootstrap, y_bootstrap, 
                                     xerr = x_err_bootstrap, yerr = y_err_bootstrap, p0 = [1., 2.])
        
        bootstrap_params.append(params)
    
    # Calculate the mean and standard deviation of the bootstrap parameters
    bootstrap_params = np.array(bootstrap_params)
    mean_params = np.mean(bootstrap_params, axis=0)
    std_params = np.std(bootstrap_params, axis=0)
    
    # Extract the fitting parameters and their confidence intervals
    fit_slope, fit_intercept = mean_params
    conf_slope = [fit_slope - 2 * std_params[0], fit_slope + 2 * std_params[0]]
    conf_intercept = [fit_intercept - 2 * std_params[1], fit_intercept + 2 * std_params[1]]
    
    if plotnow:
        # Output the fitting parameters and their confidence intervals
        print("Fitted slope (a): {:.2f}".format(fit_slope))
        print("Fitted intercept (b): {:.2f}".format(fit_intercept))
        print("Confidence interval for slope (a): {:.2f} to {:.2f}".format(conf_slope[0], conf_slope[1]))
        print("Confidence interval for intercept (b): {:.2f} to {:.2f}".format(conf_intercept[0], conf_intercept[1]))
        
        # Plot the data and the fitted line with confidence intervals
        plt.scatter(x_values, y_values, label='Data')
        plt.plot(x_values, fit_slope * x_values + fit_intercept, 'r-', label='Fitted Line')
        plt.fill_between(x_values, (conf_slope[0] * x_values + conf_intercept[0]), 
                         (conf_slope[1] * x_values + conf_intercept[1]), 
                         color='gray', alpha=0.2, label='Confidence Interval')

    return fit_slope, fit_intercept, conf_slope, conf_intercept

def lin_regress(xvals, yvals, confid_level = 0.95,
                plotnow = False, ax = None, 
                color = 'pink', alpha =1, linecolor = 'r'):
    
    #performs linear fits and returns regression parameters needed to construct
    #confidence interval. From standard error and student t distribution
    
    
    tstat = (1- confid_level)/2 + confid_level 
    
    #deal with nans
    valid_indices = ~np.isnan(xvals) & ~np.isnan(yvals)
    x = xvals[valid_indices]
    y = yvals[valid_indices]
    
    p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
    y_model = np.polyval(p, x)                                 # model using the fit parameters; NOTE: parameters here are coefficients
    
    # Statistics
    n = x.size                                                 # number of observations
    m = p.size                                                 # number of parameters
    dof = n - m                                                # degrees of freedom
    t = stats.t.ppf(tstat, n - m)                              # t-statistic; used for CI and PI bands
    meanx = np.mean(x)  
    varxn = np.sum((x - np.mean(x))**2)   
    
    # Estimates of Error in Data/Model
    resid = y - y_model                                        # residuals; diff. actual data from predicted values
    s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
              
    
    regressparams = [t, s_err, n, meanx, varxn]
    
    #plot the fit on a scatter plot
    if plotnow:
        if ax is None:
            ax = plt.gca()
        #find the plot limits
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        xi = np.arange(xmin, xmax, (xmax - xmin)/100)
        yi = np.polyval(p,xi)
        confid_int, prob_int = lin_regress_confid_int(xi, regressparams)
        ax.fill_between(xi, yi-prob_int,   yi+prob_int,
                          color=color, label='Confidence Interval', alpha = 0.5)
        ax.fill_between(xi, yi-confid_int,   yi+confid_int,
                          color='silver', label='Confidence Interval', alpha = alpha)
        ax.plot(xi, yi, color = linecolor)
        ax.plot(x, y,'ko')
    
    return p, regressparams

def lin_regress_confid_int(xi, regressparams):
    #constructs confidence interval for a range of xi values using regressparams
    #determined by lin_regress
    
    t, s_err, n, meanx, varxn = regressparams
    ci = t * s_err * np.sqrt( 1/n + (xi - meanx)**2 / varxn)  
    pi = t * s_err * np.sqrt(1 + 1/n + (xi - meanx)**2 / varxn)  
    
    return ci, pi

def lin_correl(x_vals, y_vals):
    #computes correlation coefficient ignoring NANs. Returns r and N
    
    # Remove NaN values from the data
    valid_indices = ~np.isnan(x_vals) & ~np.isnan(y_vals)
    x_values = x_vals[valid_indices]
    y_values = y_vals[valid_indices]
    
    #compute correlation
    rl = np.corrcoef(x_values, y_values)
    
    return rl[0,1], len(x_values)

def mengZ(r1,n1, r2, n2, printoutput = True):
    # Fisher Z-transform
    Zr1 = 0.5 * np.log((1 + r1) / (1 - r1))
    Zr2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Standard error of the difference
    SE_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))

    # Test statistic
    Z = (Zr1 - Zr2) / SE_diff

    # Perform two-tailed hypothesis test
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    if printoutput:
        # Print results
        print("Test Statistic (Z):", Z)
        print("P-Value:", p_value)
    
        # Compare p-value to significance level (e.g., 0.05) to make a decision
        if p_value < 0.05:
            print("The correlation coefficients are significantly different.")
        else:
            print("The correlation coefficients are not significantly different.")
        
    return p_value

def plotspeedmap(vr_map, vr_longs, vr_lats, fig = None, ax = None, 
                 plot_colourbar = True, plot_sinelat = False):
    # a function to plot a speed map, such as output from WSA or MAS.
    
    if fig == None:
        fig = plt.figure(figsize = (9,4.5))
    
    if ax == None:
        ax = plt.subplot(1,1,1)
    
    ylims = [-90,90]
    yticks = [-90,-45,0,45,90]
    if plot_sinelat:
        ylims = [-1, 1]
        yticks = [-1,-0.5,0,0.5,1]

    if plot_sinelat:
        pc = ax.pcolor(vr_longs.value*180/np.pi, np.sin(vr_lats.value), vr_map.value, 
               shading='auto',vmin=250, vmax=650)
    else:
        pc = ax.pcolor(vr_longs.value*180/np.pi, vr_lats.value*180/np.pi, vr_map.value, 
               shading='auto',vmin=250, vmax=650)

    ax.set_ylim(ylims); 
    ax.set_xlim([0,360])
    ax.set_xlabel('Carrington Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')


    ax.axes.xaxis.set_ticks([0,90,180,270,360])
    ax.axes.yaxis.set_ticks(yticks)
    #ax.axes.xaxis.set_ticklabels([])
    #ax.axes.yaxis.set_ticklabels([])
    plt.sca(ax)
    #colorbar
    if plot_colourbar:
        axins = inset_axes(ax,
                            width="100%",  # width = 50% of parent_bbox width
                            height="10%",  # height : 5%
                            loc='upper right',
                            bbox_to_anchor=(0.28, 0.58, 0.72, 0.5),
                            bbox_transform=ax.transAxes,
                            borderpad=0,)
        cb = fig.colorbar(pc, cax = axins, orientation = 'horizontal',  pad = -0.1)
        cb.ax.xaxis.set_ticks_position("top")
        ax.text(0.15,1.05,r'$V_{SW}$ [km/s]' , 
                fontsize = 11, transform=ax.transAxes, backgroundcolor = 'w')
    else:
        axins = np.nan
        
    return fig, ax, axins, pc
