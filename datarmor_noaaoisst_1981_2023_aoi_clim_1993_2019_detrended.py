import numpy as np
from scipy.stats import linregress
import xarray as xr
import pandas as pd
import marineHeatWaves as mhw

import glob #is this already installed in virtual envt?

import datetime #is this already installed in virtual envt?

from datetime import date

from functools import partial #is this already installed in virtual envt?


# In[2]:


fnames = glob.glob('/home/datawork-lead/datarmor-only/shilpa/noaaoisst/*.nc')  
fnames.sort()


# In[3]:


filelist = []
for i in fnames:
    filelist.append(i)


# In[3]:


# get files from 1993 to 2019


# In[11]:


#print(filelist[11:]
#filelist[:2]


# In[24]:


def _preprocess(x, lon_bnds, lat_bnds):

    return x.sel(lon=slice(*lon_bnds), lat=slice(*lat_bnds))


lon_bnds, lat_bnds = (145.125, 209.875), (-34.875,-2.625)

partial_func = partial(_preprocess, lon_bnds=lon_bnds, lat_bnds=lat_bnds)

#ds = xr.open_mfdataset(filelist[12:-2], combine='nested',concat_dim="time",preprocess=partial_func)  
ds = xr.open_mfdataset(filelist[:], combine='nested',concat_dim="time",preprocess=partial_func)

ds = ds.load()

# In[25]:


ds


# In[ ]:


time = ds.time
strtime = time.data[:].astype(str)
pdtime = pd.to_datetime(strtime)


# In[ ]:


t = np.array([date.toordinal(x) for x in pdtime])


temparr_raw = np.array(ds.sst[:,:,:])

detrended_sst = np.zeros((15274, 130, 260))


for j in range(130):
	for k in range(260):
	# get temperature at each lat and lon
		to = temparr_raw[:,j,k]
		y = to
		x = np.arange(len(y))   # simple integer time index

	# Fit trend
		slope, intercept, r, p, stderr = linregress(x, y)

	# Detrend
		y_detrended = y - (slope * x + intercept)
		detrended_sst[:,j,k] = y_detrended

temparr = detrended_sst


spatial_extent = np.zeros((15274, 130, 260))
#date_start_arr = np.zeros((9861, 144, 588))
#date_end_arr = np.zeros((9861, 144, 588))
#date_peak_arr = np.zeros((9861, 144, 588))
index_start_arr = np.zeros((15274, 130, 260))
index_end_arr = np.zeros((15274, 130, 260))
index_peak_arr = np.zeros((15274, 130, 260))
duration_arr = np.zeros((15274, 130, 260))
intensity_max_arr = np.zeros((15274, 130, 260))
intensity_mean_arr = np.zeros((15274, 130, 260)) 
intensity_cumulative_arr = np.zeros((15274, 130,260))
#category_arr = np.zeros((9861, 144, 588))
rateonset_arr = np.zeros((15274, 130, 260))
ratedecline_arr = np.zeros((15274, 130, 260))

intensity_t_minus_clim_arr = np.zeros((15274, 130, 260)) 
intensity_t_minus_thresh_arr = np.zeros((15274, 130, 260))

reaction_window_days_arr = np.zeros((15274, 130, 260))
coping_window_days_arr = np.zeros((15274, 130, 260))
recovery_window_days_arr = np.zeros((15274, 130, 260))

severity_index_arr = np.zeros((15274, 130, 260))
no_mhw_events_arr = np.zeros((15274, 130, 260))

# In[ ]:


for j in range(130):
    for k in range(260):

        mhws, clim = mhw.detect(t, temparr[:,j,k], climatologyPeriod=[1993,2019])
        print("I am location ", "j =",j,",","k =",k)
        
        events = mhws['n_events']
        print(events)
        date_start = mhws['date_start'] 
        date_end = mhws['date_end'] 
        date_peak = mhws['date_peak']
        index_start = mhws['index_start']
        index_end = mhws['index_end']
        index_peak = mhws['index_peak']
        duration = mhws['duration'] 
        intensity_max = mhws['intensity_max']
        intensity_mean = mhws['intensity_mean'] 
        intensity_cumulative = mhws['intensity_cumulative'] 
        #category = mhws['category'] 
        rateonset = mhws['rate_onset']
        ratedecline = mhws['rate_decline']  
        
        intensity_t_minus_clim = temparr[:,j,k] - clim['seas']
        intensity_t_minus_thresh = temparr[:,j,k] - clim['thresh']
        print ('intensity_t_minus_clim shape =',intensity_t_minus_clim.shape) 
        print('intensity_t_minus_clim_arr shape = ',intensity_t_minus_clim_arr[:,j,k].shape)
        intensity_t_minus_clim_arr[:,j,k] = intensity_t_minus_clim       # array is filled here
        intensity_t_minus_thresh_arr[:,j,k] = intensity_t_minus_thresh   # array is filled here
        
        severity_index  = (temparr[:,j,k] - clim['seas'][:]) / (clim['thresh'][:] - clim['seas'][:])
        severity_index_arr[:,j,k] = severity_index   # array is filled here

        
        
        #fill array below
        for i in range(events):
            if index_end[i] != 15274: #if events are not ending on the last day
                spatial_extent[index_start[i]:(index_end[i])+1,j,k] = 1
                #date_start_arr[index_start[i]:(index_end[i])+1,j,k] =  date_start[i]
                #date_end_arr[index_start[i]:(index_end[i])+1,j,k] =  date_end[i]
                #date_peak_arr[index_start[i]:(index_end[i])+1,j,k] =  date_peak[i]
                no_mhw_events_arr[index_start[i]:(index_end[i])+1,j,k] =  i+1   
                index_start_arr[index_start[i]:(index_end[i])+1,j,k] =  index_start[i]
                index_end_arr[index_start[i]:(index_end[i])+1,j,k] =  index_end[i]
                index_peak_arr[index_start[i]:(index_end[i])+1,j,k] =  index_peak[i]
                duration_arr[index_start[i]:(index_end[i])+1,j,k] =  duration[i]
                intensity_max_arr[index_start[i]:(index_end[i])+1,j,k] =  intensity_max[i]
                intensity_mean_arr[index_start[i]:(index_end[i])+1,j,k] =  intensity_mean[i]
                intensity_cumulative_arr[index_start[i]:(index_end[i])+1,j,k] =  intensity_cumulative[i]
                #category_arr[index_start[i]:(index_end[i])+1,j,k] =  category[i]
                rateonset_arr[index_start[i]:(index_end[i])+1,j,k] =  rateonset[i]
                ratedecline_arr[index_start[i]:(index_end[i])+1,j,k] =  ratedecline[i]
                
                reaction_window_days_arr[index_start[i]:(index_end[i])+1,j,k] = ((index_peak[i]) - (index_start[i]))
                coping_window_days_arr[index_start[i]:(index_end[i])+1,j,k]   = ((index_end[i]) - (index_peak[i]))

            else: #if events are ending on the last day
                spatial_extent[index_start[i]:index_end[i],j,k] = 1
                #date_start_arr[index_start[i]:index_end[i],j,k] =  date_start[i]
                #date_end_arr[index_start[i]:index_end[i],j,k] =  date_end[i]
                #date_peak_arr[index_start[i]:index_end[i],j,k] =  date_peak[i]
                no_mhw_events_arr[index_start[i]:(index_end[i])+1,j,k] =  i+1     
                index_start_arr[index_start[i]:index_end[i],j,k] =  index_start[i]
                index_end_arr[index_start[i]:index_end[i],j,k] =  index_end[i]
                index_peak_arr[index_start[i]:index_end[i],j,k] =  index_peak[i]
                duration_arr[index_start[i]:index_end[i],j,k] =  duration[i]
                intensity_max_arr[index_start[i]:index_end[i],j,k] =  intensity_max[i]
                intensity_mean_arr[index_start[i]:index_end[i],j,k] =  intensity_mean[i]
                intensity_cumulative_arr[index_start[i]:index_end[i],j,k] =  intensity_cumulative[i]
                #category_arr[index_start[i]:index_end[i],j,k] =  category[i]
                rateonset_arr[index_start[i]:index_end[i],j,k] =  rateonset[i]
                ratedecline_arr[index_start[i]:index_end[i],j,k] =  ratedecline[i]
                
                reaction_window_days_arr[index_start[i]:index_end[i],j,k] = ((index_peak[i]) - (index_start[i]))
                coping_window_days_arr[index_start[i]:index_end[i],j,k]   = ((index_end[i]) - (index_peak[i]))

        # special case for recovery window days
        for i in range(events-1):
                if index_end[i] != 15274: #if events are not ending on the last day
                    recovery_window_days_arr[index_start[i]:(index_end[i])+1,j,k] = ((index_start[i+1])-(index_end[i]))
                    
                else:#if events are ending on the last day
                    recovery_window_days_arr[index_start[i]:index_end[i],j,k] = ((index_start[i+1])-(index_end[i]))
                    
                    
                    


# In[28]:


longi = np.array(ds.lon.values)
lati = np.array(ds.lat.values)
time = pd.date_range(start='1981-09-01', end='2023-06-26')


# In[ ]:


xrds = xr.Dataset(
       coords = dict(time = time,lat = lati,lon = longi),
       data_vars = dict(
                   spatial_extent_pacific = (['time','lat','lon'], spatial_extent),
                   #date_start_arr = (['time','lat','lon',], date_start_arr),    
                   #date_end_arr = (['time','lat','lon',], date_end_arr),  
                   #date_peak_arr = (['time','lat','lon',], date_peak_arr),  
                   
                   index_start_pacific = (['time','lat','lon'], index_start_arr),  
                    
                   index_end_pacific = (['time','lat','lon'], index_end_arr),  
                   index_peak_pacific = (['time','lat','lon'], index_peak_arr),  
                   duration_pacific = (['time','lat','lon'], duration_arr),  
                   intensity_max_pacific = (['time','lat','lon'], intensity_max_arr),  
                   intensity_mean_pacific = (['time','lat','lon'], intensity_mean_arr),  
                   intensity_cumulative_pacific = (['time','lat','lon'], intensity_cumulative_arr),  
                   #category_arr = (['time','lat','lon',], category_arr),  
                   rateonset_pacific = (['time','lat','lon'],  rateonset_arr),  
                   ratedecline_arr = (['time','lat','lon'], ratedecline_arr),  
                   intensity_t_minus_clim_pacific = (['time','lat','lon'], intensity_t_minus_clim_arr),  
                   intensity_t_minus_thresh_pacific = (['time','lat','lon'], intensity_t_minus_thresh_arr),  

                   reaction_window_days_pacific = (['time','lat','lon'], reaction_window_days_arr),  
                   coping_window_days_pacific = (['time','lat','lon'], coping_window_days_arr),  
                   recovery_window_days_pacific = (['time','lat','lon'], recovery_window_days_arr),
               
                   severity_index_pacific = (['time','lat','lon'], severity_index_arr),
                   mhw_event_no_pacific = (['time','lat','lon'], no_mhw_events_arr)

                           
                  
                       ))
             



# In[ ]:




xrds.to_netcdf('/home/datawork-lead/datarmor-only/shilpa/noaaoisst_mhw_output_1981_2023_aoi_clim_1993_2019_detrended.nc')


