#get plots for signficant trends in MHW properties along new cal coast
#get timeseries of %eez spatial extent
#get total number of days in hot season and cold season around new caledonia


from cartopy import config
import cartopy.crs as ccrs
import xarray as xr
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os

data_crs = ccrs.PlateCarree(central_longitude=0)

# But you want the map to include the dateline.
proj_crs = ccrs.PlateCarree(central_longitude=180)

# eez and country boundaries
eez = gpd.read_file("/media/shilpa/Expansion/for_sophie_18_april_2024/World_EEZ_v11_20191118/eez_v11.shp")

ncd = eez[eez["TERRITORY1"] == "New Caledonia"]

# there are three types of files:
# 1. all events at a particular location
# 2. mean, median, std, trend, significance at a particular location
# 3. timeseries for the spatial extent plots


# We will use the second type of file; mean, median, std, trend, significance at a particular location to make the following plot:
# significant trends in number of events, duration, mean and max intensities , total annual days and total annual cumaltive intensities

directory = '/media/shilpa/Expansion/for_sophie_18_april_2024/coastal_newcal/'

# Get a list of CSV files starting with 'samoa'
coastal_files = [file for file in os.listdir(directory) if file.startswith("New Caledonia_mean_trends")]

# Initialize an empty list to store dataframes
dfs = []

# Loop through each CSV file, read it into a dataframe, and append it to the list
for file in coastal_files:
    filepath = os.path.join(directory, file)
    df = pd.read_csv(filepath)
    dfs.append(df)

# Concatenate all dataframes into one
result = pd.concat(dfs, ignore_index=True)

# Print or do whatever you want with the concatenated dataframe
#print(len(result)) # there are 709 coastal point for New Caledonia


gdf = gpd.GeoDataFrame(result, geometry=gpd.points_from_xy(result.longitude, result.latitude))

# filter to keep only significant trends....its just like masking in xarray, it will apply the filter to the whole dataframe

filtered_gdf_count = gdf[gdf['sig_trend_per_decade_count'] == 1].copy()

filtered_gdf_duration = gdf[gdf['sig_trend_per_decade_duration'] == 1].copy()

filtered_gdf_mean_intensity = gdf[gdf['sig_trend_per_decade_intensity_mean'] == 1].copy()

filtered_gdf_max_intensity = gdf[gdf['sig_trend_per_decade_intensity_max_max'] == 1].copy()

filtered_gdf_annual_days = gdf[gdf['sig_trend_per_decade_total_days'] == 1].copy()

filtered_gdf_annual_icum = gdf[gdf['sig_trend_per_decade_total_icum'] == 1].copy()

def easy_plot_unevencolorbars_(data,column,i,s,name,legend_kwds, vmin,vmax):

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='OrRd')


    data.plot(column,ax=axs[i],marker = 's',s = s,transform=data_crs,alpha = 1,cmap='OrRd', norm = norm,
            legend=True,legend_kwds=legend_kwds)

    axs[i].coastlines()

    ncd.boundary.plot(ax=axs[i], color="black",alpha = 0.4,transform=data_crs)

    gl = axs[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    axs[i].set_xlim(-25,-8)
    axs[i].set_ylim(-27,-14)
    axs[i].set_title(name, fontsize=18)

# only ploting significant trend

nrows = 1
ncols = 6

fig, axs = plt.subplots(nrows=nrows,ncols=ncols,subplot_kw={'projection': proj_crs},figsize=(34,15))



easy_plot_unevencolorbars_(data=filtered_gdf_count,column= 'count_trend_per_decade',name='MHW events Trend per decade',i=0,s=10, 
                          vmin = gdf.count_trend_per_decade.min(),vmax = gdf.count_trend_per_decade.max(),legend_kwds={'label': "[events per decade] ", 
                                       'orientation': "horizontal",'shrink': 0.7})

easy_plot_unevencolorbars_(data=filtered_gdf_duration,column= 'duration_trend_per_decade',name='Duration Trend per decade',i=1,s=10, 
                          vmin = gdf.duration_trend_per_decade.min(),vmax = gdf.duration_trend_per_decade.max(),legend_kwds={'label': "[days per decade] ", 
                                       'orientation': "horizontal",'shrink': 0.7})

easy_plot_unevencolorbars_(data=filtered_gdf_mean_intensity,column= 'intensity_mean_trend_per_decade',name='Mean Intensity trend per decade',i=2, s=10,
                          vmin = gdf.intensity_mean_trend_per_decade.min(),vmax = gdf.intensity_mean_trend_per_decade.max(),legend_kwds={'label': "[°C per decade] ", 
                                       'orientation': "horizontal",'shrink': 0.7})

easy_plot_unevencolorbars_(data=filtered_gdf_max_intensity,column= 'intensity_max_max_trend_per_decade',name='Max Intensity trend per decade',i=3, s=10,
                          vmin = gdf.intensity_max_max_trend_per_decade.min(),vmax = gdf.intensity_max_max_trend_per_decade.max(),legend_kwds={'label': "[°C per decade]", 
                                       'orientation': "horizontal",'shrink': 0.7})

easy_plot_unevencolorbars_(data=filtered_gdf_annual_days,column= 'total_days_trend_per_decade',name='Total annual MHW days trend per decade',i=4, s=10,
                          vmin = gdf.total_days_trend_per_decade.min(),vmax = gdf.total_days_trend_per_decade.max(),legend_kwds={'label': "[days per decade] ", 
                                       'orientation': "horizontal",'shrink': 0.7})

easy_plot_unevencolorbars_(data=filtered_gdf_annual_icum,column= 'total_icum_trend_per_decade',name='Total annual Cum. Int. trend per decade',i=5, s=10,
                          vmin = gdf.total_icum_trend_per_decade.min(),vmax = gdf.total_icum_trend_per_decade.max(),legend_kwds={'label': "[°C days per decade]", 
                                       'orientation': "horizontal",'shrink': 0.7})

axs[0].text(-0.2, 1.05, 'a', transform=axs[0].transAxes, fontsize=20, fontweight='bold')
axs[1].text(-0.2, 1.05, 'b', transform=axs[1].transAxes, fontsize=20, fontweight='bold')
axs[2].text(-0.2, 1.05, 'c', transform=axs[2].transAxes, fontsize=20, fontweight='bold')
axs[3].text(-0.2, 1.05, 'd', transform=axs[3].transAxes, fontsize=20, fontweight='bold')
axs[4].text(-0.2, 1.05, 'e', transform=axs[4].transAxes, fontsize=20, fontweight='bold')
axs[5].text(-0.2, 1.05, 'f', transform=axs[5].transAxes, fontsize=20, fontweight='bold')

plt.tight_layout()
#plt.savefig('/home/shilpa/glory_mat_analysis/coastal_points/newcal_coast_trends_only_signf2.png',bbox_inches='tight')

# We will use the third type of file; timeseries for the spatial extent plots to make the following plot:
# timeseries of  % new caledonia coastal and EEZ in MHW state


# glorys linear original
glorys_linear = pd.read_csv('/media/shilpa/Expansion/for_sophie_18_april_2024/combined_linear_glorys.csv')
sorted_gl = glorys_linear.sort_values('time_index')
sorted_gl['time'] = pd.to_datetime(sorted_gl['time'])

# noaa full
noaa_1981_2023 = pd.read_csv('/media/shilpa/Expansion/for_sophie_18_april_2024/noaa_%eez_spatial_extent.csv')
sorted_n_1981_2023 = noaa_1981_2023.sort_values('time_index')
sorted_n_1981_2023['time'] = pd.to_datetime(sorted_n_1981_2023['time'])

# noaa full macroscale
noaa_macro_scale = pd.read_csv('/media/shilpa/Expansion/for_sophie_18_april_2024/%eez_spatialextent_25_more.csv')
noaa_macro_scale_1981_2023 = noaa_macro_scale.sort_values('time_index')
noaa_macro_scale_1981_2023['time'] = pd.to_datetime(noaa_macro_scale_1981_2023['time'])


# glorys full macroscale
glorys_macro_scale = pd.read_csv('/media/shilpa/Expansion/for_sophie_18_april_2024/glorys_spatial_macroscale.csv')
glorys_macro_scale_1993_2019 = glorys_macro_scale.sort_values('time_index')
glorys_macro_scale_1993_2019['time'] = pd.to_datetime(glorys_macro_scale_1993_2019['time'])


# replace zeros with nans
df_replacedgl = sorted_gl.replace(0, np.nan)
df_replacedn_1981_2023 = sorted_n_1981_2023.replace(0, np.nan)
noaa_macro_scale_1981_2023_replaced = noaa_macro_scale_1981_2023.replace(0,np.nan)
glorys_macro_scale_1993_2019_replaced = glorys_macro_scale_1993_2019.replace(0,np.nan)

# lets get the coastal information
directory = '/media/shilpa/Expansion/for_sophie_18_april_2024/coastal_newcal'
coastal_files = [file for file in os.listdir(directory) if file.startswith("New Caledonia_spatial")]

# Read all CSV files and concatenate into a single DataFrame
dfs = []
for file in coastal_files:
    # Handle file names with spaces by adding double quotes around the file path
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames
concatenated_df = pd.concat(dfs, ignore_index=True)

# Assuming the column containing presence and absence data is named 'presence'
# Group by 'time' (assuming it's the column name) and calculate the percentage of presence for each day
coastal_percentage = concatenated_df.groupby('time')['spatial_extent'].mean() * 100

presence_df = pd.DataFrame({'time': coastal_percentage.index, 'coastal_percentage': coastal_percentage.values})

start_date = '1993-01-01'
end_date = '2023-10-24'

# Create datetime range
date_range = pd.date_range(start=start_date, end=end_date)
print(len(date_range))


presence_df = pd.DataFrame({'time': coastal_percentage.index,'time_dt':date_range.values, 'coastal_percentage': coastal_percentage.values})

coastal = presence_df.replace(0,np.nan)

fig, ax = plt.subplots(figsize=(15, 4))# constrained_layout=True)

ax.plot(df_replacedgl.time,df_replacedgl['New_Caledonia_%spatial_extent'], color='cyan',alpha = 0.7,linestyle='solid',label='% New Caledonia GLORYS [1993-2019] ')
ax.plot(df_replacedn_1981_2023.time,df_replacedn_1981_2023['New_Caledonia_%spatial_extent'], color='black',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST [1981-2023]')
ax.plot(noaa_macro_scale_1981_2023_replaced.time,noaa_macro_scale_1981_2023_replaced['New_Caledonia_%spatial_extent'], color='magenta',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST Macroscale [1981-2023]')
ax.plot(glorys_macro_scale_1993_2019_replaced.time,glorys_macro_scale_1993_2019_replaced['New_Caledonia_%spatial_extent'], color='green',alpha = 0.5,linestyle='solid',label='% New Caledonia GLORYS Macroscale [1993-2019]')
ax.plot(coastal.time_dt,coastal['coastal_percentage'], color='blue',alpha = 0.5,linestyle='dashed',label='% New Caledonia coastal GLORYS [1993-2023]')

ax.legend()
ax.set_ylim(0, 100)
fig.supylabel('% EEZ MHW spatial extent')
#fig.savefig('/home/shilpa/glory_mat_analysis/coastal_points/timeseries_%eez_spatial_extent_newcaledonia.png')



fig, axs = plt.subplots(6, 1,sharex=True, figsize=(19, 12))# constrained_layout=True)

ax = axs[0]
ax.plot(df_replacedgl.time,df_replacedgl['New_Caledonia_%spatial_extent'], color='cyan',alpha = 0.7,linestyle='solid',label='% New Caledonia GLORYS [1993-2019] ')
ax.plot(df_replacedn_1981_2023.time,df_replacedn_1981_2023['New_Caledonia_%spatial_extent'], color='black',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST [1981-2023]')
ax.plot(noaa_macro_scale_1981_2023_replaced.time,noaa_macro_scale_1981_2023_replaced['New_Caledonia_%spatial_extent'], color='magenta',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST Macroscale [1981-2023]')
ax.plot(glorys_macro_scale_1993_2019_replaced.time,glorys_macro_scale_1993_2019_replaced['New_Caledonia_%spatial_extent'], color='green',alpha = 0.5,linestyle='solid',label='% New Caledonia GLORYS Macroscale [1993-2019]')
ax.plot(coastal.time_dt,coastal['coastal_percentage'], color='blue',alpha = 0.4,linestyle='dashed',label='% New Caledonia coastal GLORYS [1993-2023]')
ax.legend()
ax.set_ylim(0, 100)

ax = axs[1]
ax.plot(df_replacedgl.time,df_replacedgl['New_Caledonia_%spatial_extent'], color='cyan',alpha = 0.7,linestyle='solid',label='% New Caledonia GLORYS [1993-2019] ')
ax.plot(df_replacedn_1981_2023.time,df_replacedn_1981_2023['New_Caledonia_%spatial_extent'], color='black',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST [1981-2023]')
#ax.plot(noaa_macro_scale_1981_2023_replaced.time,noaa_macro_scale_1981_2023_replaced['New_Caledonia_%spatial_extent'], color='magenta',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST Macroscale [1981-2023]')
#ax.plot(glorys_macro_scale_1993_2019_replaced.time,glorys_macro_scale_1993_2019_replaced['New_Caledonia_%spatial_extent'], color='green',alpha = 0.5,linestyle='solid',label='% New Caledonia GLORYS Macroscale [1993-2019]')
#ax.plot(coastal.time_dt,coastal['coastal_percentage'], color='blue',alpha = 0.5,linestyle='dashed',label='% New Caledonia coastal GLORYS [1993-2023]')
ax.legend()
ax.set_ylim(0, 100)

ax = axs[2]
#ax.plot(df_replacedgl.time,df_replacedgl['New_Caledonia_%spatial_extent'], color='cyan',alpha = 0.7,linestyle='solid',label='% New Caledonia GLORYS [1993-2019] ')
#ax.plot(df_replacedn_1981_2023.time,df_replacedn_1981_2023['New_Caledonia_%spatial_extent'], color='black',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST [1981-2023]')
ax.plot(noaa_macro_scale_1981_2023_replaced.time,noaa_macro_scale_1981_2023_replaced['New_Caledonia_%spatial_extent'], color='magenta',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST Macroscale [1981-2023]')
ax.plot(glorys_macro_scale_1993_2019_replaced.time,glorys_macro_scale_1993_2019_replaced['New_Caledonia_%spatial_extent'], color='green',alpha = 0.6,linestyle='solid',label='% New Caledonia GLORYS Macroscale [1993-2019]')
#ax.plot(coastal.time_dt,coastal['coastal_percentage'], color='blue',alpha = 0.5,linestyle='dashed',label='% New Caledonia coastal GLORYS [1993-2023]')
ax.legend()
ax.set_ylim(0, 100)

ax = axs[3]
#ax.plot(df_replacedgl.time,df_replacedgl['New_Caledonia_%spatial_extent'], color='cyan',alpha = 0.7,linestyle='solid',label='% New Caledonia GLORYS [1993-2019] ')
#ax.plot(df_replacedn_1981_2023.time,df_replacedn_1981_2023['New_Caledonia_%spatial_extent'], color='black',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST [1981-2023]')
ax.plot(noaa_macro_scale_1981_2023_replaced.time,noaa_macro_scale_1981_2023_replaced['New_Caledonia_%spatial_extent'], color='magenta',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST Macroscale [1981-2023]')
ax.plot(glorys_macro_scale_1993_2019_replaced.time,glorys_macro_scale_1993_2019_replaced['New_Caledonia_%spatial_extent'], color='green',alpha = 0.6,linestyle='solid',label='% New Caledonia GLORYS Macroscale [1993-2019]')
ax.plot(coastal.time_dt,coastal['coastal_percentage'], color='blue',alpha = 0.4,linestyle='dashed',label='% New Caledonia coastal GLORYS [1993-2023]')
ax.legend()
ax.set_ylim(0, 100)

ax = axs[4]
#ax.plot(df_replacedgl.time,df_replacedgl['New_Caledonia_%spatial_extent'], color='cyan',alpha = 0.7,linestyle='solid',label='% New Caledonia GLORYS [1993-2019] ')
#ax.plot(df_replacedn_1981_2023.time,df_replacedn_1981_2023['New_Caledonia_%spatial_extent'], color='black',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST [1981-2023]')
#ax.plot(noaa_macro_scale_1981_2023_replaced.time,noaa_macro_scale_1981_2023_replaced['New_Caledonia_%spatial_extent'], color='magenta',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST Macroscale [1981-2023]')
ax.plot(glorys_macro_scale_1993_2019_replaced.time,glorys_macro_scale_1993_2019_replaced['New_Caledonia_%spatial_extent'], color='green',alpha = 0.6,linestyle='solid',label='% New Caledonia GLORYS Macroscale [1993-2019]')
ax.plot(coastal.time_dt,coastal['coastal_percentage'], color='blue',alpha = 0.4,linestyle='dashed',label='% New Caledonia coastal GLORYS [1993-2023]')
ax.legend()
ax.set_ylim(0, 100)

ax = axs[5]
#ax.plot(df_replacedgl.time,df_replacedgl['New_Caledonia_%spatial_extent'], color='cyan',alpha = 0.7,linestyle='solid',label='% New Caledonia GLORYS [1993-2019] ')
#ax.plot(df_replacedn_1981_2023.time,df_replacedn_1981_2023['New_Caledonia_%spatial_extent'], color='black',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST [1981-2023]')
ax.plot(noaa_macro_scale_1981_2023_replaced.time,noaa_macro_scale_1981_2023_replaced['New_Caledonia_%spatial_extent'], color='magenta',alpha = 0.5,linestyle='solid',label='% New Caledonia NOAAOISST Macroscale [1981-2023]')
#ax.plot(glorys_macro_scale_1993_2019_replaced.time,glorys_macro_scale_1993_2019_replaced['New_Caledonia_%spatial_extent'], color='green',alpha = 0.5,linestyle='solid',label='% New Caledonia GLORYS Macroscale [1993-2019]')
ax.plot(coastal.time_dt,coastal['coastal_percentage'], color='blue',alpha = 0.4,linestyle='dashed',label='% New Caledonia coastal GLORYS [1993-2023]')
ax.legend()
ax.set_ylim(0, 100)

fig.supylabel('% EEZ MHW spatial extent')
#fig.savefig('/home/shilpa/glory_mat_analysis/coastal_points/timeseries_%eez_spatial_extent_newcaledonia.png')



# We will use the third type of file; timeseries for the spatial extent plots to make the following plot:
# total number of days in hot season and cold season around new caledonia


start_date = '1993-01-01'
end_date = '2023-10-24'

# Create datetime range
date_range = pd.date_range(start=start_date, end=end_date)
print(len(date_range))


# lets get the coastal information
directory = '/media/shilpa/Expansion/for_sophie_18_april_2024/coastal_newcal'
coastal_files = [file for file in os.listdir(directory) if file.startswith("New Caledonia_spatial")]

# Read all CSV files and concatenate into a single DataFrame

# hot season

hot_daily_events = []
hot_lat = []
hot_lon = []

for file in coastal_files:
    # Handle file names with spaces by adding double quotes around the file path
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)

    # use the first value of lat and lon because all values are the same...this file is for only one location
    lat = df['latitude'].iloc[0]
    lon = df['longitude'].iloc[0]
    hot_lat.append(lat)
    hot_lon.append(lon)

    df['time_dt'] = date_range.values
    df['time_dt'] = pd.to_datetime(df['time_dt'])

    # Extract the month component
    df['Month'] = df['time_dt'].dt.month

    # Filter the DataFrame for months November (11) to April (4)
    filtered_df = df[(df['Month'] >= 11) | (df['Month'] <= 4)]

    # Optionally, you can drop the 'Month' column if you don't need it anymore
    filtered_df = filtered_df.drop(columns=['Month'])

    filtered_df_count = filtered_df[filtered_df['spatial_extent'] == 1].copy() # only keep lines with value of 1 for spatail extent

    #len(filtered_df_count) # gives the number of days in hot season with MHW
    hot_daily_events.append(len(filtered_df_count))

hot_df = pd.DataFrame({'count_': hot_daily_events, 'hot_lat':hot_lat, 'hot_lon': hot_lon})

hot_gdf = gpd.GeoDataFrame(hot_df, geometry=gpd.points_from_xy(hot_df.hot_lon, hot_df.hot_lat))


# cold season

cold_daily_events = []
cold_lat = []
cold_lon = []

for file in coastal_files:
    # Handle file names with spaces by adding double quotes around the file path
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)

    # use the first value of lat and lon because all values are the same...this file is for only one location
    lat = df['latitude'].iloc[0]
    lon = df['longitude'].iloc[0]
    cold_lat.append(lat)
    cold_lon.append(lon)

    df['time_dt'] = date_range.values
    df['time_dt'] = pd.to_datetime(df['time_dt'])

    # Extract the month component
    df['Month'] = df['time_dt'].dt.month

    # Filter the DataFrame for months November (11) to April (4)
    # Filter the DataFrame for months May (5) to October (10)
    filtered_df = df[(df['Month'] >= 5) & (df['Month'] <= 10)]

    # Optionally, you can drop the 'Month' column if you don't need it anymore
    filtered_df = filtered_df.drop(columns=['Month'])

    filtered_df_count = filtered_df[filtered_df['spatial_extent'] == 1].copy() # only keep lines with value of 1 for spatail extent

    #len(filtered_df_count) # gives the number of days in hot season with MHW
    cold_daily_events.append(len(filtered_df_count))

cold_df = pd.DataFrame({'count_': cold_daily_events, 'cold_lat':cold_lat, 'cold_lon': cold_lon})

cold_gdf = gpd.GeoDataFrame(cold_df, geometry=gpd.points_from_xy(cold_df.cold_lon, cold_df.cold_lat))

nrows = 1
ncols = 2

fig, axs = plt.subplots(nrows=nrows,ncols=ncols,subplot_kw={'projection': proj_crs},figsize=(20,10))



easy_plot_unevencolorbars_(data=hot_gdf,column= 'count_',name='MHW days Hot Season',i=0, s = 10,
                          vmin = hot_gdf.count_.min(),vmax = hot_gdf.count_.max(),legend_kwds={'label': "[No. of MHW days] ",
                                       'orientation': "horizontal",'shrink': 0.7})

easy_plot_unevencolorbars_(data=cold_gdf,column= 'count_',name='MHW days Cold Season',i=1, s = 10,
                          vmin = cold_gdf.count_.min(),vmax = cold_gdf.count_.max(),legend_kwds={'label': "[No. of MHW days] ",
                                       'orientation': "horizontal",'shrink': 0.7})


axs[0].text(-0.2, 1.05, 'a', transform=axs[0].transAxes, fontsize=20, fontweight='bold')
axs[1].text(-0.2, 1.05, 'b', transform=axs[1].transAxes, fontsize=20, fontweight='bold')


plt.tight_layout()
#plt.savefig('/home/shilpa/glory_mat_analysis/coastal_points/newcal_coast_trends_only_signf2.png',bbox_inches='tight')

vmin = hot_gdf.count_.min()
vmax = hot_gdf.count_.max()
print( "hot_vmin =", vmin)
print( "hot_vmax =", vmax)

vmin = cold_gdf.count_.min()
vmax = cold_gdf.count_.max()
print( "cold_vmin =", vmin)
print( "cold_vmax =", vmax)

# change min and max colors to 380 and 750 ..so color bar is consistent.
# change marker size maybe

nrows = 1
ncols = 2

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={'projection': proj_crs}, figsize=(20, 7))

# Plot for hot season
plot_hot = easy_plot_unevencolorbars_(data=hot_gdf, column='count_', name='MHW days Hot Season', i=0, s=10,
                                      vmin=380, vmax=750,
                                      legend_kwds={'label': "[No. of MHW days]",
                                                   'orientation': "horizontal",
                                                   'shrink': 0.7})

# Plot for cold season
plot_cold = easy_plot_unevencolorbars_(data=cold_gdf, column='count_', name='MHW days Cold Season', i=1, s=10,
                                       vmin=380, vmax=750,
                                       legend_kwds={'label': "[No. of MHW days]",
                                                    'orientation': "horizontal",
                                                    'shrink': 0.7})

# Add text markers
axs[0].text(-0.2, 1.05, 'a', transform=axs[0].transAxes, fontsize=20, fontweight='bold')
axs[1].text(-0.2, 1.05, 'b', transform=axs[1].transAxes, fontsize=20, fontweight='bold')

plt.tight_layout()
#plt.show()
plt.savefig('/media/shilpa/Expansion/for_sophie_18_april_2024/newcal_coast_hot_cold_days.png',bbox_inches='tight')
