#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import num2date
from matplotlib import dates as mdates
mpl.use('Agg') #nonguifor use on datarmor
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
import matplotlib.animation as animation
import geopandas as gpd
from matplotlib.patches import Polygon as mplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import cascaded_union
from shapely.geometry import MultiPolygon


# In[ ]:


def are_elements_identical(input_list):
    if not input_list:
        return True

    first_element = input_list[0]
    for element in input_list[1:]:
        if element != first_element:
            return False

    return True


# In[ ]:


def are_all_elements_zero(input_list):
    return all(element == 0 for element in input_list)


# In[ ]:




#dtf_p90 = xr.open_dataset('/home/shilpa/glory_mat_analysis/noaaoisst_lastday_included/full_spatial_noaaoistt_aoi_lastdayincluded.nc')
dtf_p90 = xr.open_dataset('/home/datawork-lead/datarmor-only/shilpa/noaaoisst_mhw_output_1981_2023_aoi_clim_1993_2019.nc')
mhwstate_p90_ = dtf_p90.spatial_extent_pacific
mhwstate_p90 = np.zeros((15274,132,262)) #trying to close all polygons so adding zeros on the sides
mhwstate_p90[:,1:131,1:261] = mhwstate_p90_




longi = dtf_p90.lon
lati = dtf_p90.lat




newlongi = np.zeros((262))
newlati = np.zeros((132))
newlati[0] = lati[0] - 0.25  # making new lat by adding quarter degree
newlati[1:131] = lati[:]
newlati[-1] = lati[-1] + 0.25
newlongi[0] = longi[0] - 0.25 # making new lon by adding quarter degree
newlongi[1:261] = longi[:]
newlongi[-1] = longi[-1] + 0.25
xx,yy = np.meshgrid(newlongi,newlati)


# In[5]:


datessfull = pd.date_range(start='1981-09-01', end='2023-06-26', periods=15274)
#print(len(datessfull))


# In[64]:


for tx in range (15000,15274):

    poly_numbers = np.zeros((1,132, 262))
    spatial_with_areas = np.zeros((1,132, 262))
  


    fig = plt.figure(figsize=(10,5))
    clevs = np.arange(0.5, 1.5, 0.5)
    contours = plt.contour(xx[:,:], yy[:,:], mhwstate_p90[tx,:,:],levels=clevs)


    areas = []
    polygonslist = []
    vtcslist = []

    paths = contours.collections[0].get_paths() #get the first contour polygon

    for path in paths:
        if len(path.vertices)>=3: 
            vtcs = path.vertices
            vtcslist.append(vtcs)
            polygon = ShapelyPolygon(path.vertices)
            polygonslist.append(polygon)
            print(polygon)
            area = polygon.area
            print(area)
            areas.append(area)

    plt.close(fig)

    if  areas == [] or areas == None : #means there are no mhw polygons on this day
        print(f'no mhw events on this day {datessfull[tx]}')
        data = {'time': [datessfull[tx]],'Sum_polygonareas': [0],'max_polygonarea':[0],'number_polygons':[0]}
        df = pd.DataFrame(data)
        df.to_csv(f'/home2/datawork/slal/noaaoisst_csv_polyareas/mhw_polygon_areas_{tx}.csv', index=False)


        
        
        
        #xrds = xr.Dataset(coords = dict(time = [datessfull[tx]],lat = newlati,lon = newlongi),data_vars = dict(mhw_spatial_extent_area = (['time','lat','lon'],spatial_with_areas)))

        #xrds.to_netcdf(f'/home2/datawork/slal/polyareas_no_negatives_fx_1981_2023_1993_2019_baseline/mhw_polygon_areas_{tx}.nc')
        #xrds.to_netcdf(f'noaaoisst_mhw_polygon_areas_{tx}.nc')

    else: #find real area of polygon
        
        polylist = polygonslist
        
        polyareastest = np.zeros(len(polylist))
        
        colored_polygons_inside_white_index = []
        list_of_intersecting_polygons_full = []

        for i in range(len(polylist)):
            #print(i)
            print(f'polygon index {i}')
            interesctionareapoint = []  # for each polygon get a list of intersecting areas
            intersecting_polygons_ls = []
            for j in range(len(polylist)):

                if j != i:   # if polygon is not the same
                    
                    # Create polygon objects
                    polygon1 = polylist[i]
                    polygon2 = polylist[j]

                    # Example: Check if the polygons intersect
                    if polygon1.intersects(polygon2):
                        print("Polygons intersect.")
                    else:
                        print("Polygons do not intersect.")


                    #intersectarea = polygon1.intersection(polygon2).area
                    #print(intersectarea)



                    if not polygon1.is_valid:
                        polygon1 = polygon1.buffer(0)
                    if not polygon2.is_valid:
                        polygon2 = polygon2.buffer(0)

                    # Example: Check if the polygons intersect and compute intersection area
                    if polygon1.intersects(polygon2):
                        intersectarea = polygon1.intersection(polygon2).area
                        interesctionareapoint.append(intersectarea)
                        
                        print("Polygons intersect.")
                        print("Intersection area:", intersectarea)
                        
                        if intersectarea != 0:
                            intersecting_polygons_ls.append(j) #add index of intersecting polygons
                    else:
                        print("Polygons do not intersect.")
                        interesctionareapoint.append(0)
                    #intersectarea = polylist[i].intersection(polylist[j]).area
                    #interesctionareapoint.append(intersectarea) # get intersection area and add to empty list above, 0 if no intersection
                    #if intersectarea != 0:
                        #intersecting_polygons_ls.append(j) #add index of intersecting polygons
                    #else:
                        #intersecting_polygons_ls.append([]) #if no intersection add empty list

            list_of_intersecting_polygons_full.append(intersecting_polygons_ls)      
           

            print(f'polygonindex {i}, intersectionarea = {interesctionareapoint}')


            nonzero_values = [num for num in interesctionareapoint if num != 0]

            if are_all_elements_zero(interesctionareapoint)== True:   #no intersection with any polygon
                #print(f'polygongonindex {i}, intersectionarea = {interesctionareapoint}')
                polyareastest[i] = polylist[i].area
                print(f'polygonarea= {polylist[i].area}')

            elif (len(nonzero_values)) == 1:
                #nonzero_values = [num for num in interesctionareapoint if num != 0]   #for polygons with 1 intersection, can be colored or non colored
                #if (len(nonzero_values)) == 1:  #one intersection
                    #print(f'polygongonindex {i}, intersectionarea = {interesctionareapoint}')
                polyareas_ = (polylist[i].area) - (np.unique(nonzero_values))
                polyareastest[i] = polyareas_  #area of polygon minus intersection area
                print(f'polygonarea minus one intersection = {polyareastest[i]}, polygonarea = {polylist[i].area}')
                    # usually white area becomea zero here

                #else:
                       #two intersections (one with a white polygon and one with a colored polygon, of identical size)
                    #print(nonzero_values)
            elif (len(nonzero_values)) == 2:
                if are_elements_identical(nonzero_values) == True:
                    print('colored polygon inside white polygon inside colored polygon')
                    area_special = (np.unique(nonzero_values)) 
                    print(area_special)
                    polyareastest[i] = area_special

                    colored_polygons_inside_white_index.append(i) #add polygon index to list of colored polygons in white areas
            else:

                polyareas_t = (polylist[i].area) - (sum(interesctionareapoint))
                polyareastest[i] = polyareas_t




                    #else:
                        #print('not colored polygon inside white polygon inside colored polygon, probably polygon with multiple intersections')




                    
                    
                    
                    
                    
                        #print('not colored polygon inside white polygon inside colored polygon, probably polygon with multiple intersections')

        for i in range (len(polylist)):
            set_b = set(list_of_intersecting_polygons_full[i])
            elementlist = []
            for element in colored_polygons_inside_white_index:
                if element in set_b:
                    elementlist.append(element)
            print(elementlist)

            if len(elementlist)!= 0:

                inlist = []
                for j in range(len(polylist)):
                                if j != i:
                                    if not polylist[i].is_valid:
                                        polylist[i] = polylist[i].buffer(0)
                                    if not polylist[j].is_valid:
                                        polylist[j] = polylist[j].buffer(0)
                                    if polylist[i].intersects(polylist[j]):    
                                        intersectarea = polylist[i].intersection(polylist[j]).area
                                        inlist.append(intersectarea)
                                  
                print(inlist)
                polyarea = round((polylist[i].area),5) - round((sum(inlist)),5)

                print(polylist[i].area)
                print(sum(inlist))
                print(polyarea)

                for pi in elementlist:
                    print(pi)
                    polyarea += round((polylist[pi].area),5)

                print(polyarea)
                polyareastest[i] = polyarea
        
        
        
        data = {'time': [datessfull[tx]],'Sum_polygonareas': [(np.nansum(polyareastest))],'max_polygonarea':[(np.nanmax(polyareastest))],'number_polygons':[(sum(1 for item in polyareastest if item != 0))]}
        df = pd.DataFrame(data)
        df.to_csv(f'/home2/datawork/slal/noaaoisst_csv_polyareas/mhw_polygon_areas_{tx}.csv', index=False)
        #for l in range (132):
        #            for m in range (262):
        #                if mhwstate_p90[tx,l,m] != 0.0:
        #                    point = ShapelyPoint(newlongi[m],newlati[l]) #turn each location in mhw state i.e. 1 to a point
        #                    transformed_point = point
        #                    for pp in range (len(polygonslist)):
        #                        is_inside = polygonslist[pp].contains(transformed_point)
        #                        print(is_inside)
        #                        if  is_inside == True:
        #                            #print(f'this point lat ={l}, lon = {m}, insects polygon no. {pp}, with real area = {realpolygonareas_original[pp]}')
        #                            spatial_with_areas[:,l,m] = polyareastest[pp] 
        #                            poly_numbers[:,l,m] = pp


        #xrds = xr.Dataset(coords = dict(time = [datessfull[tx]],lat = newlati,lon = newlongi),data_vars = dict(
        #               mhw_spatial_extent_area = (['time','lat','lon'],spatial_with_areas), mhw_polygon_number = (['time','lat','lon'],poly_numbers)))

        #xrds.to_netcdf(f'/home2/datawork/slal/polyareas_no_negatives_fx_1981_2023_1993_2019_baseline/mhw_polygon_areas_{tx}.nc')
                

