# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:55:32 2024

The scritp is to autosend TSalarm message by LINE

@author: user
"""

import requests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from PIL import Image
import matplotlib.font_manager as fm
from matplotlib.patches import Polygon, Rectangle
#from shapely.geometry import Point, Polygon
import sys
from datetime import datetime, timedelta
import os
import json
# ============================================================================
# Extra file and data path setting
# Windows
# exfile_station_info = './TSalarm_pylineauto_20240625/station_info_new.txt' # Need to change in server
# exfile_rail_map_image = './TSalarm_pylineauto_20240625/test-Photoroom.png'
# exfile_font_path ="./TSalarm_pylineauto_20240625/msjhbd.ttc"
# alarmfile_dir = './result/' # Need to change in server
# Server
exfile_station_info = '/home/yhlee/python/program/TR_TSalarm/exfile/station_info_UTF8.txt' # Need to change in server
exfile_station_grid_info = '/home/yhlee/python/program/TR_TSalarm/exfile/grid_station_info_UTF8.txt'
exfile_rail_map_image = '/home/yhlee/python/program/TR_TSalarm/exfile/test-Photoroom.png'
exfile_font_path ="/home/yhlee/python/program/TR_TSalarm/exfile/msjhbd.ttc"
alarmfile_dir = '/data/analysis/TRA/TS_alarm/'
# ============================================================================
# Function to swap lon lat of polygon from CWA (Copy)
def swap_columns(matrix, col1, col2):
    for row in matrix:
        row[col1], row[col2] = row[col2], row[col1]
# ============================================================================
# Fixed QPF version, but need to check
def loadCWAQPF(poly, R1, R2):
    # Set the URL for the data source
    url = 'https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/F-B0046-001?Authorization=rdec-key-123-45678-011121314&format=JSON'
    
    # Fetch the data from the URL
    response = requests.get(url)
    data = response.json()
    
    # Extract the content
    content = data['cwaopendata']['dataset']['contents']['content']
    time = data['cwaopendata']['dataset']['datasetInfo']['parameterSet']['DateTime']
    # Extract the time part
    time_str = time[11:16]
    
    # Convert the content string to a list of floats
    c = np.fromstring(content, sep=',', dtype=float)
    
    # Replace values less than 0 with NaN
    c[c < 0] = np.nan
    # Reshape the values into a 441x561 array
    cc = c.reshape((561, 441))
    
    # Create meshgrid for latitude and longitude
    x = np.arange(118, 123.5 + 0.012, 0.0125)
    y = np.arange(20, 27 + 0.0125, 0.0125)
    xx, yy = np.meshgrid(x, y)
    
    # Check which points are inside the polygon
    poly1 = plt.Polygon(poly, closed=True)
    
    # Flatten the meshgrid arrays to get a list of points
    points = np.vstack((xx.ravel(), yy.ravel())).T

    # Create a Path object from the polygon
    path = Path(poly1.get_xy())

    # Use the contains_points method to get a boolean array indicating which points are inside the polygon
    mask = path.contains_points(points)
    # Reshape the boolean array back to the shape of the original meshgrid arrays
    mask = mask.reshape(xx.shape)
    
    # Read railway array
    all_railway = np.loadtxt('/home/yhlee/python/program/TR_TSalarm/exfile/railway_region.txt', dtype=int)
    affected_railway = all_railway*mask.astype(int)
    affected_railway_bool = affected_railway.astype(bool)
    
    if np.all(np.isnan(c)):
        QPF1 = -999.
        QPF2 = -999.
    elif np.any(affected_railway_bool) == False: # 241007 受影響區域不包含臺鐵線路時，仍輸出QPF
        QPF1 = np.nanpercentile(cc[mask], R1)
        QPF2 = np.nanpercentile(cc[mask], R2)
    else:
        QPF1 = np.nanpercentile(cc[affected_railway_bool], R1)
        QPF2 = np.nanpercentile(cc[affected_railway_bool], R2)
    
    QPF1 = np.round(QPF1 / 5) * 5
    QPF2 = np.round(QPF2 / 5) * 5
    
    return time_str, QPF1, QPF2
# ============================================================================
# Load CWA data
url = 'https://cbph.cwa.gov.tw/api/cells/?order=asc&offset=0&limit=20'
response = requests.get(url)
wr = response.json()    # All data stored here
# ============================================================================
# Check new alarm exist or not
t_str = datetime.strptime(str(wr[0]['effective']),"%Y-%m-%dT%H:%M:%SZ")
t_lst = t_str+timedelta(hours=8)
t_lst_str = t_lst.strftime("%Y%m%d")
alarm_id = wr[0]['id']
alarmfile_list = alarmfile_dir+'list/'
if os.path.exists(alarmfile_list) == False:
    os.makedirs(alarmfile_list)

checkfile_path = alarmfile_list+t_lst_str+'_'+str(alarm_id)+'.json'
if os.path.isfile(checkfile_path):
    sys.exit('No new alarm!!!') ###The process will finish if no new alarm

with open(checkfile_path, 'w', encoding='utf-8') as json_file:
    json.dump(wr[0], json_file, ensure_ascii=False, indent=4)
# ============================================================================
# Set date directory to save result figure
figdir = alarmfile_dir+t_lst_str[:4]+'/'+t_lst_str[4:6]+'/'+t_lst_str[6:8]+'/'
if os.path.exists(figdir) == False:
    os.makedirs(figdir)
# ============================================================================
# Set font style(copy)
# font_path = "c:\WINDOWS\Fonts\MSJH.TTC"
font_path = exfile_font_path # Fixed font file in .zip
my_font = fm.FontProperties(fname=font_path)
# ============================================================================
linemsg_str = "\n\n"
# ============================================================================
# Load station info (modified version with re-arranged station order) (copy)
station_info = np.loadtxt(exfile_station_info, delimiter='\t', skiprows=1, dtype=str)
station_grid = np.loadtxt(exfile_station_grid_info, delimiter='\t', skiprows=1, dtype=str)

# Save into different arrays
lineName = station_info[:, 0]
staMil = station_info[:, 2].astype(float)
stationName = station_info[:, 3]
x = station_info[:, 4].astype(float)
y = station_info[:, 5].astype(float)
show = station_info[:, 6].astype(bool)

stationName_grid = station_grid[:, 3]
x_grid = station_grid[:, 4].astype(float)
y_grid = station_grid[:, 5].astype(float)

station_show = stationName[show]
x_show = x[show]
y_show = y[show]

# Modify coordinates of important stations for plotting
x_show = (x_show - 118) * 600   # Unify x-coordinates with radar image
y_show = 3600 - (y_show - 20.5) * 600   # Unify y-coordinates with radar image (need to flip y-coordinates)
# All stations for checking
#x = (x - 118) * 600   # Unify x-coordinates with radar image
#y = 3600 - (y - 20.5) * 600   # Unify y-coordinates with radar image (need to flip y-coordinates)
# ============================================================================
# Create a dictionary to group stations by lineName (copy)
line_stations = {}
for i, line_name in enumerate(lineName):
    if line_name not in line_stations:
        line_stations[line_name] = []   # Create new group if lineName not found
    line_stations[line_name].append((i, stationName[i]))    # Add station into group of respective lines
# ============================================================================
# Branch Line array
branch_line = np.array(["平溪線","內灣線","集集線","深澳線","六家線","成追線","沙崙線"])
# ============================================================================
# Prepare for loop if necessary
# for num in range(end-1,start-1,-1): 
num = 0
# ============================================================================
wpoly = [list(map(float, point.split(','))) for point in wr[num]['polygon'].split()] # Create polygon list from wr info
swap_columns(wpoly, 0, 1) # Re-arrange lon lat order

# Create time string for text message (Effective)
tt0 = (np.datetime64(wr[num]['effective']) + np.timedelta64(8, 'h')).astype(str)
tt0 = tt0[:16].replace('T', ' ')
# Create time string for text message (Expire)
tt1 = (np.datetime64(wr[num]['expires']) + np.timedelta64(8, 'h')).astype(str)
tt1 = tt1[11:16]
# ============================================================================
msg0 = wr[num]['description']   # Store whole message
# Extract wanted info from message
m1 = msg0.find('，') 
m2 = msg0.find('；')
# m3 = msg0.find('，', m2+1)

# 241008 因訊息格式改版，修正判斷式
m3 = msg0.find('。')
if m3 == -1:    # If end(m3) not identified, set m3 to include rest of message
    m3 = msg0.find('，', m2+1)
    if m3 == -1:
        m3 = len(msg0)

msg1 = f"{tt0} {msg0[m1-12:m1+6]} {tt1}"    # Info for time
msg2 = msg0[m2:m3]  # Info for potential hazards

linemsg_str += (msg1 + msg2)

# Define polygon (effective area) for checking and plotting
poly1 = plt.Polygon(wpoly, closed=True)

# Create a list to store station indices
all_station_indices = []

# covered = False # To ensure only 1 extra station is added
# # Put stations that lie inside the polygon (Included in alarm area) into one list
# for i in range(len(lineName)):  # Iterate over all stations
    
#     if poly1.contains_point((x[i], y[i])):  # Check if the station lies inside the polygon (Included in alarm area)
#         if covered == False and i != 0 and lineName[i] == lineName[i-1]:
#             all_station_indices.append(i-1) # Add and consider 1 station before the stations included in polygon
#             covered = True
#         all_station_indices.append(i)   # Add the station index to the list if yes
#     elif i != 0:
#         if poly1.contains_point((x[i-1], y[i-1])) and lineName[i] == lineName[i-1]:
#             all_station_indices.append(i)
#             covered = False
#     else:
#         covered = False

all_grid_contain = np.zeros(len(x_grid))
for i in range(len(x_grid)):  # Iterate over all stations
    if poly1.contains_point((x_grid[i], y_grid[i])):  # Check if the station lies inside the polygon (Included in alarm area)
        all_grid_contain[i] = 1
all_grid_contain = all_grid_contain.astype(bool)
for i in range(len(lineName)):
    if stationName[i] in stationName_grid[all_grid_contain]:
        all_station_indices.append(i)

extra_indices = []

for i in range(len(all_station_indices)):
    if all_station_indices[i] != 0 and lineName[all_station_indices[i]] == lineName[all_station_indices[i]-1] and all_station_indices[i] in all_station_indices and all_station_indices[i]-1 not in all_station_indices:
        extra_indices.append(all_station_indices[i]-1)
    if all_station_indices[i] != len(lineName)-1 and lineName[all_station_indices[i]] == lineName[all_station_indices[i]+1] and all_station_indices[i] in all_station_indices and all_station_indices[i]+1 not in all_station_indices:
        extra_indices.append(all_station_indices[i]+1)
if all_station_indices:
    merged_list = all_station_indices + extra_indices
    merged_list.sort()
    all_station_indices = merged_list.copy()

sorted_stations_by_line = []
# ============================================================================
# Booleans for further decisions (if any stations affected / group stations by line)
tra_affected = False
heading_printed = False
first_line_printed = False

# Iterate over all TRA stations to obtain affect sections
for line_name, stations in line_stations.items():   # Iterate over all TRA stations
    # Names and indices of affected stations are stored in list
    stations_in_polygon = [(idx, name) for idx, name in stations if idx in all_station_indices]
    
    # If there are stations within the polygon for the current line
    if stations_in_polygon: # If any station is included in affected area
        tra_affected = True
        
        # Print heading
        if heading_printed == False:
            # linemsg_str += "\n對臺鐵線路影響範圍："
            affected_str = "\n\n影響鐵路區間："
            heading_printed = True
        
        # Sort stations by station index
        sorted_stations = sorted(stations_in_polygon, key=lambda x: x[0])

        # Output "whole line" instead of specific section when all stations are affected
        if line_name in branch_line:
            if first_line_printed == True:
                # linemsg_str += "、"
                affected_str += "、"
            # linemsg_str += f"{line_name}"
            affected_str += f"{line_name}"
            first_line_printed = True
        elif len(stations_in_polygon) == len(stations):   # If all stations for this line are within the polygon
            if first_line_printed == True:
                # linemsg_str += "、"
                affected_str += "、"
            # linemsg_str += f"{line_name} (全線)"
            affected_str += f"{line_name} (全線)"
            first_line_printed = True
        else:
            if first_line_printed == True:
                # linemsg_str += "、"
                affected_str += "、"
            # Extract the first and last station names
            first_station = sorted_stations[0][1]
            last_station = sorted_stations[-1][1]
            # Print the line and the first and last station names to indicate with section
            # Note: May need manual adjustments for selection of "major" stations
            if first_station == last_station:
                # linemsg_str += f"{line_name} ({first_station})"
                affected_str += f"{line_name} ({first_station})"
            else:
                # linemsg_str += f"{line_name} ({first_station}-{last_station})"
                affected_str += f"{line_name} ({first_station}-{last_station})"
            first_line_printed = True

# Print message if no stations are affected
if tra_affected == False:
    # linemsg_str_ref = linemsg_str + "\n對臺鐵線路無影響"
    # linemsg_str = linemsg_str + "\n對臺鐵線路無影響"
    affected_str = "\n\n對臺鐵線路無影響"
    
# if tra_affected == True:
# Get CWAQPF Data
R1 = 50 # Choose target percentile
R2 = 95 # Choose target percentile
qpf_time_str, QPF1, QPF2 = loadCWAQPF(wpoly, R1, R2)  # No precip data = nan?

if tra_affected == True:
    if QPF1 >= 0. and QPF2 >= 0. and QPF1 != QPF2:
        # linemsg_str_ref = linemsg_str + "\n\nQPF：未來一小時內降雨可能達"+"{:.1f}".format(QPF1)+"~"+"{:.1f}".format(QPF2)+"mm"
        # qpf_str = "\n\nQPF：未來一小時內降雨可能達"+"{:.1f}".format(QPF1)+"~"+"{:.1f}".format(QPF2)+"mm"
        qpf_str = "\n\n"+qpf_time_str+"起一小時內受影響路段降雨可能達"+"{:.0f}".format(QPF1)+"~"+"{:.0f}".format(QPF2)+"mm"
    elif QPF1 == QPF2:
        qpf_str = "\n\n"+qpf_time_str+"起一小時內受影響路段降雨可能達"+"{:.0f}".format(QPF1)+"mm"
elif tra_affected == False:
    if QPF1 >= 0. and QPF2 >= 0. and QPF1 != QPF2:
        qpf_str = "\n\n"+qpf_time_str+"起一小時內受影響區域降雨可能達"+"{:.0f}".format(QPF1)+"~"+"{:.0f}".format(QPF2)+"mm" # 241007 增加對台鐵路線無影響的QPF訊息
    elif QPF1 == QPF2:
        qpf_str = "\n\n"+qpf_time_str+"起一小時內受影響區域降雨可能達"+"{:.0f}".format(QPF1)+"mm" # 241007 增加對台鐵路線無影響的QPF訊息
else:
    # linemsg_str_ref = linemsg_str + "\n\nQPF：無法估計未來一小時內降雨或受影響鐵路沿線未來一小時內無顯著降雨"
    qpf_str = "\n\n"+qpf_time_str+"起一小時內降雨無法被估計或受影響路段一小時內無顯著降雨"

#==================================================================================
# Combine text message
linemsg_str_pub = affected_str + linemsg_str
linemsg_str_ref = affected_str + qpf_str + linemsg_str

# ============================================================================
# Load radar map & colorbar
url = 'https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/O-A0058-006?Authorization=rdec-key-123-45678-011121314&format=JSON'
response = requests.get(url)
SR = response.json()    # CWA radar data stored here

radar_url = SR['cwaopendata']['dataset']['resource']['ProductURL']
radar_image = Image.open(requests.get(radar_url, stream=True).raw)  # Radar image stored here

ttR = SR['cwaopendata']['dataset']['DateTime']  # Time of Radar image
ttR = ttR[:10] + ' ' + ttR[11:16]   # Time as string

radar_colorbar_url = 'https://www.cwa.gov.tw/V8/assets/img/radar/colorbar_n.png'
radar_colorbar = Image.open(requests.get(radar_colorbar_url, stream=True).raw).convert("RGBA")  # Radar colorbar stored here as image
radar_colorbar_array = np.array(radar_colorbar)

# Define the area where you want to apply transparency
x_start, y_start, x_end, y_end = 1, 1, 1124, 68  # Adjust these coordinates

# Create an alpha mask with the same size as the image
alpha_mask = np.ones((radar_colorbar.height, radar_colorbar.width), dtype=np.float32)
alpha_mask[y_start:y_end, x_start:x_end] = 0.55  # Set alpha to 0.55 in the defined area

# Convert alpha mask to 4-channel RGBA image
alpha_layer = (alpha_mask * 255).astype(np.uint8)
rgba_image_array = np.dstack((radar_colorbar_array[:, :, :3], alpha_layer))

# Convert back to Image for display
radar_colorbar = Image.fromarray(rgba_image_array)
# ============================================================================
# Load rail map .png file
rail_map_image = Image.open(exfile_rail_map_image)
# ============================================================================
# Plotting section
fig, ax = plt.subplots(figsize=(6, 6))  # Set image output size
# Warning color tone
clr = 'orangered'  # TODO: Test different colors

# Process Polygon
wpoly_mod = np.array(wpoly)
wpoly_mod[:,0] = (wpoly_mod[:,0] - 118) * 600   # Unify x-coordinates with radar image
wpoly_mod[:,1] = 3600 - (wpoly_mod[:,1] - 20.5) * 600   # Unify y-coordinates with radar image (need to flip y-coordinates)
poly1_mod = plt.Polygon(wpoly_mod, closed=True, facecolor=matplotlib.colors.to_rgba(clr, alpha=0.5), edgecolor=matplotlib.colors.to_rgba(clr, alpha=1), linewidth=2) # Polygon settings

# ax.add_patch(poly1_mod) # Plot polygon (affected area)

# Set x, y lim (zoom in)
max_x = np.max(wpoly_mod[:,0])
min_x = np.min(wpoly_mod[:,0])
max_y = np.max(wpoly_mod[:,1])
min_y = np.min(wpoly_mod[:,1])

mid_x = (max_x + min_x) / 2 # Locate center of polygon
mid_y = (max_y + min_y) / 2

radius = np.max(np.array([max_x-min_x, max_y-min_y, 450])/2) / 0.75  # Set x, y lim such that polygon takes up ~75% the image (minimum 400 pixel)
ratio_adjust = radius/(225/0.7)
# ============================================================================
# Process Radar and Rail Map
ax.imshow(radar_image, alpha=0.55)  # Plot radar image
ax.imshow(rail_map_image, extent=[1800-480*1.69, 1800+480*1.69, 1800+640*1.6, 1800-640*1.785], alpha=0.8)   # Plot updated rail map image
# ============================================================================
# Plot important stations
ax.scatter(x_show, y_show, s=50, c='k')
ax.scatter(x_show, y_show, s=20, c='white')
# All stations for checking
#ax.scatter(x, y, s=50, c='k')
#ax.scatter(x, y, s=20, c='white')

# Define area to show station name labels
margin = 0.9    # only show names within {margin}*radius
vertices_show = [(mid_x-radius*margin, mid_y-radius*(margin-0.2)),(mid_x+radius*margin, mid_y-radius*(margin-0.2)),
                 (mid_x+radius*margin, mid_y+radius*margin),(mid_x-radius*margin, mid_y+radius*margin)]

# Print station name label
for i, txt in enumerate(station_show):
    if Path(vertices_show).contains_point((x_show[i], y_show[i])):
        if i in np.array([6,7,11,12,15,22]):
            ax.annotate(txt, (x_show[i]+12*ratio_adjust, y_show[i]+12*ratio_adjust), fontproperties=my_font)  # right-mid
        elif i in np.array([1,3,4,14,16,18]):
            ax.annotate(txt, (x_show[i], y_show[i]-12*ratio_adjust), fontproperties=my_font)  # right-top
        elif i in np.array([2,10,13]):
            ax.annotate(txt, (x_show[i]+6*ratio_adjust, y_show[i]+24*ratio_adjust), fontproperties=my_font)  # right-bottom
        elif i in np.array([5]):
            ax.annotate(txt, (x_show[i]-12*ratio_adjust, y_show[i]+33*ratio_adjust), fontproperties=my_font)  # mid-bottom
        elif i in np.array([0,17,21]):
            ax.annotate(txt, (x_show[i]-42*ratio_adjust, y_show[i]-6*ratio_adjust), fontproperties=my_font)  # left-top
        elif i in np.array([8]):
            ax.annotate(txt, (x_show[i]-54*ratio_adjust, y_show[i]+12*ratio_adjust), fontproperties=my_font)  # left-mid
        elif i in np.array([9,19]):
            ax.annotate(txt, (x_show[i]-42*ratio_adjust, y_show[i]+24*ratio_adjust), fontproperties=my_font)  # left-bottom
        elif i in np.array([20]):
            ax.annotate(txt, (x_show[i]-66*ratio_adjust, y_show[i]+24*ratio_adjust), fontproperties=my_font)  # left-bottom 蘇澳新
#==================================================================================
# Special stations
# 和仁=崇德
# 和仁: 121.71192	24.24219
# 崇德: 121.65536	24.17199
# 八掌溪橋: 120.386061  23.404249
x_special = np.array([121.71192, 121.65536])
y_special = np.array([24.24219,  24.17199])
# Modify coordinates of important stations for plotting
x_special = (x_special - 118) * 600   # Unify x-coordinates with radar image
y_special = 3600 - (y_special - 20.5) * 600   # Unify y-coordinates with radar image (need to flip y-coordinates)
ax.scatter(x_special, y_special, s=50, c='k')
ax.scatter(x_special, y_special, s=20, c='white')

if Path(vertices_show).contains_point((x_special[0], y_special[0])):
    ax.annotate("和仁", (x_special[0]+6*ratio_adjust, y_special[0]+24*ratio_adjust), fontproperties=my_font)  # right-bottom
if Path(vertices_show).contains_point((x_special[1], y_special[1])):
    ax.annotate("崇德", (x_special[1]+6*ratio_adjust, y_special[1]+24*ratio_adjust), fontproperties=my_font)  # right-bottom
#==================================================================================
# Special bridges
# 八掌溪橋: 120.386061  23.404249
x_special = np.array([120.386061])
y_special = np.array([23.404249])
# Modify coordinates of important stations for plotting
x_special = (x_special - 118) * 600   # Unify x-coordinates with radar image
y_special = 3600 - (y_special - 20.5) * 600   # Unify y-coordinates with radar image (need to flip y-coordinates)
ax.scatter(x_special, y_special, marker='*', s=50, c='r')

if Path(vertices_show).contains_point((x_special[0], y_special[0])):
    ax.annotate("八掌溪橋", (x_special[0]+6*ratio_adjust, y_special[0]+24*ratio_adjust), fontproperties=my_font)  # right-bottom
#==================================================================================
# Set x, y lim (zoom in)
ax.set_xlim(mid_x-radius, mid_x+radius)
ax.set_ylim(mid_y+radius, mid_y-radius)
ax.add_patch(poly1_mod) # Plot polygon (affected area)
# Custom title textbox
rect = Rectangle((mid_x-radius*0.99, mid_y-radius+2.2), radius*1.98, 66*ratio_adjust, linewidth=3, edgecolor=clr, facecolor='white')
ax.add_patch(rect)
ax.text(0.5, 0.924, f"{tt0} 大雷雨影響範圍", transform=ax.transAxes, fontsize=20, ha='center', fontproperties=my_font)
# Plot radar time text box
ax.text(0.97, 0.02, f"{ttR} 雷達回波", transform=ax.transAxes,
    fontsize=8, ha='right', va='bottom', bbox=dict(edgecolor='none', facecolor='white'), fontproperties=my_font)
#==================================================================================
# Add colorbar below the main plot
cax = fig.add_axes([0.01, 0.04, 1., 0.0725])  # [left, bottom, width, height]; Adjust the position and size as needed
cax.imshow(radar_colorbar)  # Plot colorbar image

# Hide axes
ax.axis('off')
cax.axis('off')
# ============================================================================figdir
# Save the image (File name has effective time and radar time on it)
#plt.savefig(f"./output/TS{tt0.replace(':', '').replace(' ','').replace('-','')}_R{ttR.replace(':', '').replace(' ','').replace('-','')}.png", bbox_inches='tight', dpi=300)
# plt.savefig(f"./pic_output/TS{tt0.replace(':', '').replace(' ','').replace('-','')}_R{ttR.replace(':', '').replace(' ','').replace('-','')}.png", bbox_inches='tight', dpi=300)
plt.savefig(figdir+f"TS{tt0.replace(':', '').replace(' ','').replace('-','')}_R{ttR.replace(':', '').replace(' ','').replace('-','')}.png", bbox_inches='tight', dpi=300)
plt.close(fig)
# # ============================================================================
# # Send Line message with picture
#if tra_affected == True:
    #linemsg_url = 'https://notify-api.line.me/api/notify'
    #token = 'HTOmJSTcJFSaL2RMpeue46WPMc31AuLS8Kz3oH8N0JM'   # for private test
    #token = 'g2gGl76tsQSlMUW0RZum7kxQs4HDhbmedbBZ71JeM1D'   # 台鐵值班群組測試用
    #token = '6qIvGmpDi60lOyLYwnnUPOvkjYoujKVRCQK2T7462Hu'  # 台鐵大群組用

    #headers = {
        #'Authorization': 'Bearer ' + token
    #}

    #data = {
        #'message': linemsg_str
    #}

    # Open the image file
    #with open(figdir+f"TS{tt0.replace(':', '').replace(' ','').replace('-','')}_R{ttR.replace(':', '').replace(' ','').replace('-','')}.png", 'rb') as image:
        #imageFile = {'imageFile': image}
            
        # Send the POST request
        #r = requests.post(linemsg_url, headers=headers, data=data, files=imageFile)




# 
# 
# 
# 



#內部參考:所有大雷雨即時訊息+QPF
if tra_affected == True:
    linemsg_url = 'https://notify-api.line.me/api/notify'
    # token = 'HTOmJSTcJFSaL2RMpeue46WPMc31AuLS8Kz3oH8N0JM'
    # token = 'wYhJwUSpdtCEkZ1j0htjjEWlpvy7riSY5L6Hqgdsxlw'
    # token = 'e2OVWOGXYrdrxdSqgKSAEHT3hNK7kI1QAtX4UnMHMkw'  # 台鐵值班群組用
    token = 'NPQ8VyJ7G3Ps9uNBBlP2CseapRPv7UhjQm4TlpKLzw8' # 台鐵對外群組用
    
    headers = {
        'Authorization': 'Bearer ' + token
    }
    
    data = {
    'message': linemsg_str_pub
    }
    
    # Open the image file
    with open(figdir+f"TS{tt0.replace(':', '').replace(' ','').replace('-','')}_R{ttR.replace(':', '').replace(' ','').replace('-','')}.png", 'rb') as image:
        imageFile = {'imageFile': image}
            
        # Send the POST request
        r = requests.post(linemsg_url, headers=headers, data=data, files=imageFile)
# # ============================================================================
linemsg_str = "\n\n" # Clear the message


# 
# 
# 
# 
# 
# 
# 
# 



linemsg_url = 'https://notify-api.line.me/api/notify'
# token = 'HTOmJSTcJFSaL2RMpeue46WPMc31AuLS8Kz3oH8N0JM'
# token = 'wYhJwUSpdtCEkZ1j0htjjEWlpvy7riSY5L6Hqgdsxlw'
token = 'e2OVWOGXYrdrxdSqgKSAEHT3hNK7kI1QAtX4UnMHMkw'  # 台鐵值班群組用
# token = 'NPQ8VyJ7G3Ps9uNBBlP2CseapRPv7UhjQm4TlpKLzw8' # 台鐵對外群組用

headers = {
    'Authorization': 'Bearer ' + token
}

data = {
'message': linemsg_str_ref
}

if tra_affected == False:
        # Open the image file
        with open(figdir+f"TS{tt0.replace(':', '').replace(' ','').replace('-','')}_R{ttR.replace(':', '').replace(' ','').replace('-','')}.png", 'rb') as image:
            imageFile = {'imageFile': image}
            
            # Send the POST request
            r = requests.post(linemsg_url, headers=headers, data=data, files=imageFile)
    
else:
    r = requests.post(linemsg_url, headers=headers, data=data)


print('Finish')
