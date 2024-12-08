import os
import sys
import numpy as np
import cudf as cu
sys.path.append('./build') # Ensure this directory exists
import haversine_library # Ensure this import is properly configured
# Import any other necessary libraries 

DIR = "/data/csc59866_f24/tlcdata/" # Ensure this directory exists and has all the data
desired_cols = ["Start_Lon", "Start_Lat", "End_Lon", "End_Lat"] # Ensure these columns exist in the dataframe
main_df = cu.DataFrame()

# Loop over all files
for file_name in os.listdir(DIR):
    file_path = os.path.join(DIR, file_name)
    data = cu.read_parquet(file_path)

    data = data[desired_cols]
    
    # Filter out non-NYC locations
    inbounds = (
            (data["Start_Lon"] >= -74.15) & (data["Start_Lon"] <= -73.7004) &
            (data["End_Lon"] >= -74.15) & (data["End_Lon"] <= -73.7004) &
            (data["Start_Lat"] >= 40.5774) & (data["Start_Lat"] <= 40.9176) &
            (data["End_Lat"] >= 40.5774) & (data["End_Lat"] <= 40.9176)
            )
    inbound_data = data[inbounds]

    # Merge with the main dataframe
    main_df = cu.concat([main_df, data], ignore_index=True)

# Convert columns to numpy arrays
x1=main_df['Start_Lon'].to_numpy()
y1=main_df['Start_Lat'].to_numpy()
x2=main_df['End_Lon'].to_numpy()
y2=main_df['End_Lat'].to_numpy()
size=len(x1)
dist=np.zeros(size)
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist) # Ensure this function exists

# Print results
print(dist)
