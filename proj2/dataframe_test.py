import sys
sys.path.append('./build')

from haversine_library import haversine_distance
import cudf
import numpy as np


NYC_BOUNDING_BOX = [-74.15, 40.5774, -73.7004, 40.9176] #see dosc for more info

def filter_nyc(df): 
    return df[
        (df['Start_Lon'] >= NYC_BOUNDING_BOX[0]) & (df['Start_Lon'] <= NYC_BOUNDING_BOX[2]) &
        (df['Start_Lat'] >= NYC_BOUNDING_BOX[1]) & (df['Start_Lat'] <= NYC_BOUNDING_BOX[3]) &
        (df['End_Lon'] >= NYC_BOUNDING_BOX[0]) & (df['End_Lon'] <= NYC_BOUNDING_BOX[2]) &
        (df['End_Lat'] >= NYC_BOUNDING_BOX[1]) & (df['End_Lat'] <= NYC_BOUNDING_BOX[3])
    ]

#the loop
file_paths = [f'yellowcab_2009/yellow_tripdata_2009-{str(i).zfill(2)}.parquet' for i in range(1, 13)]
dfs = []
for file_path in file_paths:
    for chunk in cudf.read_parquet(file_path):
        df_filtered = filter_nyc(chunk)
        dfs.append(df_filtered)

#one big datagrame
combined_df = cudf.concat(dfs, ignore_index=True)
start_lon = combined_df['Start_Lon'].to_numpy()
start_lat = combined_df['Start_Lat'].to_numpy()
end_lon = combined_df['End_Lon'].to_numpy()
end_lat = combined_df['End_Lat'].to_numpy()

#this kinda assumes it works. dunno if it does
distances = haversine_library.haversine_distance(start_lon, start_lat, end_lon, end_lat)
#i think i have to add the rest of the distances to a dataframe.