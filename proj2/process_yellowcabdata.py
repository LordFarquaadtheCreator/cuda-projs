import os
import pandas as pd

dir = "yellowcab_2009"
desired_cols = ["Start_Lon", "Start_Lat", "End_Lon", "End_Lat"]
main_df = pd.DataFrame() # init df that will have all 12 months
# note that both Start_Lon and End_Lon must lie between the bounding range of west and east: -74.15, -73.7004
# Start_Lat and End_Lat must lie between bounding range of north and south: 40.5774, 40.9176
for file_name in os.listdir(dir):
    file_path = os.path.join(dir, file_name)
    data = pd.read_parquet(file_path)

    data = data[desired_cols]
    # conditions to ensure coords are inbounds
    inbounds = (
            (data["Start_Lon"] >= -74.15) & (data["Start_Lon"] <= -73.7004) &
            (data["End_Lon"] >= -74.15) & (data["End_Lon"] <= -73.7004) &
            (data["Start_Lat"] >= 40.5774) & (data["Start_Lat"] <= 40.9176) &
            (data["End_Lat"] >= 40.5774) & (data["End_Lat"] <= 40.9176)
            )
    inbound_data = data[inbounds]
    main_df = pd.concat([main_df, data], ignore_index=True)
# at this point all DFs are combined
df_to_numpy = main_df.to_numpy()
print(df_to_numpy.shape)
