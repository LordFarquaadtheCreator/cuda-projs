import os
import sys
import numpy as np
import cudf as cu
sys.path.append('./build')
import haversine_library 
from utils import haversine, timer

class question4:
    def __init__(self):
        self.main_df = cu.DataFrame()
        self.x1, self.x2, self.y1, self.y2 = None, None, None, None

    def make_big_df(self):
        DIR = "/data/csc59866_f24/tlcdata/" 
        desired_cols = ["Start_Lon", "Start_Lat", "End_Lon", "End_Lat"]

        for file_name in os.listdir(DIR):
            file_path = os.path.join(DIR, file_name)
            data = cu.read_parquet(file_path)

            data = data[desired_cols]
            
            inbounds = (
                    (data["Start_Lon"] >= -74.15) & (data["Start_Lon"] <= -73.7004) &
                    (data["End_Lon"] >= -74.15) & (data["End_Lon"] <= -73.7004) &
                    (data["Start_Lat"] >= 40.5774) & (data["Start_Lat"] <= 40.9176) &
                    (data["End_Lat"] >= 40.5774) & (data["End_Lat"] <= 40.9176)
                    )

            self.main_df = cu.concat([self.main_df, data[inbounds]], ignore_index=True)
            
            self.x1 = self.main_df['Start_Lon'].to_numpy()
            self.y1 = self.main_df['Start_Lat'].to_numpy()
            self.x2 = self.main_df['End_Lon'].to_numpy()
            self.y2 = self.main_df['End_Lat'].to_numpy()
    
    @timer
    def cuda_haversine(self):
        size=len(self.x1)
        dist=np.zeros(size)
        haversine_library.haversine_distance(size, self.x1, self.y1, self.x2, self.y2, dist) 

        assert len(dist) == size
        print("CUDA Havertsine Distances Calculated!")

    @timer
    def cpu_haversine(self):
        size = len(self.x1)
        distances = []

        for i in range(size):
            distances.append(haversine(self.x1[i], self.x2[i], self.y1[i], self.y2[i]))

        assert len(distances) == size
        print("Python Haversine Distances Calculated!")

if __name__ == "__main__":
    q = question4()
    q.make_big_df()
    q.cuda_haversine()
    q.cpu_haversine()