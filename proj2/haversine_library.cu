#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

__global__ void haversine_distance_kernel(int size, const double *x1, const double *y1,
                                          const double *x2, const double *y2, double *dist)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    double lat1 = x1[idx];
    double lon1 = y1[idx];
    double lat2 = x2[idx];
    double lon2 = y2[idx];

    printf("These are the values: %f, %f, %f, %f\n", lat1, lon1, lat2, lon2);

    double R = 6378.0; 
    double deltaLat = (lat2 - lat1) * (M_PI / 180.0); 
    double deltaLon = (lon2 - lon1) * (M_PI / 180.0); 
    double lat1Rad = lat1 * (M_PI / 180.0);
    double lat2Rad = lat2 * (M_PI / 180.0);

    double a = sin(deltaLat / 2.0) * sin(deltaLat / 2.0) +
               cos(lat1Rad) * cos(lat2Rad) * sin(deltaLon / 2.0) * sin(deltaLon / 2.0);
    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));

    printf("The haversine distance is: %f\n", R * c);
    dist[idx] = R * c;
}



void run_kernel(int size, const double *x1,const double *y1, const double *x2,const double *y2, double *dist)
   
{
  dim3 dimBlock(1024);
  printf("in run_kernel dimBlock.x=%d\n",dimBlock.x);

  dim3 dimGrid(ceil((double)size / dimBlock.x));
  
  haversine_distance_kernel<<<dimGrid, dimBlock>>>
    (size,x1,y1,x2,y2,dist);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
