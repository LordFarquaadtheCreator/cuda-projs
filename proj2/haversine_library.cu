#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

__global__ void haversine_distance_kernel(int size, const double *x1,const double *y1,
    const double *x2,const double *y2, double *dist)
{
  double deltaLat = floor(*x1 - *x2);
  double deltaLong = floor(*y1 - *y2);
  double cosinesMultiplied = cos(*x1) * cos(*x2);
  double sinDeltaLat = 2.0 * sin(deltaLat / 2.0) * cos(deltaLat / 2.0);
  double sinDeltaLong = 2.0 * sin(deltaLong / 2.0) * cos(deltaLong / 2.0);
  
  double a = sinDeltaLat + cosinesMultiplied * sinDeltaLong;
  double R = 6378.0; // kilometers
  double c = 2.0 * atan2(sqrt(a), sqrt((1.0 - a)));
  
  *dist = R * c;
  return;
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
