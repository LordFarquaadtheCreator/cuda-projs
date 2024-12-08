#include <sstream>
#include <iostream>
#include <cuda_runtime.h>

__global__ void haversine_distance_kernel(int size, const double *x1, const double *y1,
                                          const double *x2, const double *y2, double *dist)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    
    double lon1 = x1[idx];
    double lat1 = y1[idx];
    double lon2 = x2[idx];
    double lat2 = y2[idx];

    double R = 6371000.0;

    double phi_1 = lat1 * (M_PI / 180);
    double phi_2 = lat2 * (M_PI / 180);
    double delta_phi = (lat2 - lat1) * (M_PI / 180);
    double delta_lambda = (lon2 - lon1) * (M_PI / 180);
    
    double a = pow(sin(delta_phi / 2.0), 2.0) + cos(phi_1) * cos(phi_2) * pow(sin(delta_lambda / 2.0), 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    
    double meters = (R * c);
    double kilometers = meters / 1000.0;

    // printf("The haversine distance is: %f\n",kilometers);
    dist[idx] = kilometers;
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
