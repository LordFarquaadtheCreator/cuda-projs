#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

void run_kernel(int size, const double *x1,const double *y1, const double *x2,const double *y2, double *dist);

float calc_time(const char *msg,timeval t0, timeval t1)
{
 	long d = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
 	float t=(float)d/1000;
 	if(msg!=NULL)
 		printf("%s ...%10.3f\n",msg,t);
 	return t;
}

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void haversine_distance(int size,pybind11::array_t<double> x1_v,pybind11::array_t<double> y1_v,
    pybind11::array_t<double> x2_v,pybind11::array_t<double> y2_v,pybind11::array_t<double> dist_v)
{
  assert(x1_v.request().ndim==1);
  assert(x2_v.request().ndim==1);
  assert(y1_v.request().ndim==1);
  assert(y2_v.request().ndim==1);
  assert(dist_v.request().ndim==1);

  timeval s0, s1, s2, s3;

  double *d_x1,*d_y1,*d_x2,*d_y2,*d_dist;
  // cudaMalloc below allocates memory onto GPU from host
  gettimeofday(&s0, NULL);
  HANDLE_ERROR( cudaMalloc(&d_x1, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_y1, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_x2, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_y2, size * sizeof(double)) );
  HANDLE_ERROR( cudaMalloc(&d_dist, size * sizeof(double)) );

  double* h_x1 = reinterpret_cast<double*>(x1_v.request().ptr);
  double* h_y1 = reinterpret_cast<double*>(y1_v.request().ptr);
  double* h_x2 = reinterpret_cast<double*>(x2_v.request().ptr);
  double* h_y2 = reinterpret_cast<double*>(y2_v.request().ptr);
  double* h_dist = reinterpret_cast<double*>(dist_v.request().ptr);
  // copy (cpy) data from host to device (cudaMemcpyHostToDevice) 
  HANDLE_ERROR( cudaMemcpy(d_x1, h_x1, size * sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_y1, h_y1, size * sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_x2, h_x2, size * sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(d_y2, h_y2, size * sizeof(double), cudaMemcpyHostToDevice) );
  gettimeofday(&s1, NULL);
  calc_time("transferring from CPU to GPU\n", s0, s1);
  //printf("before\n");
  // run the kernel
  run_kernel(size,d_x1,d_y1,d_x2,d_y2,d_dist);
  gettimeofday(&s2, NULL);
  calc_time("running kernel\n", s1, s2);
  //printf("after\n");
  //transfer data back from device to host or gpu to cpu (cudaMemcpyDeviceToHost)
  HANDLE_ERROR( cudaMemcpy(h_dist, d_dist, size * sizeof(double), cudaMemcpyDeviceToHost) );
  gettimeofday(&s3, NULL);
  calc_time("GPU to CPU\n", s2, s3);

  HANDLE_ERROR( cudaFree(d_x1) );
  HANDLE_ERROR( cudaFree(d_y1) );
  HANDLE_ERROR( cudaFree(d_x2) );
  HANDLE_ERROR( cudaFree(d_y2) );
  HANDLE_ERROR( cudaFree(d_dist) );

}

PYBIND11_MODULE(haversine_library, m)
{
  m.def("haversine_distance", haversine_distance);
}
