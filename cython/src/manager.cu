/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
using namespace std;

GPUAdder::GPUAdder (int* array_, int length_) {
  array = array_;
  length = length_;
  int size = length * sizeof(int);
  cudaDeviceReset();
  cudaDeviceProp prop;
  cudaError_t err;
  err = cudaGetDeviceProperties(&prop, 0);
  assert(prop.canMapHostMemory);
  err = cudaSetDeviceFlags(cudaDeviceMapHost);
  //cout << "Trying to cudaSetDeviceFlags(cudaDeviceMapHost)" << endl;
  assert(err == 0);
  err = cudaHostRegister( (void*)array, size, cudaHostRegisterMapped );
  //cout << "Trying to cudaHostRegister( (void*)array, size, cudaHostRegisterMapped )" << endl;
  assert(err == 0);
  err = cudaHostGetDevicePointer( (void**)&array_d, (void*)array, 0);
  //cout << "Trying to cudaHostGetDevicePointer( (void**)&array_d, (void*)array, 0)" << endl;

  // Prefetch array to GPU from CPU
  err = cudaStreamAttachMemAsync(NULL, array_d, 0, cudaMemAttachGlobal);
  cudaStreamSynchronize(NULL);
  //cout << "End of constructor" << endl;
  //assert(err == 0);
}

void GPUAdder::increment() {
  kernel_add_one<<<64, 64>>>(array_d, length);
  //cudaError_t err = cudaGetLastError();
  //assert(err == 0);
}

void GPUAdder::retreive() {
  // Prefetch array to CPU from GPU
  cudaError_t err = cudaStreamAttachMemAsync(NULL, array_d, 0, cudaMemAttachHost);
  cudaStreamSynchronize(NULL); 
  //cout << "End of retreive()" << endl;
  //cudaDeviceSynchronize();
}

void GPUAdder::retreive_to(int* array_, int length_) {
  assert(length == length_);
  int size = length * sizeof(int);
  cudaError_t err = cudaStreamAttachMemAsync(NULL, array_d, 0, cudaMemAttachHost);
  cudaStreamSynchronize(NULL); 
  memcpy( array_, array, size );
  //cout << "End of retreive_to()" << endl;
}

GPUAdder::~GPUAdder() {
}
