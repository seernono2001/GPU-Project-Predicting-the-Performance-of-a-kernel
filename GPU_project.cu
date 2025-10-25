#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <string.h>

__global__ void addvector(int *, int *, int *, int);

int main(int argc, char *argv[]){
  int i;
  int num = 0; // number of elements in the arrays
  int *a, *b, *c; // arrays at host
  int *ad, *bd, *cd; // arrays at device
  int THREADS = 0; // user decides number of threads per block  

  // to measure the time
  float single_time_taken = 0;
  float multi_time_taken = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  char *op;

  int numblocks;
  int threadsperblock;


  if(argc != 5){
    printf("usage: addvec numelements threads_per_block num_blocks\n type of op");
    exit(1);
  }

  num = atoi(argv[1]);
  THREADS = atoi(argv[2]);
  numblocks = atoi(argv[3]);
  op = argv[4];  
  threadsperblock = THREADS;
  int total_threads = threadsperblock * numblocks;

  if(total_threads < num){
    printf("ERROR: Insufficient threads!\n");
    printf("Need at least %d blocks for %d elements with %d threads/block\n", (num + THREADS - 1) / THREADS, num, THREADS);
    exit(1);
  }

  a = (int *)malloc(num*sizeof(int));
  if(!a){
    printf("Cannot allocate array a with %d elements\n", num);
    exit(1);	
  }


  b = (int *)malloc(num*sizeof(int));
  if(!b){
    printf("Cannot allocate array b with %d elements\n", num);
    exit(1);	
  }


  c = (int *)malloc(num*sizeof(int));
  if(!c){
    printf("Cannot allocate array c with %d elements\n", num);
    exit(1);	
  }

  //Fill out arrays a and b with some random numbers
  srand(time(0));
  for( i = 0; i < num; i++){
    a[i] = rand() % num;
    b[i] = rand() % num; 
  }

  //Now zero C[] in preparation for single thread version
  for( i = 0; i < num; i++){
	  c[i] = 0;
  }

  //assume a block can have THREADS threads
  dim3 grid(numblocks, 1, 1);
  dim3 block(threadsperblock, 1, 1);

  cudaMalloc((void **)&ad, num*sizeof(int));
  if(!ad){ 
    printf("cannot allocated array ad of %d elements\n", num);
    exit(1);
  }
  cudaMalloc((void **)&bd, num*sizeof(int));
  if(!bd){
    printf("cannot allocated array bd of %d elements\n", num);
    exit(1);
  }
  cudaMalloc((void **)&cd, num*sizeof(int));
  if(!cd){
    printf("cannot allocated array cd of %d elements\n", num);
    exit(1);
  }

  //mov a and b to the device
  cudaMemcpy(ad, a, num*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(bd, b, num*sizeof(int), cudaMemcpyHostToDevice);

  // The single thread version   
  
  cudaEventRecord(start); // start measuring

  //Launch the kernel
  if(strcmp(op, "addition") == 0){
    addvector<<<1 , 1>>>(ad, bd, cd, num);
  }
   //add more operations here
  else {
    printf("Unknown operation: %s\n", op);
    printf("Available operations: addition\n");
    // free and exit
    free(a); free(b); free(c);
    cudaFree(ad); cudaFree(bd); cudaFree(cd);
    exit(1);
  }

  cudaEventRecord(stop);  // end of measuring
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&single_time_taken, start, stop);
  printf("Single thread time = %lf secs\n", single_time_taken); 

  //Now zero C[] in preparation for kernel version
  for( i = 0; i < num; i++){
    c[i] = 0;
  }
  
  // The kernel version
  
  printf("GPU: %d blocks of %d threads each\n", numblocks, threadsperblock);     

  cudaEventRecord(start); //start measuring time for GPU

  //Launch the kernel
  addvector<<<numblocks , threadsperblock>>>(ad, bd, cd, num);

  //bring data back 
  cudaMemcpy(c, cd, num*sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaEventRecord(stop);  // end of measuring
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&multi_time_taken, start, stop);

  printf("Kernel time = %lf secs\n", multi_time_taken); 

  cudaDeviceSynchronize(); //block host till device is done.

  //check the result is correct
  if(strcmp(op, "addition") == 0){
    for(i=0; i < num; i++){
      if(c[i] != (a[i] + b[i])){
        printf("Incorrect result for element c[%d] = %d\n", i, c[i]);
      }
    }
  }
  //add more operations here
  
  free(a);
  free(b);
  free(c);
  
  cudaFree(ad);
  cudaFree(bd);
  cudaFree(cd);

  // Speedup
  float speedup = single_time_taken / multi_time_taken;
  printf("Speedup: %.2fx\n", speedup);

  return 0;
}

__global__ void addvector(int * a, int * b, int *c, int n){
   int index;

   index = (blockIdx.x * blockDim.x) + threadIdx.x;

   if(index < n){
    c[index] = a[index] + b[index];
  }
}