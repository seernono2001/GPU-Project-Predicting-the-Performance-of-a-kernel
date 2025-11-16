#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <string.h>

__global__ void addvector(int*, int*, int*, int);
__global__ void subtractvector(int*, int*, int*, int);
__global__ void MatrixMulKernel(int*, int*, int*, int);
__global__ void minReductionKernel(int*, int*, int);
void matrixMulCPU(int*, int*, int*, int);
void addVectorCPU(int*, int*, int*, int);
void subtractVectorCPU(int*, int*, int*, int);
void minReductionCPU(int*, int*, int);
void getDeviceInformation();

int main(int argc, char* argv[]) {
	int i;
	int num = 0; // number of elements in the arrays
	int* a, * b, * c; // arrays at host
	int* ad, * bd, * cd; // arrays at device
	int THREADS = 0; // user decides number of threads per block  
	int total_elements = 0;

	// to measure the time
	float multi_time_taken = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double single_time_taken = 0;
	clock_t single_start, single_end;

	char* op;

	int numblocks;
	int threadsperblock;


	if (argc != 5) {
		printf("usage: addvec numelements threads_per_block num_blocks\n type of op");
		exit(1);
	}

	num = atoi(argv[1]);
	THREADS = atoi(argv[2]);
	numblocks = atoi(argv[3]);
	op = argv[4];
	threadsperblock = THREADS;
	int total_threads = threadsperblock * numblocks;
	total_elements = num;

	if (strcmp(op, "addition") != 0 && strcmp(op, "subtraction") != 0 && 
	    strcmp(op, "multiplication") != 0 && strcmp(op, "reduction") != 0) {
		printf("Unknown operation: %s\n", op);
		printf("Available operations: addition, subtraction, multiplication, reduction\n");
		// exit
		exit(1);
	}

	if(strcmp(op, "multiplication") == 0){
		if (num > 2048){
			printf("WARNING: Matrix size %d too large, limiting to 2048x2048\n", num);
			num = 2048;
		}
		total_elements = num * num;

		int coverage = numblocks * threadsperblock;
		// printf("\n--- Matrix Multiplication Configuration ---\n");
		// printf("Matrix size: %d x %d = %d elements\n", num, num, total_elements);
		// printf("Grid: %d x %d blocks\n", numblocks, numblocks);
		// printf("Block: %d x %d threads\n", threadsperblock, threadsperblock);
		// printf("Coverage: %d x %d elements\n", coverage, coverage);
		

		// check if there are enough threads to execute all multiplications
		if (coverage < num) {
			printf("ERROR: Insufficient coverage! Will only compute %dx%d region.\n", coverage, coverage);
			printf("Need at least %d blocks per dimension for full coverage.\n", (num + threadsperblock - 1) / threadsperblock);
			exit(1);
		} else if (coverage > num) {
			printf("WARNING: INFO: Over-provisioned. Some threads will be idle.\n");
		}

	}
	else if(strcmp(op, "reduction") == 0) {
		total_elements = num;
		if (total_threads < total_elements) {
			printf("ERROR: Insufficient threads!\n");
			printf("Need at least %d blocks for %d elements with %d threads/block\n", (total_elements + THREADS - 1) / THREADS, total_elements, THREADS);
			exit(1);
		}
	}
	else{
		if (total_threads < total_elements) {
			printf("ERROR: Insufficient threads!\n");
			printf("Need at least %d blocks for %d elements with %d threads/block\n", (total_elements + THREADS - 1) / THREADS, total_elements, THREADS);
			exit(1);
		}
	}


	a = (int*)malloc(total_elements * sizeof(int));
	if (!a) {
		printf("Cannot allocate array a with %d elements\n", total_elements);
		exit(1);
	}

	b = (int*)malloc(total_elements * sizeof(int));
	if (!b) {
		printf("Cannot allocate array b with %d elements\n", total_elements);
		exit(1);
	}

	c = (int*)malloc(total_elements * sizeof(int));
	if (!c) {
		printf("Cannot allocate array c with %d elements\n", total_elements);
		exit(1);
	}

	getDeviceInformation();
	printf("\n");

	//Fill out arrays a and b with some random numbers
	srand(time(0));
	for (i = 0; i < total_elements; i++) {
		a[i] = rand() % total_elements;
		b[i] = rand() % total_elements;
	}

	//Now zero C[] in preparation for single thread version
	for (i = 0; i < total_elements; i++) {
		c[i] = 0;
	}
	
	dim3 grid, block;

	//assume a block can have THREADS threads
	if (strcmp(op, "multiplication") == 0) {
		// multiplication uses 2D grid and block
		block = dim3(threadsperblock, threadsperblock, 1);
		grid = dim3(numblocks, numblocks, 1);
	} else {
		// addition uses 1D
		block = dim3(threadsperblock, 1, 1);
		grid = dim3(numblocks, 1, 1);
	}

	cudaMalloc((void**)&ad, total_elements * sizeof(int));
	if (!ad) {
		printf("cannot allocated array ad of %d elements\n", total_elements);
		exit(1);
	}
	cudaMalloc((void**)&bd, total_elements * sizeof(int));
	if (!bd) {
		printf("cannot allocated array bd of %d elements\n", total_elements);
		exit(1);
	}
	cudaMalloc((void**)&cd, total_elements * sizeof(int));
	if (!cd) {
		printf("cannot allocated array cd of %d elements\n", total_elements);
		exit(1);
	}

	// CPU version   

  	single_start = clock(); // start measuring
	//Launch the kernel
	if (strcmp(op, "addition") == 0) {
		addVectorCPU(a, b, c, total_elements);
	}
	else if( strcmp(op, "subtraction") == 0){
		subtractVectorCPU(a, b, c, total_elements);
	}
	else if (strcmp(op, "multiplication") == 0) {
		matrixMulCPU(a, b, c, num);
	}
	else if (strcmp(op, "reduction") == 0) {
		minReductionCPU(a, c, total_elements);
	}

	single_end = clock();  // end of measuring
	single_time_taken = ((double)(single_end - single_start)) / CLOCKS_PER_SEC;


	printf("Single thread time = %lf secs\n", single_time_taken);
	// // check how many elements are calculated
	// int computed = 0;
	// for (i = 0; i < total_elements; i++) {
	// 	if (c[i] == (a[i] + b[i])) {
	// 		computed++;
	// 	}
	// }
	// printf("Single thread computed %d / %d elements (%.2f%%)\n", 
	// 	computed, total_elements, 100.0 * computed / total_elements);

	//Now zero C[] in preparation for kernel version
	for (i = 0; i < total_elements; i++) {
		c[i] = 0;
	}

	// The kernel version

	printf("GPU: %d blocks of %d threads each\n", numblocks, threadsperblock);

	//mov a and b to the device
	cudaMemcpy(ad, a, total_elements * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, total_elements * sizeof(int), cudaMemcpyHostToDevice);

	//kernel warm-up
	if (strcmp(op, "addition") == 0) {
		addvector << <numblocks, threadsperblock >> > (ad, bd, cd, total_elements);
	}
	else if( strcmp(op, "subtraction") == 0){
		subtractvector << <numblocks, threadsperblock >> > (ad, bd, cd, total_elements);
	}
	else if (strcmp(op, "multiplication") == 0) {
		MatrixMulKernel << <grid, block >> > (ad, bd, cd, num);
	}
	else if (strcmp(op, "reduction") == 0) {
		minReductionKernel << <numblocks, threadsperblock >> > (ad, cd, total_elements);
	}
	cudaDeviceSynchronize();

	//start measuring time for GPU
	cudaEventRecord(start); 

	//Launch the kernel
	if (strcmp(op, "addition") == 0) {
		addvector << <numblocks, threadsperblock >> > (ad, bd, cd, total_elements);
	}
	else if( strcmp(op, "subtraction") == 0){
		subtractvector << <numblocks, threadsperblock >> > (ad, bd, cd, total_elements);
	}
	else if (strcmp(op, "multiplication") == 0) {
		MatrixMulKernel << <grid, block >> > (ad, bd, cd, num);
	}
	else if (strcmp(op, "reduction") == 0) {
		minReductionKernel << <numblocks, threadsperblock >> > (ad, cd, total_elements);
	}

	cudaDeviceSynchronize(); //block host till device is done.

	cudaEventRecord(stop);  // end of measuring
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&multi_time_taken, start, stop);

	double mul_time_taken = multi_time_taken / 1000.0;
	printf("Kernel time = %lf secs\n", mul_time_taken);

	//bring data back 
	cudaMemcpy(c, cd, total_elements * sizeof(int), cudaMemcpyDeviceToHost);

	//check the result is correct
	if (strcmp(op, "addition") == 0) {
		for (i = 0; i < total_elements; i++) {
			if (c[i] != (a[i] + b[i])) {
				printf("Incorrect result for element c[%d] = %d\n", i, c[i]);
			}
		}
	}
	else if(strcmp(op, "subtraction") == 0){
		for (i = 0; i < total_elements; i++) {
			if (c[i] != (a[i] - b[i])) {
				printf("Incorrect result for element c[%d] = %d\n", i, c[i]);
			}
		}		
	}
	else if (strcmp(op, "multiplication") == 0) {
		int* c_ref = (int*)malloc(total_elements * sizeof(int));
		matrixMulCPU(a, b, c_ref, num);
		
		int errors = 0;
		int zero_count = 0;
		
		for (i = 0; i < total_elements; i++) {
			if (c[i] == 0) 
				zero_count++;
			
			if (c[i] != c_ref[i]) {
				if (errors < 10) {
					int row = i / num;
					int col = i % num;
					printf("Error at c[%d,%d]: expected %d, got %d\n", 
						row, col, c_ref[i], c[i]);
				}
				errors++;
			}
		}
		
		if (errors != 0) {
			printf("✗ Found %d errors (%.2f%% incorrect)\n", errors, 100.0 * errors / total_elements);
			printf("  %d elements are zero (%.2f%% - possibly not computed)\n", zero_count, 100.0 * zero_count / total_elements);
		free(c_ref);
		}
	}
	else if (strcmp(op, "reduction") == 0) {
		int gpu_min = c[0];
		for (i = 1; i < numblocks; i++) {
			if (c[i] < gpu_min) {
				gpu_min = c[i];
			}
		}
		int cpu_min = a[0];
		for (i = 1; i < total_elements; i++) {
			if (a[i] < cpu_min) {
				cpu_min = a[i];
			}
		}
		printf("\nCPU minimum: %d\n", cpu_min);
		printf("GPU minimum: %d\n", gpu_min);
		if (gpu_min != cpu_min) {
			printf("Incorrect result! Expected %d, got %d\n", cpu_min, gpu_min);
		} else {
			printf("Result verified correctly!\n");
		}
	}

	free(a);
	free(b);
	free(c);

	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);

	// Speedup
	double speedup = single_time_taken / mul_time_taken;
	printf("Speedup: %.2fx\n", speedup);

	return 0;
}

__global__ void MatrixMulKernel(int* Md, int* Nd, int* Pd, int Width){
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx >= Width || ty >= Width) 
		return;
    
    int Pvalue = 0;
    for(int k=0; k<Width; ++k){
        int Mdelement = Md[ty * Width + k];
        int Ndelement = Nd[k * Width + tx];
        Pvalue += Mdelement * Ndelement;
    }
    Pd[ty * Width + tx] = Pvalue;
}

__global__ void addvector(int* a, int* b, int* c, int n) {
	int index;

	index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

__global__ void subtractvector(int* a, int* b, int* c, int n) {
	int index;

	index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n) {
		c[index] = a[index] - b[index];
	}
}

__global__ void minReductionKernel(int* input, int* output, int n) {
	__shared__ int sdata[1024];
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n) {
		sdata[tid] = input[index];
	} else {
		sdata[tid] = 2147483647; // INT_MAX
	}
	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			if (sdata[tid + s] < sdata[tid]) {
				sdata[tid] = sdata[tid + s];
			} 
		}
		__syncthreads();
	}
	if (tid == 0) {
		output[blockIdx.x] = sdata[0];
	}
}

void matrixMulCPU(int* M, int* N, int* P, int Width) {
	for (int row = 0; row < Width; ++row) {
		for (int col = 0; col < Width; ++col) {
			int Pvalue = 0;
			for (int k = 0; k < Width; ++k) {
				int Melement = M[row * Width + k];
				int Nelement = N[k * Width + col];
				Pvalue += Melement * Nelement;
			}
			P[row * Width + col] = Pvalue;
		}
	}
}

void addVectorCPU(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

void subtractVectorCPU(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		c[i] = a[i] - b[i];
	}
}

void minReductionCPU(int* a, int* c, int n) {
	int min = a[0];
	for (int i = 1; i < n; i++) {
		if (a[i] < min) {
			min = a[i];
		}
	}
	c[0] = min;
}

void getDeviceInformation() {
	cudaError_t error;
	cudaDeviceProp dev;
	int dev_cnt = 0;

	int currentDevice;
	cudaGetDevice(&currentDevice);

	// return device numbers with compute capability >= 1.0
	error = cudaGetDeviceCount(&dev_cnt);
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	printf("Number of devices: %d\n", dev_cnt);
	printf("Currently using Device: %d\n", currentDevice);

	// Get properties of each device
	error = cudaGetDeviceProperties(&dev, currentDevice);
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	printf("\nDevice %d:\n", currentDevice);
	printf("name: %s\n", dev.name);
	printf("total global memory(KB): %ld\n", dev.totalGlobalMem / 1024);
	printf("shared mem per block: %d\n", dev.sharedMemPerBlock);
	printf("warp size: %d\n", dev.warpSize);
	printf("clock rate(KHz): %d\n", dev.clockRate);
}
