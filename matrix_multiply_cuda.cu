// mmm_cuda.cu, Crispin Bernier, chb2ab
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

//----------------------------------- Structures and Globals---------------------------------------------

typedef struct {
	int dimension1;
	int dimension2;	
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory  
float *A, *B, *C, *C_CPU;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;

//----------------------------------- host function definitions -----------------------------------------

void allocateAndInitializeAB();
void computeCpuMMM();
void copyMatricesToGPU();
void copyResultFromGPU();
void compareHostAndGpuOutput();
void die(const char *error); 
void check_error(cudaError e);

//----------------------------------- CUDA function definitions -----------------------------------------

void computeGpuMMM();

// For 10,000x10,000 matrices use a 313x313 block grid to completely cover the matrix
const int blocks2d_sz  = 313;
const int threads2d_sz = 32;
//-------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
	// Read in the matrix sizes
	A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 100;
	A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
	B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
	B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;
	C_MD.dimension1 = A_MD.dimension1;
	C_MD.dimension2 = B_MD.dimension2;

	printf("Matrix A is %d-by-%d\n", A_MD.dimension1, A_MD.dimension2);
	printf("Matrix B is %d-by-%d\n", B_MD.dimension1, B_MD.dimension2);
	printf("Matrix C is %d-by-%d\n", C_MD.dimension1, C_MD.dimension2);

	// Initialize A and B to random floats.
	// Initialization is not part of total runtime.
	allocateAndInitializeAB();

	clock_t start;
	clock_t end;
	double elapsed;
//	// matrix matrix multiplication in the CPU for comparison purposes
//	start = clock();	
//	computeCpuMMM();
//	end = clock();
//  elapsed = (end - start) / (double) CLOCKS_PER_SEC;
//  printf("Computation time in the CPU: %f seconds\n", elapsed);

	// GPU matrix multiplication
	start = clock();	
	computeGpuMMM();
	end = clock();
    elapsed = (end - start) / (double) CLOCKS_PER_SEC;
    printf("Computation time in the GPU: %f seconds\n", elapsed);

//  // compare if GPU implementation is correct
//	compareHostAndGpuOutput();
	return 0;
}

// Matrix multiply kernel
__global__ void mm_multiply_kernel(float *A, float *B, float *C, int N) {
	// calculate index into the result matrix
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	// shared matrices to hold a tile from the global matrices
	__shared__ float a_shared[threads2d_sz][threads2d_sz];
	__shared__ float b_shared[threads2d_sz][threads2d_sz];
	// final result of vector product
	float result = 0;
	int tile_iter, k, A_row_glob, A_col_glob, B_row_glob, B_col_glob;
	// iterate over the tiles in the global matrices
	int tile_length = N/blockDim.x;
	for (tile_iter = 0; tile_iter < tile_length; tile_iter++) {
		// calculate the indexes into A and B to load into shared memory
		A_row_glob = blockIdx.y*blockDim.y + threadIdx.y; 
		A_col_glob = tile_iter*blockDim.x + threadIdx.x;

		B_row_glob = tile_iter*blockDim.y + threadIdx.y;
		B_col_glob = blockIdx.x*blockDim.x + threadIdx.x;
		// each thread loads 1 entry from A and B into shared memory
		a_shared[threadIdx.y][threadIdx.x] = A[A_row_glob*N + A_col_glob];
		b_shared[threadIdx.y][threadIdx.x] = B[B_row_glob*N + B_col_glob];
		// wait for all threads in the block to load
		__syncthreads();
		// Unrolled loop to calculate vector product
		result += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
		result += a_shared[threadIdx.y][k+1]  * b_shared[k+1][threadIdx.x];
		result += a_shared[threadIdx.y][k+2]  * b_shared[k+2][threadIdx.x];
		result += a_shared[threadIdx.y][k+3]  * b_shared[k+3][threadIdx.x];
		result += a_shared[threadIdx.y][k+4]  * b_shared[k+4][threadIdx.x];
		result += a_shared[threadIdx.y][k+5]  * b_shared[k+5][threadIdx.x];
		result += a_shared[threadIdx.y][k+6]  * b_shared[k+6][threadIdx.x];
		result += a_shared[threadIdx.y][k+7]  * b_shared[k+7][threadIdx.x];
		result += a_shared[threadIdx.y][k+8]  * b_shared[k+8][threadIdx.x];
		result += a_shared[threadIdx.y][k+9]  * b_shared[k+9][threadIdx.x];
		result += a_shared[threadIdx.y][k+10] * b_shared[k+10][threadIdx.x];
		result += a_shared[threadIdx.y][k+11] * b_shared[k+11][threadIdx.x];
		result += a_shared[threadIdx.y][k+12] * b_shared[k+12][threadIdx.x];
		result += a_shared[threadIdx.y][k+13] * b_shared[k+13][threadIdx.x];
		result += a_shared[threadIdx.y][k+14] * b_shared[k+14][threadIdx.x];
		result += a_shared[threadIdx.y][k+15] * b_shared[k+15][threadIdx.x];
		result += a_shared[threadIdx.y][k+16] * b_shared[k+16][threadIdx.x];
		result += a_shared[threadIdx.y][k+17] * b_shared[k+17][threadIdx.x];
		result += a_shared[threadIdx.y][k+18] * b_shared[k+18][threadIdx.x];
		result += a_shared[threadIdx.y][k+19] * b_shared[k+19][threadIdx.x];
		result += a_shared[threadIdx.y][k+20] * b_shared[k+20][threadIdx.x];
		result += a_shared[threadIdx.y][k+21] * b_shared[k+21][threadIdx.x];
		result += a_shared[threadIdx.y][k+22] * b_shared[k+22][threadIdx.x];
		result += a_shared[threadIdx.y][k+23] * b_shared[k+23][threadIdx.x];
		result += a_shared[threadIdx.y][k+24] * b_shared[k+24][threadIdx.x];
		result += a_shared[threadIdx.y][k+25] * b_shared[k+25][threadIdx.x];
		result += a_shared[threadIdx.y][k+26] * b_shared[k+26][threadIdx.x];
		result += a_shared[threadIdx.y][k+27] * b_shared[k+27][threadIdx.x];
		result += a_shared[threadIdx.y][k+28] * b_shared[k+28][threadIdx.x];
		result += a_shared[threadIdx.y][k+29] * b_shared[k+29][threadIdx.x];
		result += a_shared[threadIdx.y][k+30] * b_shared[k+30][threadIdx.x];
		result += a_shared[threadIdx.y][k+31] * b_shared[k+31][threadIdx.x];
		__syncthreads();
		// wait for all threads in the block before loading
	}
	// handle leftover rows and columns for matrix sizes that aren't evenly divisible
	int leftover = N-tile_length*blockDim.x;
	if (leftover > 0) {
		// calculate the indexes into A and B to load into shared memory
		A_row_glob = blockIdx.y*blockDim.y + threadIdx.y; 
		A_col_glob = tile_iter*blockDim.x + threadIdx.x;

		B_row_glob = tile_iter*blockDim.y + threadIdx.y;
		B_col_glob = blockIdx.x*blockDim.x + threadIdx.x;

		// if the indexes are within bounds, load them
		// only loads the leftover entries into the tile
		if (A_col_glob < N) {
			a_shared[threadIdx.y][threadIdx.x] = A[A_row_glob*N + A_col_glob];
		}
		if (B_row_glob < N) {
			b_shared[threadIdx.y][threadIdx.x] = B[B_row_glob*N + B_col_glob];
		}
		__syncthreads();
		// edge case calculations are not unrolled because they don't have a big impact on performance
		for (k = 0; k < leftover; k++) {
			result += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
		}
		__syncthreads();
	}
	// put result into global memory if it is a valid index
	if (r < N && c < N) {
		C[r*N + c] = result;
	}
}

void computeGpuMMM() {
	// Allocate GPU memory for the inputs and the result
	clock_t start = clock();	
	copyMatricesToGPU();
	clock_t end = clock();
    double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
    printf("\nGPU: \tTransfer to GPU: %f seconds\n", elapsed);
	
	// block size and grid size
	dim3 blocks2d(blocks2d_sz,blocks2d_sz);
	dim3 threads2d(threads2d_sz,threads2d_sz);

	// Execute the kernel to compute the vector sum on the GPU
	start = clock();
	mm_multiply_kernel <<<blocks2d, threads2d>>> (A_GPU, B_GPU, C_GPU, C_MD.dimension1);
	// make the CPU main thread waite for the GPU kernel call to complete
	cudaThreadSynchronize();  // This is only needed for timing and error-checking purposes
	end = clock();
    elapsed = (end - start) / (double) CLOCKS_PER_SEC;
    printf("\tKernel execution: %f seconds\n", elapsed);
	
	// Check for kernel errors
	check_error(cudaGetLastError());

	// Transfer the result from the GPU to the CPU
	start = clock();
	copyResultFromGPU();
	end = clock();
	elapsed = (end - start) / (double) CLOCKS_PER_SEC;
	printf("\tTransfer from GPU: %f seconds\n", elapsed);
	
	// Free the GPU memory
	check_error(cudaFree(A_GPU));
	check_error(cudaFree(B_GPU));
	check_error(cudaFree(C_GPU));	
}

// allocate and initialize A and B using a random number generator
void allocateAndInitializeAB() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	A = (float*) malloc(sizeofA);
	
	srand(time(NULL));
  	for (int i = 0; i < A_MD.dimension1; i++) {
		for (int j = 0; j < A_MD.dimension2; j++) {
			int index = i * A_MD.dimension2 + j;
			A[index] = (rand() % 1000) * 0.001; 
		}
	}
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	B = (float*) malloc(sizeofB);
  	for (int i = 0; i < B_MD.dimension1; i++) {
		for (int j = 0; j < B_MD.dimension2; j++) {
			int index = i * B_MD.dimension2 + j;
			B[index] = (rand() % 1000) * 0.001; 
		}
	}
}

// allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {
	
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A, sizeofA, cudaMemcpyHostToDevice));
	
	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B, sizeofB, cudaMemcpyHostToDevice));
	
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
}

// copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C_CPU = (float*) malloc(sizeofC);
	check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
}

// do a straightforward matrix-matrix multiplication in the CPU
// notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are
// not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {
	
	// allocate the result matrix for the CPU computation
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C = (float*) malloc(sizeofC);
	
	// compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
	for (int i = 0; i < A_MD.dimension1; i++) {
		int a_i = i * A_MD.dimension2;
		int c_i = i * C_MD.dimension2;
		for (int j = 0; j < B_MD.dimension2; j++) {
			int c_index = c_i + j;
			C[c_index] = 0;
			for (int k = 0; k < B_MD.dimension1; k++) {
				int a_index = a_i + k;
				int b_index = k * B_MD.dimension2 + j;
				C[c_index] += A[a_index] * B[b_index];
			}
		}
	}
}

// function to determine if the GPU computation is done correctly by comparing the output from the GPU with that
// from the CPU
void compareHostAndGpuOutput() {
	int totalElements = C_MD.dimension1 * C_MD.dimension2;
	int missmatchCount = 0;
	for (int i = 0; i < totalElements; i++) {
		if (fabs(C[i] - C_CPU[i]) > 0.01) {
			missmatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
		}
	}
	if (missmatchCount > 0) {
		printf("Computation is incorrect: outputs do not match in %d indexes\n", missmatchCount);
	} else {
		printf("Computation is correct: CPU and GPU outputs match\n");
	}
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

