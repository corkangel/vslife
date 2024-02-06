#include "cuda_runtime.h"
#include "./kernel.cuh"

__global__ void vectorAdditionKernel(double* A, double* B, double* C, int arraySize) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if thread is within array bounds.
    if (threadID < arraySize) {
        // Add a and b.
        C[threadID] = A[threadID] + B[threadID];
    }
}



/**
 * Wrapper function for the CUDA kernel function.
 * @param A Array A.
 * @param B Array B.
 * @param C Sum of array elements A and B directly across.
 * @param arraySize Size of arrays A, B, and C.
 */
void kernel(double* A, double* B, double* C, int arraySize) {

    // Initialize device pointers.
    double* d_A, * d_B, * d_C;

    // Allocate device memory.
    cudaMalloc((void**)&d_A, arraySize * sizeof(double));
    cudaMalloc((void**)&d_B, arraySize * sizeof(double));
    cudaMalloc((void**)&d_C, arraySize * sizeof(double));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_A, A, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / arraySize + 1, 1);

    // Launch CUDA kernel.
    vectorAdditionKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, arraySize);

    // Copy result array c back to host memory.
    cudaMemcpy(C, d_C, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
}


__global__ void update_board_kernel(const unsigned char* cells, unsigned char* new_cells, const unsigned int boardSize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < boardSize && j < boardSize)
    {
		int count = 0;
        for (int di = -1; di <= 1; ++di)
        {
            for (int dj = -1; dj <= 1; ++dj)
            {
				if (di == 0 && dj == 0)
					continue;

				int ni = i + di;
				int nj = j + dj;

                if (ni >= 0 && ni < boardSize && nj >= 0 && nj < boardSize && cells[ni * boardSize + nj] == 1)
                {
					count++;
				}
			}
		}

        if (cells[i * boardSize + j] == 1)
        {
            if (count < 2 || count > 3)
            {
				new_cells[i * boardSize + j] = 0;
			}
            else
            {
				new_cells[i * boardSize + j] = 1;
			}
		}
        else
        {
            if (count == 3)
            {
				new_cells[i * boardSize + j] = 1;
			}
            else
            {
				new_cells[i * boardSize + j] = 0;
			}
		}
	}
}

void update_board(const unsigned char* cells, unsigned char* new_cells, const unsigned int boardSize)
{
    unsigned char* d_cells, * d_new_cells;
    cudaMalloc((void**)&d_cells, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_new_cells, boardSize * boardSize * sizeof(unsigned char));

    cudaMemcpy(d_cells, cells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_cells, new_cells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(boardSize / 16 + 1, boardSize / 16 + 1);

    update_board_kernel << <gridSize, blockSize >> > (d_cells, d_new_cells, boardSize);

    cudaMemcpy(new_cells, d_new_cells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_cells);
    cudaFree(d_new_cells);
}