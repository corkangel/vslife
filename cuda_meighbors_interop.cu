
#include "cuda_runtime.h"
#include "./kernel.cuh"

#include <stdio.h>
#include <cassert>

unsigned char* interop_cells;
unsigned char* interop_cells2;
unsigned char* interop_uploadCells;
unsigned char* interop_neighbors;
char* interop_dirty;
unsigned int interop_boardSize;

__global__ void interop_count_intitial_neighbors(unsigned char* cells, unsigned char* neighbors, unsigned int boardSize)
{
    for (unsigned int i = 0; i < boardSize; i++)
    {
        for (unsigned int j = 0; j < boardSize; j++)
        {
            int index = i * boardSize + j;
            int numNeighbors = 0;
            for (int i2 = -1; i2 <= 1; i2++)
            {
                for (int j2 = -1; j2 <= 1; j2++)
                {
                    int x = j + j2;
                    int y = i + i2;
                    if (x >= 0 && x < boardSize && y >= 0 && y < boardSize)
                    {
                        if (i2 != 0 || j2 != 0)
                        {
                            if (cells[y * boardSize + x] == 1)
                            {
                                numNeighbors++;
                            }
                        }
                    }
                }
            }
            neighbors[index] = numNeighbors;
        }
    }
}


__global__ void update_interop_kernel(const char* dirty, unsigned char* neighbors, const unsigned int boardSize, const unsigned int ox, const unsigned int oy)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int x = i * 3 + ox;
    const unsigned int y = j * 3 + oy;

    if (x >= boardSize || y >= boardSize)
        return;

    const unsigned int index = y * boardSize + x;
    const char d = dirty[index];
    if (d == 0)
        return;

    for (int i2 = -1; i2 <= 1; i2++)
    {
        for (int j2 = -1; j2 <= 1; j2++)
        {
            int x2 = x + j2;
            int y2 = y + i2;
            if (x2 >= 0 && x2 < boardSize && y2 >= 0 && y2 < boardSize)
            {
                if (x2 == x && y2 == y)
                {
                    continue;
                }
                neighbors[y2 * boardSize + x2] += d;
            }
        }
    }
}


__global__ void update_all_interop_kernel(const char* dirty, unsigned char* neighbors, const unsigned int boardSize)
{
    for (int y = 0; y < boardSize; y++)
    {
        for (int x = 0; x < boardSize; x++)
        {
            const unsigned int index = y * boardSize + x;
            const char d = dirty[index];
            if (d == 0)
                continue;

            for (int i2 = -1; i2 <= 1; i2++)
            {
                for (int j2 = -1; j2 <= 1; j2++)
                {
                    int x2 = x + j2;
                    int y2 = y + i2;
                    if (x2 >= 0 && x2 < boardSize && y2 >= 0 && y2 < boardSize)
                    {
                        if (x2 == x && y2 == y)
                        {
                            continue;
                        }
                        neighbors[y2 * boardSize + x2] += d;
                    }
                }
            }
        }
    }
}

__global__ void interop_update_cell(unsigned char* cells, unsigned char* cells2, char* dirty, const unsigned char* neighbors, unsigned int boardSize, float* colorsPtr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= boardSize || y >= boardSize)
        return;

    int index = y * boardSize + x;
    int numNeighbors = neighbors[index];
    if (cells[index] == 1)
    {
        if (numNeighbors < 2 || numNeighbors > 3)
        {
            cells2[index] = 0;
            dirty[index] = -1;

            colorsPtr[index * 4] = 0.0f;
            colorsPtr[index * 4 + 1] = 0.0f;
            colorsPtr[index * 4 + 2] = 0.0f;
            colorsPtr[index * 4 + 3] = 0.0f;
        }
    }
    else
    {
        if (numNeighbors == 3)
        {
            cells2[index] = 1;

            colorsPtr[index * 4] = 1.0f;
            colorsPtr[index * 4 + 1] = 1.0f;
            colorsPtr[index * 4 + 2] = 1.0f;
            colorsPtr[index * 4 + 3] = 1.0f;
                
            dirty[index] = 1;
        }
    }
}


__global__ void interop_merge_cells(unsigned char* cells, unsigned char* cells2, unsigned char* uploadCells, unsigned int boardSize, float* colorsPtr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= boardSize || y >= boardSize)
        return;

    int index = y * boardSize + x;
    if (uploadCells[index] == 1)
    {
        cells[index] = 1;
        cells2[index] = 1;

        colorsPtr[index * 4] = 1.0f;
        colorsPtr[index * 4 + 1] = 1.0f;
        colorsPtr[index * 4 + 2] = 1.0f;
        colorsPtr[index * 4 + 3] = 1.0f;
    }
}

void interop_init(unsigned char* initialCells, const unsigned int boardSize)
{
    interop_boardSize = boardSize;
    cudaMalloc((void**)&interop_cells, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&interop_cells2, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&interop_uploadCells, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&interop_neighbors, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&interop_dirty, boardSize * boardSize * sizeof(char));

    cudaMemcpy(interop_cells, initialCells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(interop_cells2, initialCells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMemset(interop_neighbors, 0, boardSize * boardSize * sizeof(unsigned char));
    interop_count_intitial_neighbors<<<1, 1>>>(interop_cells, interop_neighbors, boardSize);
}

void interop_reupload(unsigned char* uploadCells, float* colorsPtr)
{
    cudaMemcpy(interop_uploadCells, uploadCells, interop_boardSize * interop_boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(interop_boardSize / 16 + 1, interop_boardSize / 16 + 1);
    interop_merge_cells<<<gridSize, blockSize>>>(interop_cells, interop_cells2, interop_uploadCells, interop_boardSize, colorsPtr);

    cudaMemset(interop_neighbors, 0, interop_boardSize * interop_boardSize * sizeof(unsigned char));
    interop_count_intitial_neighbors << <1, 1 >> > (interop_cells, interop_neighbors, interop_boardSize);
}

void interop_update(float* colorsDevicePtr)
{
    cudaMemset(interop_dirty, 0, interop_boardSize * interop_boardSize * sizeof(char));

    const int N = 32;

    dim3 blockSize(N, N);
    dim3 gridSize(interop_boardSize / N + 1, interop_boardSize / N + 1);

    interop_update_cell<<<gridSize, blockSize>>>(interop_cells, interop_cells2, interop_dirty, interop_neighbors, interop_boardSize, colorsDevicePtr);

    cudaMemcpy(interop_cells, interop_cells2, interop_boardSize * interop_boardSize * sizeof(unsigned char), cudaMemcpyDeviceToDevice);

    dim3 nBlockSize(N, N);
    dim3 nGridSize(interop_boardSize / N / 3+1, interop_boardSize / N / 3+1);

    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 0, 0);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 0, 1);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 1, 0);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 1, 1);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 2, 0);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 0, 2);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 2, 1);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 1, 2);
    update_interop_kernel<<<nBlockSize, nGridSize>>>(interop_dirty, interop_neighbors, interop_boardSize, 2, 2);

}

void interop_destroy()
{
    cudaFree(interop_cells);
    cudaFree(interop_cells2);
    cudaFree(interop_uploadCells);
    cudaFree(interop_neighbors);
    cudaFree(interop_dirty);
}
