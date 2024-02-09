
#include "cuda_runtime.h"
#include "./kernel.cuh"

#include <stdio.h>
#include <cassert>

unsigned char* n_cells;
unsigned char* n_cells2;
unsigned char* n_neighbors;
char* n_dirty;
unsigned int n_boardSize;

__global__ void count_intitial_neighbors(unsigned char* cells, unsigned char* neighbors, unsigned int boardSize)
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


__global__ void update_neighbors_kernel(const char* dirty, unsigned char* neighbors, const unsigned int boardSize, const unsigned int ox, const unsigned int oy)
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


__global__ void update_all_neighbors_kernel(const char* dirty, unsigned char* neighbors, const unsigned int boardSize)
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

__global__ void neighbors_update_cell(unsigned char* cells, unsigned char* cells2, char* dirty, const unsigned char* neighbors, unsigned int boardSize)
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
        }
    }
    else
    {
        if (numNeighbors == 3)
        {
            cells2[index] = 1;
            dirty[index] = 1;
        }
    }
}

void neighbors_init(unsigned char* initialCells, const unsigned int boardSize)
{
    n_boardSize = boardSize;
    cudaMalloc((void**)&n_cells, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&n_cells2, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&n_neighbors, boardSize * boardSize * sizeof(unsigned char));
    cudaMalloc((void**)&n_dirty, boardSize * boardSize * sizeof(char));

    cudaMemcpy(n_cells, initialCells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(n_cells2, initialCells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaMemset(n_neighbors, 0, boardSize * boardSize * sizeof(unsigned char));

    count_intitial_neighbors<<<1, 1>>>(n_cells, n_neighbors, boardSize);
}

void neighbors_update(char* dirty)
{
    cudaMemset(n_dirty, 0, n_boardSize * n_boardSize * sizeof(char));

    dim3 blockSize(16, 16);
    dim3 gridSize(n_boardSize / 16 + 1, n_boardSize / 16 + 1);

    neighbors_update_cell<<<gridSize, blockSize>>>(n_cells, n_cells2, n_dirty, n_neighbors, n_boardSize);

    cudaMemcpy(n_cells, n_cells2, n_boardSize * n_boardSize * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dirty, n_dirty, n_boardSize * n_boardSize * sizeof(char), cudaMemcpyDeviceToHost);

    dim3 nBlockSize(32, 32);
    dim3 nGridSize(n_boardSize / 32 / 3+1, n_boardSize / 32 / 3+1);

    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 0, 0);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 0, 1);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 1, 0);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 1, 1);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 2, 0);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 0, 2);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 2, 1);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 1, 2);
    update_neighbors_kernel<<<nBlockSize, nGridSize>>>(n_dirty, n_neighbors, n_boardSize, 2, 2);

}

void neighbors_destroy()
{
    cudaFree(n_cells);
    cudaFree(n_cells2);
    cudaFree(n_neighbors);
    cudaFree(n_dirty);
}
