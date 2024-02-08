
#include "cuda_runtime.h"
#include "./kernel.cuh"


unsigned char* d_cells;
unsigned char* d_cells2;
char* d_dirty;
unsigned int g_boardSize;

__global__ void board_update_cell(unsigned char* cells, unsigned char* cells2, char* dirty, unsigned int boardSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= boardSize || y >= boardSize)
		return;

	int index = y * boardSize + x;
	int numNeighbors = 0;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int x2 = x + i;
			int y2 = y + j;
			if (x2 >= 0 && x2 < boardSize && y2 >= 0 && y2 < boardSize)
			{
				if (i != 0 || j != 0)
				{
					if (cells[y2 * boardSize + x2] == 1)
					{
						numNeighbors++;
					}
				}
			}
		}
	}
	if (cells[index] == 1)
	{
		if (numNeighbors < 2 || numNeighbors > 3)
		{
			cells2[index] = 0;
			dirty[index] = -1;
		}
		else
		{
			cells2[index] = 1;
		}
	}
	else
	{
		if (numNeighbors == 3)
		{
			cells2[index] = 1;
			dirty[index] = 1;
		}
		else
		{
			cells2[index] = 0;
		}
	}
}

void board_init(unsigned char* initialCells, const unsigned int boardSize)
{
	g_boardSize = boardSize;
	cudaMalloc((void**)&d_cells, boardSize * boardSize * sizeof(unsigned char));
	cudaMalloc((void**)&d_cells2, boardSize * boardSize * sizeof(unsigned char));
	cudaMalloc((void**)&d_dirty, boardSize * boardSize * sizeof(char));

	cudaMemcpy(d_cells, initialCells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cells2, initialCells, boardSize * boardSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
}

void board_update(char* dirty)
{
	cudaMemset(d_dirty, 0, g_boardSize * g_boardSize * sizeof(char));

	dim3 blockSize(16, 16);
	dim3 gridSize(g_boardSize / 16 + 1, g_boardSize / 16 + 1);

	board_update_cell<<<gridSize, blockSize>>>(d_cells, d_cells2, d_dirty, g_boardSize);

	cudaMemcpy(d_cells, d_cells2, g_boardSize * g_boardSize * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dirty, d_dirty, g_boardSize * g_boardSize * sizeof(char), cudaMemcpyDeviceToHost);
}

void board_destroy()
{
	cudaFree(d_cells);
	cudaFree(d_cells2);
	cudaFree(d_dirty);
}
