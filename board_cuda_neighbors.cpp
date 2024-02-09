#include "utils.h"
#include "board.h"

#include "kernel.cuh"
#include <omp.h>

CudaNeighborsBoard::CudaNeighborsBoard(const uint32 boardSize) : Board(boardSize)
{
	dirtyCells.resize(boardSize * boardSize);

	neighbors_init(&cells[0], boardSize);
}

CudaNeighborsBoard::~CudaNeighborsBoard()
{
	neighbors_destroy();
}

void CudaNeighborsBoard::Update()
{
	neighbors_update(&dirtyCells[0]);
}

void CudaNeighborsBoard::Draw(std::vector<GLfloat>& colors)
{
	#pragma omp parallel for
	for (unsigned int n = 0; n < boardSize * boardSize; ++n)
	{
		if (dirtyCells[n] == 0)
			continue;

		const unsigned int pos = n * 4;
		GLfloat value = (dirtyCells[n] == 1) ? 1.0f : 0.0f;
		colors[pos] = value;
		colors[pos + 1] = value;
		colors[pos + 2] = value;
		colors[pos + 3] = value;
	}
}