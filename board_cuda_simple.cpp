#include "utils.h"
#include "board.h"

#include "kernel.cuh"

#include <cassert>

CudaSimpleBoard::CudaSimpleBoard(const uint32 boardSize) : Board(boardSize)
{
	cells2.resize(boardSize * boardSize);
	memcpy(cells2.data(), cells.data(), boardSize * boardSize * sizeof(uint8));
}

void CudaSimpleBoard::Update()
{
	update_board(&cells[0], &cells2[0], boardSize);
}

void CudaSimpleBoard::Draw(std::vector<GLfloat>& colors)
{
	for (unsigned int n = 0; n < boardSize * boardSize; ++n)
	{
		if (cells[n] == cells2[n])
			continue;

		if (cells2[n] == 1)
		{
			const unsigned int pos = n * 4;
			colors[pos] = 1.0f;
			colors[pos + 1] = 1.0f;
			colors[pos + 2] = 1.0f;
			colors[pos + 3] = 1.0f;
		}
		else
		{
			const unsigned int pos = n * 4;
			colors[pos] = 0.0f;
			colors[pos + 1] = 0.0f;
			colors[pos + 2] = 0.0f;
			colors[pos + 3] = 0.0f;
		}
	}
	memcpy(cells.data(), cells2.data(), boardSize * boardSize * sizeof(uint8));
}