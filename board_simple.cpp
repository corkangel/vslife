#include "board.h"

#include <random>

Board::Board(const uint32 boardSize) : boardSize(boardSize)
{
	cells.resize(boardSize * boardSize);
	FillRandom();
}

void Board::FillRandom()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 1);

	for (unsigned int n = 0; n < boardSize * boardSize; ++n) {
		cells[n] = dis(gen);
	}
}

SimpleBoard::SimpleBoard(const uint32 boardSize) : Board(boardSize)
{
	cells2.resize(boardSize * boardSize);
	memcpy(cells2.data(), cells.data(), boardSize * boardSize * sizeof(uint8));
}

void SimpleBoard::Update()
{
	for (uint32 i = 0; i < boardSize; ++i)
	{
		for (uint32 j = 0; j < boardSize; ++j)
		{
			uint32 count = 0;
			for (int32 di = -1; di <= 1; ++di)
			{
				for (int32 dj = -1; dj <= 1; ++dj)
				{
					if (di == 0 && dj == 0)
						continue;

					uint32 ni = i + di;
					uint32 nj = j + dj;

					if (ni < boardSize && nj < boardSize)
					{
						count += cells[ni * boardSize + nj];
					}
				}
			}

			uint32 idx = i * boardSize + j;
			if (cells[idx] == 1)
			{
				if (count < 2 || count > 3)
				{
					cells2[idx] = 0;
				}
			}
			else
			{
				if (count == 3)
				{
					cells2[idx] = 1;
				}
			}
		}
	}
}

void SimpleBoard::Draw(std::vector<GLfloat>& colors)
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