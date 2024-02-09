#include "utils.h"
#include "board.h"

#include <cassert>


void DrawGrid(const char* label, uint32 sz, const std::vector<uint32>& data)
{
	printf("------------------ %s ------------------\n", label);
	for (uint32 y = 0; y != sz; y++)
	{
		for (uint32 x = 0; x < sz; x++)
		{
			printf("%d ", data[y * sz + x]);
		}
		printf("\n");
	}
}

void DrawGrid(const char* label, uint32 sz, const std::vector<uint8>& data)
{
	printf("------------------ %s ------------------\n", label);
	for (uint32 y = 0; y != sz; y++)
	{
		for (uint32 x = 0; x < sz; x++)
		{
			printf("%d ", data[y * sz + x]);
		}
		printf("\n");
	}
}


void DrawGrid(const char* label, uint32 sz, const std::vector<int8>& data)
{
	printf("------------------ %s ------------------\n", label);
	for (uint32 y = 0; y != sz; y++)
	{
		for (uint32 x = 0; x < sz; x++)
		{
			printf("%+d ", data[y * sz + x]);
		}
		printf("\n");
	}
}

NeighborsBoard::NeighborsBoard(const uint32 boardSize) : Board(boardSize)
{
	dirtyCells.resize(boardSize * boardSize);
	cells2.resize(boardSize * boardSize);
	memcpy(cells2.data(), cells.data(), boardSize * boardSize * sizeof(uint8));

	neighborCounts.resize(boardSize * boardSize);
	CountNeighbors();
}

void NeighborsBoard::CountNeighbors()
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

					int32 ni = i + di;
					int32 nj = j + dj;

					if (ni >= 0 && ni < boardSize && nj >= 0 && nj < boardSize)
					{
						count += cells[ni * boardSize + nj];
					}
				}
			}
			neighborCounts[i * boardSize + j] = count;
		}
	}
}

void NeighborsBoard::Update()
{
	memset(dirtyCells.data(), 0, boardSize * boardSize * sizeof(int8));

	for (uint32 i = 0; i < boardSize; ++i)
	{
		for (uint32 j = 0; j < boardSize; ++j)
		{
			uint32 idx = i * boardSize + j;
			if (cells[idx] == 1)
			{
				if (neighborCounts[idx] < 2 || neighborCounts[idx] > 3)
				{
					cells2[idx] = 0;
					dirtyCells[idx] = -1;
				}
			}
			else
			{
				if (neighborCounts[idx] == 3)
				{
					cells2[idx] = 1;
					dirtyCells[idx] = 1;
				}
			}
		}
	}
	UpdateNeighbors();
}

void NeighborsBoard::Draw(std::vector<GLfloat>& colors)
{
	for (unsigned int n = 0; n < boardSize * boardSize; ++n)
	{
		const int8 d = dirtyCells[n];
		if (d == 0)
			continue;

		GLfloat value = d == 1 ? 1.0f : 0.0f;
		const unsigned int pos = n * 4;
		colors[pos] = value;
		colors[pos + 1] = value;
		colors[pos + 2] = value;
		colors[pos + 3] = value;
	}
	memcpy(cells.data(), cells2.data(), boardSize * boardSize * sizeof(uint8));
}

void NeighborsBoard::UpdateNeighbors()
{
	for (uint32 i = 0; i < boardSize; ++i)
	{
		for (uint32 j = 0; j < boardSize; ++j)
		{
			uint32 idx = i * boardSize + j;
			const int8 diff = dirtyCells[idx];
			if (diff == 0)
				continue;

			for (int32 di = -1; di <= 1; ++di)
			{
				for (int32 dj = -1; dj <= 1; ++dj)
				{
					if (di == 0 && dj == 0)
						continue;

					const int32 y = i + di;
					const int32 x = j + dj;
					if (y >= 0 && y < boardSize && x >= 0 && x < boardSize)
						neighborCounts[y*boardSize + x] += diff;
				}
			}
		}
	}
}