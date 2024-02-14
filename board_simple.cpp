#include "board.h"

#include <random>

std::vector<std::vector<uint8>> rotate_shape(const std::vector<std::vector<uint8>>& shape)
{
	uint32 n = uint32(shape.size());
	uint32 m = uint32(shape[0].size());
	std::vector<std::vector<uint8>> rotated_shape(m, std::vector<uint8>(n));

	for (uint32 i = 0; i < n; ++i) {
		for (uint32 j = 0; j < m; ++j) {
			rotated_shape[j][i] = shape[i][j];
		}
	}

	for (auto& row : rotated_shape) {
		std::reverse(row.begin(), row.end());
	}

	return rotated_shape;
}

void add_shape(std::vector<uint8>& cells, uint32 x, uint32 y, const std::vector<std::vector<uint8>>& shape, const uint32 boardSize)
{
	uint32 n = uint32(shape.size());
	uint32 m = uint32(shape[0].size());

	for (uint32 i = 0; i < n; ++i) {
		for (uint32 j = 0; j < m; ++j) {
			cells[x + i + (y + j) * boardSize] = shape[i][j];
		}
	}
}


void add_rotated_shape(std::vector<uint8>& cells, uint32 x, uint32 y, const std::vector<std::vector<uint8>>& shape, uint32 rotation, const uint32 boardSize)
{
	std::vector<std::vector<uint8>> rotated_shape = shape;
	while (rotation > 0) {
		rotated_shape = rotate_shape(shape);
		rotation -= 90;
	}

	add_shape(cells, x, y, rotated_shape, boardSize);
}


void data_to_cells(std::vector<uint8>& cells, const std::vector<std::vector<int>>& data, int x, int y, uint32 boardSize)
{
	for (int i = 0; i < data.size(); ++i) {
		for (int j = 0; j < data[0].size(); ++j) {
			cells[x + i + (y + j) * boardSize] = data[i][j];
		}
	}
}

void add_spaceship(std::vector<uint8>& cells, int x, int y, uint32 boardSize)
{
	// random rotation
	uint32 r = (rand() % 4) * 90;
	const std::vector<std::vector<uint8>> data = {
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 0}
	};
	add_rotated_shape(cells, x, y, data, r, boardSize);
}

void add_blinker(std::vector<uint8>& cells, int x, int y, uint32 boardSize)
{
	const std::vector<std::vector<uint8>> data = {
		{0, 0, 0},
        {1, 1, 1},
        {0, 0, 0}
    };
	add_rotated_shape(cells, x, y, data, 0, boardSize);
}

void add_toad(std::vector<uint8>& cells, int x, int y, uint32 boardSize)
{
	// random rotation
	uint32 r = (rand() % 4) * 90;
	const std::vector<std::vector<uint8>> data = {
		{0, 1, 1, 1},
		{1, 1, 1, 0}
	};
	add_rotated_shape(cells, x, y, data, r, boardSize);
}

void add_beacon(std::vector<uint8>& cells, int x, int y, uint32 boardSize)
{
	const std::vector<std::vector<uint8>> data = {
		{1, 1, 0, 0},
		{1, 1, 0, 0},
		{0, 0, 1, 1},
		{0, 0, 1, 1}
	};
	add_rotated_shape(cells, x, y, data, 0, boardSize);
}

Board::Board(const uint32 boardSize) : boardSize(boardSize)
{
	cells.resize(boardSize * boardSize);
	RandomSpaceships(boardSize*4);
}

void Board::FillRandom()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 1);

	for (unsigned int n = 0; n < boardSize * boardSize / 8; ++n) {
		cells[n] = dis(gen);
	}
}

void Board::RandomSpaceships(const uint32 num)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, boardSize - 5);

	for (unsigned int n = 0; n < num; ++n)
	{
		int32 nn = n % 4;
		if (nn == 0) add_blinker(cells, dis(gen), dis(gen), boardSize);
		else if (nn == 1) add_toad(cells, dis(gen), dis(gen), boardSize);
		else if (nn == 2) add_beacon(cells, dis(gen), dis(gen), boardSize);
		else add_spaceship(cells, dis(gen), dis(gen), boardSize);
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