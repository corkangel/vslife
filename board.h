#pragma once

#include <vector>
#include <random>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "kernel.cuh"

class Board
{
public:

    std::vector<unsigned char> cells;
    std::vector<unsigned char> tmp;
    std::vector<unsigned char> neighborCounts;
    std::vector<unsigned int> dirtyPlus;
    std::vector<unsigned int> dirtyMinus;
    const unsigned int gridSize;

    Board(unsigned int size) : gridSize(size)
    {
        cells.resize(gridSize * gridSize);
        tmp.resize(gridSize * gridSize);
        neighborCounts.resize(gridSize * gridSize, 0);
        dirtyPlus.reserve(gridSize * gridSize);
        dirtyMinus.reserve(gridSize * gridSize);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 1);

        for (unsigned int n = 0; n < gridSize * gridSize; ++n) {
            cells[n] = dis(gen);
        }
        memcpy(&tmp[0], &cells[0], gridSize * gridSize * sizeof(unsigned char));

        CountNeighbors();
    }

    void CountNeighbors()
    {
        for (unsigned int i = 0; i < gridSize; ++i)
        {
            for (unsigned int j = 0; j < gridSize; ++j)
            {
                for (int di = -1; di <= 1; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        if (di == 0 && dj == 0)
                            continue;

                        unsigned int ni = i + di;
                        unsigned int nj = j + dj;

                        if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize && cells[ni * gridSize + nj] == 1)
                        {
                            neighborCounts[i * gridSize + j]++;
                        }
                    }
                }
            }
        }
    }

    void UpdateNeighbor(const int i, const char diff)
    {

        if (i >= 0 && i < gridSize * gridSize)
        {
            neighborCounts[i] += diff;
        }
    }

    void UpdateNeighbors(const std::vector<unsigned int>& dirty, const char diff)
    {
        for (unsigned int n = 0; n < dirty.size(); ++n)
        {
            const int i = dirty[n];
            UpdateNeighbor(i - 1, diff);
            UpdateNeighbor(i + 1, diff);
            UpdateNeighbor(i - gridSize, diff);
            UpdateNeighbor(i + gridSize, diff);
            UpdateNeighbor(i + gridSize - 1, diff);
            UpdateNeighbor(i + gridSize + 1, diff);
            UpdateNeighbor(i - gridSize + 1, diff);
            UpdateNeighbor(i - gridSize - 1, diff);
        }
    }


    void UpdateOldCuda()
    {
        update_board(&cells[0], &tmp[0], gridSize);
    }

    void UpdateOld()
    {
        for (unsigned int n = 0; n < gridSize * gridSize; ++n)
        {
            unsigned char count = 0;
			unsigned int i = n / gridSize;
			unsigned int j = n % gridSize;
            for (int di = -1; di <= 1; ++di)
            {
                for (int dj = -1; dj <= 1; ++dj)
                {
					if (di == 0 && dj == 0)
						continue;

					unsigned int ni = i + di;
					unsigned int nj = j + dj;

                    if (ni >= 0 && ni < gridSize && nj >= 0 && nj < gridSize && cells[ni * gridSize + nj] == 1)
                    {
						count++;
					}
				}
			}

            if (cells[n] == 1)
            {
                if (count < 2 || count > 3)
                {
					tmp[n] = 0;
				}
			}
            else
            {
                if (count == 3)
                {
					tmp[n] = 1;
				}
			}
        }
    }


    void DrawOld(std::vector<GLfloat>& colors)
    {
        for (unsigned int n = 0; n < gridSize * gridSize; ++n)
        {
            if (cells[n] == tmp[n])
                continue;

            if (tmp[n] == 1)
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
        memcpy(&cells[0], &tmp[0], gridSize * gridSize * sizeof(unsigned char));
	}

    void Update()
    {
        dirtyPlus.clear();
        dirtyMinus.clear();
        for (unsigned int n = 0; n < gridSize * gridSize; ++n)
        {
            unsigned char count = neighborCounts[n];

            if (cells[n] == 1)
            {
                if (count < 2 || count > 3)
                {
                    tmp[n] = 0;
                    dirtyMinus.push_back(n);
                }
            }
            else
            {
                if (count == 3)
                {
                    tmp[n] = 1;
                    dirtyPlus.push_back(n);
                }
            }
        }
        memcpy(&cells[0], &tmp[0], gridSize * gridSize * sizeof(unsigned char));

        UpdateNeighbors(dirtyPlus, 1);
        UpdateNeighbors(dirtyMinus, -1);
    }

    void Draw(std::vector<GLfloat>& colors)
    {
        for (unsigned int index : dirtyPlus)
        {
            const unsigned int pos = index * 4;
            colors[pos] = 1.0f;
            colors[pos + 1] = 1.0f;
            colors[pos + 2] = 1.0f;
            colors[pos + 3] = 1.0f;
        }
        for (unsigned int index : dirtyMinus)
        {
            const unsigned int pos = index * 4;
            colors[pos] = 0.0f;
            colors[pos + 1] = 0.0f;
            colors[pos + 2] = 0.0f;
            colors[pos + 3] = 0.0f;
        }
    }
};

