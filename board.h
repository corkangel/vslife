#pragma once

#include <vector>

#include <GL/glew.h>

#include "utils.h"

class Board
{
public:
    uint32 boardSize;
    std::vector<uint8> cells;

    Board(const uint32 boardSize);
    ~Board() = default;

    virtual void Update() = 0;
    virtual void Draw(std::vector<GLfloat>& colors) = 0;
    virtual void Reupload() { }
    void FillRandom();

};


// The simplest implementation of the game of life (entirely on CPU)
class SimpleBoard : public Board
{
    std::vector<uint8> cells2;

public:
    SimpleBoard(const uint32 boardSize);

    void Update() override;
    void Draw(std::vector<GLfloat>& colors) override;
};

// Optimized implementation using neighbor counts (entirely on CPU)
class NeighborsBoard : public Board
{
    std::vector<uint8> cells2;
    std::vector<uint8> neighborCounts;
    std::vector<int8> dirtyCells;

public:
    NeighborsBoard(const uint32 boardSize);

    void Update() override;
    void Draw(std::vector<GLfloat>& colors) override;

    void CountNeighbors();
    void UpdateNeighbors();
};

// Simple imeplementation using CUDA on GPU.
// Copies current board to the GPU, and new board back to the CPU each frame.
class CudaSimpleBoard : public Board
{
	std::vector<uint8> cells2;

  public:
      CudaSimpleBoard(const uint32 boardSize);

	  void Update() override;
	  void Draw(std::vector<GLfloat>& colors) override;
};

// Simple imeplementation using CUDA on GPU.
// Board lives on the GPU, and a dirty list is copied back to the CPU each frame.
class CudaOnGpuBoard : public Board
{
    std::vector<int8> dirtyCells;

public:
    CudaOnGpuBoard(const uint32 boardSize);
    ~CudaOnGpuBoard();

    void Update() override;
    void Draw(std::vector<GLfloat>& colors) override;
};

// Optimized implementation using neighbor counts, using CUDA on GPU.
// Board lives on the GPU, and a dirty list is copied back to the CPU each frame.

class CudaNeighborsBoard : public Board
{
    std::vector<int8> dirtyCells;

public:
    CudaNeighborsBoard(const uint32 boardSize);
    ~CudaNeighborsBoard();

	void Update() override;
	void Draw(std::vector<GLfloat>& colors) override;
};


// CudaNeighborsBoard but updating the color buffer on the GPU directly from cuda.
// No copying back to the CPU
struct cudaGraphicsResource;
class CudaNeighborsGlInteropBoard : public Board
{
    std::vector<int8> dirtyCells;
    const uint32 VBOcolors;
    struct cudaGraphicsResource* cudaColorResource;

public:
    CudaNeighborsGlInteropBoard(const uint32 boardSize, const uint32 VBOcolors);
    ~CudaNeighborsGlInteropBoard();

    void Update() override;
    void Draw(std::vector<GLfloat>& colors) override;
    void Reupload() override;
};
