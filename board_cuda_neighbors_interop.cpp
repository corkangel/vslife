#include "utils.h"
#include "board.h"

#include "kernel.cuh"
#include <omp.h>
#include <cassert>

#include <cuda_gl_interop.h>


static void interop_map(cudaGraphicsResource* cudaColorResource, void** colorsDevicePtr)
{
	cudaError e;

	e = cudaGraphicsMapResources(1, &cudaColorResource, 0);
	assert(e == cudaSuccess);

	size_t sz = 0;
	e = cudaGraphicsResourceGetMappedPointer(colorsDevicePtr, &sz, cudaColorResource);
	assert(e == cudaSuccess);
}

static void interop_unmap(cudaGraphicsResource* cudaColorResource)
{
    cudaError e;

    e = cudaGraphicsUnmapResources(1, &cudaColorResource, 0);
    assert(e == cudaSuccess);
}

CudaNeighborsGlInteropBoard::CudaNeighborsGlInteropBoard(const uint32 boardSize, const uint32 VBOcolors) : Board(boardSize), VBOcolors(VBOcolors)
{
	dirtyCells.resize(boardSize * boardSize);

	interop_init(&cells[0], boardSize);

	cudaError e = cudaGraphicsGLRegisterBuffer(&cudaColorResource, VBOcolors, cudaGraphicsRegisterFlagsWriteDiscard);
	assert(e == cudaSuccess);
}

CudaNeighborsGlInteropBoard::~CudaNeighborsGlInteropBoard()
{
	cudaGraphicsUnregisterResource(cudaColorResource);
	interop_destroy();
}

void CudaNeighborsGlInteropBoard::Update()
{
	void* colorsDevicePtr = nullptr;
	interop_map(cudaColorResource, &colorsDevicePtr);
	interop_update((float*)colorsDevicePtr);
	interop_unmap(cudaColorResource);
}

void CudaNeighborsGlInteropBoard::Draw(std::vector<GLfloat>& colors)
{
	// empty - GL interop updates the color buffer directly
}

void CudaNeighborsGlInteropBoard::Reupload()
{
	void* colorsDevicePtr = nullptr;
	interop_map(cudaColorResource, &colorsDevicePtr);
	interop_reupload(&cells[0], (float*)colorsDevicePtr);
	interop_unmap(cudaColorResource);
}