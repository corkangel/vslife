#include "utils.h"
#include "board.h"

#include "kernel.cuh"
#include <omp.h>
#include <cassert>

#include <cuda_gl_interop.h>

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
	cudaError e;
	
	e = cudaGraphicsMapResources(1, &cudaColorResource, 0);
	assert(e == cudaSuccess);

	void* colorsDevicePtr;
	size_t sz = 0;
	cudaGraphicsResourceGetMappedPointer(&colorsDevicePtr, &sz, cudaColorResource);

	interop_update(&dirtyCells[0], (float*)colorsDevicePtr);

	e = cudaGraphicsUnmapResources(1, &cudaColorResource, 0);
	assert(e == cudaSuccess);
}

void CudaNeighborsGlInteropBoard::Draw(std::vector<GLfloat>& colors)
{
	//#pragma omp parallel for
	//for (unsigned int n = 0; n < boardSize * boardSize; ++n)
	//{
	//	if (dirtyCells[n] == 0)
	//		continue;

	//	const unsigned int pos = n * 4;
	//	GLfloat value = (dirtyCells[n] == 1) ? 1.0f : 0.0f;
	//	colors[pos] = value;
	//	colors[pos + 1] = value;
	//	colors[pos + 2] = value;
	//	colors[pos + 3] = value;
	//}
}
void CudaNeighborsGlInteropBoard::Reupload()
{
	cudaError e;

	e = cudaGraphicsMapResources(1, &cudaColorResource, 0);
	assert(e == cudaSuccess);

	void* colorsDevicePtr;
	size_t sz = 0;
	cudaGraphicsResourceGetMappedPointer(&colorsDevicePtr, &sz, cudaColorResource);

	interop_reupload(&cells[0], (float*)colorsDevicePtr);

	e = cudaGraphicsUnmapResources(1, &cudaColorResource, 0);
	assert(e == cudaSuccess);
}