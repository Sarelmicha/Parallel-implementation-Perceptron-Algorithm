
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Main.h"

int calcnMissWithCuda(double* allCoords, int* allSets, double* weights, int dim, int limit, double alpha, int size);
cudaError_t updateLocationWithCuda(double* coordsOfMyPoints, double* velocityOfMyPoints, int size, int dim, double dt);
int checkAllPoints(double* allCoords, double* weights, double* dev_coords, int* dev_set, double* dev_weights, int* dev_nMissArr, int dim, double alpha, int size, int numOfBlocks, int numOfThreadsPerBlock, cudaError_t* cudaStatus);

void checkError(cudaError_t cudaStatus, void* dev_arr, const char* errorMessage);
void freeCudaMemory(int arg_count, ...);
__device__ void calculateResult(double* dev_coords, double* weights, int dim, double* localResult);
__device__ int sign(double value);


/*Function will calculate the nMiss of all points*/
__global__ void calcnMissWithKernel(double* dev_coords, int* dev_set, double* dev_weights, int* dev_nMissArr, int dim, int size)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	double localResult = 0;

	if (id < size) // filter all unneccesary threads.
	{
		calculateResult(dev_coords + id * (dim + 1), dev_weights, dim, &localResult);

		if ((dev_set[id] > 0 && localResult <= 0) || (dev_set[id] < 0 && localResult > 0))
		{
			//In the case we need to fix the weights : devnMissArr will contain -1 or 1 in the unmachted points and 0 for matched
			dev_nMissArr[id] = sign(localResult);
		}
	}
}

/*Function will update the coordinates of every point according to its velocity*/
__global__ void updateLocationOfAllPointsWithKernel(double* dev_coords, double* dev_velocity, int size, int dim, double currentTime)
{
	int j;
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id < size) // filter all unneccesary threads.
	{
		for (j = 1; j < dim + 1; j++)
		{
			dev_coords[j + id * (dim + 1)] = dev_coords[j + id * (dim + 1)] + dev_velocity[(j - 1) + id * (dim)] * currentTime;
		}
	}
}
/*Function will calculate the result of a single Point*/
__device__ void calculateResult(double* dev_coords, double* weights, int dim, double* localResult)
{
	int i;

	for (i = 0; i < dim + 1; i++)
	{
		*localResult += (weights[i]) * (dev_coords[i]);
	}
}

/*Function is gettin a value and return its sign (+ or -)*/
__device__ int sign(double value)
{
	if (value > 0)
		return 1;
	else
		return -1;
}

// Helper function for using CUDA to calculate nMissArr 
int calcnMissWithCuda(double* allCoords, int* allSets, double* weights, int dim,int limit,double alpha, int size)
{
	char errorBuffer[100];
	int* dev_nMissArr = 0;
	double* dev_weights = 0;
	double* dev_coords = 0;
	int* dev_set = 0;
	int totalnMiss;
	int i;

	cudaDeviceProp props;
	cudaError_t cudaStatus;
	cudaGetDeviceProperties(&props, 0);
	const char* ERROR_MESSAGE = "cudaMalloc failed!";
	const int maxNumOfThreadsPerBlock = props.maxThreadsPerBlock;
	const int numOfBlocks = size % maxNumOfThreadsPerBlock == 0 ? size / maxNumOfThreadsPerBlock : (size / maxNumOfThreadsPerBlock) + 1;
	const int numOfThreadsPerBlock = size % numOfBlocks == 0 ? size / numOfBlocks : (size / numOfBlocks) + 1;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	//Allocating space in GPU
	cudaStatus = cudaMalloc((void**)&dev_nMissArr, size * sizeof(int));
	checkError(cudaStatus, dev_nMissArr, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)&dev_coords, size * (dim + 1) * sizeof(double));
	checkError(cudaStatus, dev_coords, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)&dev_set, size * sizeof(int));
	checkError(cudaStatus, dev_set, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)&dev_weights, (dim + 1) * sizeof(double));
	checkError(cudaStatus, dev_weights, ERROR_MESSAGE);


	//Copy to GPU coords and sets
	cudaStatus = cudaMemcpy(dev_coords, allCoords, (size) * (dim + 1) * sizeof(double), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_coords, "cudaMemcpy failed!");
	cudaStatus = cudaMemcpy(dev_set, allSets, (size) * sizeof(int), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_set, "cudaMemcpy failed!");
	
	for (i = 0; i < limit; i++)
	{
		totalnMiss = checkAllPoints(allCoords, weights, dev_coords, dev_set, dev_weights,dev_nMissArr, dim, alpha, size, numOfBlocks, numOfThreadsPerBlock, &cudaStatus);
		if (totalnMiss == 0)
			break;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	freeCudaMemory(4, dev_coords, dev_set, dev_weights, dev_nMissArr);

	return totalnMiss;
}

// Helper function for using CUDA to uodate each point coords
cudaError_t updateLocationWithCuda(double* coordsOfMyPoints, double* velocityOfMyPoints, int size, int dim, double dt)
{
	char errorBuffer[100];
	double* dev_coords = 0;
	double* dev_velocity = 0;
	cudaDeviceProp props;
	cudaError_t cudaStatus;
	const char* ERROR_MESSAGE = "cudaMalloc failed!";
	cudaGetDeviceProperties(&props, 0);
	const int maxNumOfThreadsPerBlock = props.maxThreadsPerBlock;
	const int numOfBlocks = size % maxNumOfThreadsPerBlock == 0 ? size / maxNumOfThreadsPerBlock : (size / maxNumOfThreadsPerBlock) + 1;
	const int numOfThreadsPerBlock = size % numOfBlocks == 0 ? size / numOfBlocks : (size / numOfBlocks) + 1;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	//Malloc space in GPU
	cudaStatus = cudaMalloc((void**)&dev_coords, size * (dim + 1) * sizeof(double));
	checkError(cudaStatus, dev_coords, ERROR_MESSAGE);
	cudaStatus = cudaMalloc((void**)&dev_velocity, size * (dim) * sizeof(double));
	checkError(cudaStatus, dev_velocity, ERROR_MESSAGE);

	//Copy data to GPU
	cudaStatus = cudaMemcpy(dev_coords, coordsOfMyPoints, (size) * (dim + 1) * sizeof(double), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_coords, "cudaMemcpy failed!");
	cudaStatus = cudaMemcpy(dev_velocity, velocityOfMyPoints, (size) * (dim) * sizeof(double), cudaMemcpyHostToDevice);
	checkError(cudaStatus, dev_velocity, "cudaMemcpy failed!");

	updateLocationOfAllPointsWithKernel << <numOfBlocks, numOfThreadsPerBlock >> >(dev_coords, dev_velocity, size, dim, dt);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	sprintf(errorBuffer, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	sprintf(errorBuffer, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	// Copy from GPU buffer to host memory.

	//Copy back to CPU each point new coords
	cudaStatus = cudaMemcpy(coordsOfMyPoints, dev_coords, size * (dim + 1) * sizeof(double), cudaMemcpyDeviceToHost);

	//Free memory on GPU 
	freeCudaMemory(2, dev_coords, dev_velocity);

	return cudaStatus;
}

void freeCudaMemory(int count, ...)
{
	int i;
	void* dev_arr;
	va_list ap;

	va_start(ap, count);

	// traverse rest of the arrays for free cuda memory 
	for (i = 0; i < count; i++)
	{
		dev_arr = va_arg(ap, void*);
		cudaFree(dev_arr);
	}

	va_end(ap);
}

void checkError(cudaError_t cudaStatus, void* dev_arr, const char* errorMessage)
{
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, errorMessage);
		fprintf(stderr, "\n");
		fflush(NULL);
		cudaFree(dev_arr);
	}
}

/*Function will check all points to calculate nMiss and fix weights if neccesary*/
int checkAllPoints(double* allCoords, double* weights, double* dev_coords, int* dev_set, double* dev_weights,int* dev_nMissArr, int dim, double alpha, int size, int numOfBlocks, int numOfThreadsPerBlock, cudaError_t* cudaStatus)
{
	int* nMissArr = (int*)calloc(size, sizeof(int));
	int index;
	int totalnMiss = 0;
	double result;
	int minIndex;

	//Copy new weights to GPU every new iteration
	*cudaStatus = cudaMemcpy(dev_weights, weights, (dim + 1) * sizeof(double), cudaMemcpyHostToDevice);
	checkError(*cudaStatus, dev_weights, "cudaMemcpy failed!");
	*cudaStatus = cudaMemcpy(dev_nMissArr, nMissArr, size * sizeof(int), cudaMemcpyHostToDevice); //Reset to zeros dev_nissMArr for next iteration
	checkError(*cudaStatus, dev_nMissArr, "cudaMemcpy failed!");

	// Launch a kernel on the GPU with one thread for each element.
	calcnMissWithKernel << <numOfBlocks, numOfThreadsPerBlock >> > (dev_coords, dev_set, dev_weights, dev_nMissArr, dim, size);

	// Copy to CPU buffer nMissArr.
	*cudaStatus = cudaMemcpy(nMissArr, dev_nMissArr, size * sizeof(int), cudaMemcpyDeviceToHost);
	checkError(*cudaStatus, nMissArr, "cudaMemcpy failed!");

	//Sum all nMissArr to a totalnMiss with OpenMP
	totalnMiss = sumnMiss(nMissArr, size,&minIndex);

	if (totalnMiss == 0) //if we finished inner loop and all points are correct return from function
		return totalnMiss;

	else
	{
		//Calculate new weights for next iteratrion
		fixWeights(weights, allCoords + minIndex * (dim + 1), dim, alpha, (-1) * nMissArr[minIndex]);
	}

	free(nMissArr);

	return totalnMiss;
}

/*Sum the nMiss with OMP and return the min index of the prooblematic coord -if exists-*/
int sumnMiss(int* nMissArr, int size,int* minIndex)
{
	int i;
	int nMiss = 0;
	*minIndex = size;

#pragma omp parallel for reduction(+:nMiss) reduction(min:*minIndex)
	for (i = 0; i < size; i++)
	{
		if (nMissArr[i] == 1 || nMissArr[i] == -1)
		{
			//min index will be equal to the index to the min prooblmatic coords index
			if (i < *minIndex)
				*minIndex = i;
		// if nMissarr is equal to 1 or -1 it means a point is not in her place - which mean we need to increase the nmiss
			nMiss++; 
		}		
	}

	return nMiss;
}

/*function will fix the old weights  according to the formula*/
void fixWeights(double* weights, double* coords, int k, double alpha, int sign)
{
	int i;

	for (i = 0; i < k + 1; i++)
	{
		weights[i] = weights[i] + alpha * sign * coords[i];		
	}
}
