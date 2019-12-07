#pragma once
#include <mpi.h>

typedef struct
{
	int n;
	int k;
	double alpha;
	double dt;
	double tMax;
	double localCurrentTime;
	double localMaxTime; 
	int limit;
	double qc;
	int numprocs;

}Data;

//File functions
int readPointsFromFile(const char* fileName, Data* data, double** allCoords, double** allVelocity, int** allSet);
void readPoint(FILE* fp, int dim, double* allCoords, double* allVelocity, int* allSets);

//Master functions
void matserSendData(Data* data, MPI_Datatype* dataType, int* numprocs);
void mallocResultsArrays(int** results, double** allQ, double** allWeights, int numprocs, int dim);
void handleByMaster(double* startTime, Data* data, MPI_Datatype* dataType, double** allCoords, double** allVelocity, int** allSets, int* numprocs, int** results, double** allQ, double** allWeights);
void createDataType(MPI_Datatype* dataType);

//Slave functions
void handleBySlave(Data* data, MPI_Datatype* dataType, double** allCoords, double** allVelocity, int** allSets);

//Cuda func
void checkCudaStatus(cudaError_t* cudaStatus);

double* perceptron(Data* data, double* allCoords, double* allVelocity, int* allSets, int* succeeded, double* q);
int checkAllocation(const void* p);
double* guessInitialWeights(int dim);
int sumnMiss(int* nMissArr, int size, int* minIndex);
double calculateQualityOfClassifier(int nMiss, int n);
int incrementTime(double currentTime, double dt, double maxTime);
void QisBiggerThenQCCase(double* weights, Data* data);
void QisSmallerThenQCCase(int* succeeded);
void writeResultsToFile(const char* fileName, Data* data, int* results, double* allWeights, double* allQ, int numproc);
void fixWeights(double* weights, double* coords, int k, double alpha, int sign);
void searchForTheRightResult(int* results, int numproc, int* index);
void broadcastPointToAllProcesses(double* allCoords, double* allVelocity, int* allSets, Data* data);
void freeAllAllocations(int count, ...);
void mallocPointsArray(double** allCoords, double** allVelocity, int** allSets, int n, int dim);
void gatherAllResults(int* succeeded, int* results, double* weights, double* allWeights, double* q, double* allQ, int dim);
