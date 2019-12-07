
#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h> 
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "main.h"

#define FILE_INPUT_NAME "input.txt"
#define FILE_OUTPUT_NAME "output.txt"
#define TIME_NOT_FOUND -1
#define TIME_FOUND 1
#define TERMINATION_TAG 1
#define CONTINUE_TAG 0
#define DATA_NUM_OF_ATTRIBUTES 9
#define MASTER 0

extern int calcnMissWithCuda(double* allCoords, int* allSets, double* weights, int dim, int limit, double alpha, int size);
cudaError_t updateLocationWithCuda(double* coordsOfMyPoints, double* velocityOfMyPoints, int size, int dim, double dt);


int main(int argc, char *argv[])
{
	//All points values
	double* allCoords;
	double* allVelocity;
	int* allSets;
	double* weights;

	Data data;
	double startTime;
	double endTime;
	int succeeded = 0;
	double q;
	
	//Results from proccessers variables
	int* results;
	double* allQ;
	double* allWeights;


	//MPI variables
	int  namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Datatype dataType;

	//MPI init variables
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;

	//Let all process "know" data type
	createDataType(&dataType);

	if (myid == MASTER) //Handle by Master
	{
		handleByMaster(&startTime, &data, &dataType, &allCoords, &allVelocity, &allSets, &numprocs, &results, &allQ, &allWeights);
	}
	else //Handle by Slave
	{
		handleBySlave(&data, &dataType, &allCoords, &allVelocity, &allSets);
	}

	//Broad cast to all process the coords, velocitys, and sets
	broadcastPointToAllProcesses(allCoords, allVelocity, allSets, &data);
	//Set the coords of each process before start looping
	updateLocationWithCuda(allCoords, allVelocity, data.n, data.k, data.localCurrentTime);
	
	//The most important method of this program - THE PERCEPTRON ALGORITHEM
	weights =  perceptron(&data, allCoords, allVelocity, allSets,&succeeded,&q);
	
	//Master gather all results
	gatherAllResults(&succeeded, results, weights, allWeights, &q, allQ, data.k);

	//Free allocation of points data
	freeAllAllocations(4, allCoords, allVelocity, allSets, weights); // free all arrays after Master gather the results

	//Master write result to the file and check the total time of the concurency program
	if (myid == MASTER) 
	{
		writeResultsToFile(FILE_OUTPUT_NAME, &data, results, allWeights, allQ, numprocs);
		endTime = MPI_Wtime();
		printf("total time is %lf\n", endTime - startTime);
		freeAllAllocations(3, results, allWeights, allQ); //free the allocation of master's arrays
	}

	MPI_Finalize();
}

/*Function will calculate the weights of the line according to the perceptron algorirthm*/
double* perceptron(Data* data, double* allCoords, double* allVelocity, int* allSets,int* succeeded, double* q)
{
	int nMiss;
	double* weights;
	
	//Main loop of the algorithm
	while ((data->localCurrentTime < data->localMaxTime) && (*succeeded == 0))
	{
		weights = guessInitialWeights(data->k);
		nMiss = calcnMissWithCuda(allCoords, allSets, weights, data->k, data->limit, data->alpha, data->n);
		*q = calculateQualityOfClassifier(nMiss, data->n);

		if (*q > data->qc) //it means we didn't acheived the proper accurate
		{
			QisBiggerThenQCCase(weights, data);
		}
		else
		{
			QisSmallerThenQCCase(succeeded);
		}

		updateLocationWithCuda(allCoords, allVelocity, data->n, data->k, data->dt);
	}

	return weights;
}
/*Function contains all the method the master need to do BEFORE starting the perceptron algorithm*/
void handleByMaster(double* startTime, Data* data, MPI_Datatype* dataType, double** allCoords, double** allVelocity, int** allSets,int* numprocs, int** results, double** allQ, double** allWeights)
{
	*startTime = MPI_Wtime(); //Start counting time
	readPointsFromFile(FILE_INPUT_NAME, data, allCoords, allVelocity, allSets);
	matserSendData(data, dataType, numprocs);
	//Alloctaing space for future results
	mallocResultsArrays(results, allQ, allWeights, *numprocs, data->k);
}

/*Function contains all the method the slave need to do BEFORE starting the perceptron algorithm*/
void handleBySlave(Data* data, MPI_Datatype* dataType, double** allCoords, double** allVelocity, int** allSets)
{
	int message;
	MPI_Status status;

	MPI_Recv(&message, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	if (status.MPI_TAG == TERMINATION_TAG)
		MPI_Finalize();

	//All process that survived here are need in the program!
	MPI_Recv(data, 1, *dataType, MASTER, 0, MPI_COMM_WORLD, &status);
	//Alloctaing space for arrays
	mallocPointsArray(allCoords, allVelocity, allSets, data->n, data->k);

}

/*Function gather all results from everyone to the master*/
void gatherAllResults(int* succeeded, int* results, double* weights, double* allWeights, double* q, double* allQ, int dim)
{
	MPI_Gather(succeeded, 1, MPI_INT, results, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Gather(weights, dim + 1, MPI_DOUBLE, allWeights, dim + 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Gather(q, 1, MPI_DOUBLE, allQ, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

}

/*Function will allocate space to all points array*/
void mallocPointsArray(double** allCoords, double** allVelocity, int** allSets, int n, int dim)
{
	*allCoords = (double*)malloc(n * (dim + 1) * sizeof(double));
	if (!checkAllocation(*allCoords))
		return;
	*allVelocity = (double*)malloc(n * dim * sizeof(double));
	if (!checkAllocation(*allVelocity))
		return;
	*allSets = (int*)malloc(n * sizeof(int));
	if (!checkAllocation(*allSets))
		return;
}

/*Function will allocate space for all master arrays*/
void mallocResultsArrays(int** results, double** allQ, double** allWeights,int numprocs, int dim) 
{
	*results = (int*)malloc(numprocs * sizeof(int));
	if (!checkAllocation(*results))
		return;
	*allQ = (double*)malloc(numprocs * sizeof(double));
	if (!checkAllocation(*allQ))
		return;
	*allWeights = (double*)malloc(numprocs * (dim + 1) * sizeof(double));
	if (!checkAllocation(*allWeights))
		return;

}

/*Function will broadcast to EVERYONE the points data*/
void broadcastPointToAllProcesses(double* allCoords, double* allVelocity,int* allSets,Data* data)
{
	MPI_Bcast(allCoords, data->n * (data->k + 1), MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(allVelocity, data->n * (data->k), MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(allSets, data->n, MPI_INT, MASTER, MPI_COMM_WORLD);
}

void writeResultsToFile(const char* fileName, Data* data,int* results, double* allWeights,double* allQ, int numproc)
{
	const int NOT_FOUND = -1;
	FILE *fp;
	int i;
	int index = NOT_FOUND;

	if ((fp = fopen(fileName, "wt")) == NULL)
	{
		printf("Cannot Open File '%s'", fileName);
		return;
	}

	//Looking for the right result
	searchForTheRightResult(results, numproc, &index);

	if (index != NOT_FOUND)
	{
		fprintf(fp, "Alpha minimum = %lf   q = %lf\n", data->alpha, allQ[index]);
		printf("Alpha minimum = %lf   q = %lf\n", data->alpha, allQ[index]);
		fflush(NULL);

		for (i = 0; i < (data->k + 1); i++)
		{
			fprintf(fp, "weights[%d] = %lf\n", i, allWeights[index * (data->k + 1) + i]);
			printf("weights[%d] = %lf\n", i, allWeights[index * (data->k + 1) + i]);
			fflush(NULL);
		}
	}
	else
	{
		fprintf(fp, "time was not found\n");
		printf("time was not found\n");
		fflush(NULL);
	}

	fclose(fp);
}

/*Function will look for the process with the shortest time that seccueded the find a result*/
void searchForTheRightResult(int* results,int numproc, int* index)
{
	const int EXIST = 1;
	int i;

	for (i = 0; i < numproc; i++)
	{
		//The process with the shortest dt will be write in the file!
		if (results[i] == EXIST)
		{
			*index = i; //save the index of the process with the fastest result
			break;
		}
	}
}

/*Functioni will check if the local time can be increased - and if does, it will increase the local current time of the process*/
void QisBiggerThenQCCase(double* weights, Data* data)
{
	//if localCurrentTime + dt is less then localMaxTime returns True
	if (incrementTime(data->localCurrentTime,data->dt, data->localMaxTime)) 
	{
		data->localCurrentTime += data->dt;
	}
}
/*Function will change succeeded to 1 which mean the process found the right result*/
void QisSmallerThenQCCase(int* succeeded)
{
	*succeeded = 1;
}

double calculateQualityOfClassifier(int nMiss, int n)
{
	return (double)nMiss / (double)n;
}

/*Function will increment the time only if global time + dt is less then tMax(and returns 1), else returns 0*/
int incrementTime(double currentTime, double dt, double maxTime)
{
	if (currentTime + dt <= maxTime)
		return 1;
	else
		return 0;
}
/*Function use by master send the data of the file (like tMax, dt, q etc) and will split the global time to all process*/
void matserSendData(Data* data, MPI_Datatype* dataType, int* numprocs)
{
	int id;
	int message = 0;
	double slice = data->tMax / *numprocs;
	int numOfIterationsPerProcess = slice / data->dt;
	int allProcessNum = *numprocs; //saving value of real num process before decrements it if neccesary.
	
	while (numOfIterationsPerProcess == 0)
	{
		(*numprocs)--; //if we are here it means we dont need all process in this program
		slice = data->tMax / *numprocs;
		numOfIterationsPerProcess = slice / data->dt;
	}

	//Sending to all process if there are needed in the program or not
	for (id = 1; id < allProcessNum; id++)
	{
		if (id > *numprocs - 1)
			MPI_Send(&message, 1, MPI_INT, id, TERMINATION_TAG, MPI_COMM_WORLD);	
		else
			MPI_Send(&message, 1, MPI_INT, id, CONTINUE_TAG, MPI_COMM_WORLD);		
	}

	//initial before sending
	data->localCurrentTime = 0;
	data->localMaxTime = data->localCurrentTime + (data->dt * numOfIterationsPerProcess);

	for (id = 1; id < *numprocs - 1; id++)
	{
		data->numprocs = *numprocs; 
		data->localCurrentTime = data->localCurrentTime + (data->dt * numOfIterationsPerProcess);
		data->localMaxTime = data->localCurrentTime + (data->dt * numOfIterationsPerProcess);
		
		//Send to all processors the data from file
		MPI_Send(data, 1,*dataType, id, 0, MPI_COMM_WORLD);
	}

	if (*numprocs > 1)
	{
		//Last process taking care of reminder - if exsists
		data->localCurrentTime = data->localCurrentTime + (data->dt * numOfIterationsPerProcess);
		data->localMaxTime = data->tMax;
		MPI_Send(data, 1, *dataType, id, 0, MPI_COMM_WORLD);
	}
	
	//update master current time  to start with
	data->localCurrentTime = 0;
	data->localMaxTime = data->localCurrentTime + (data->dt * numOfIterationsPerProcess);

}

/*Function wil initialize the weights to 0 at first*/
double* guessInitialWeights(int dim)
{
	double* weights;

	weights = (double*)calloc(dim + 1, sizeof(double));

	return weights;
}

/*Function will read all points and values from file by MASTER*/
int readPointsFromFile(const char* fileName, Data* data,double** allCoords, double** allVelocity, int** allSet)
{
	int i;
	const int ONE = 1;
	FILE *fp;

	if ((fp = fopen(fileName, "rt")) == NULL)
	{
		printf("Cannot Open File '%s' \n", fileName);
		return 0;
	}

	//Reads total number of points
	fscanf(fp, "%d", &(data->n));
	//Reads number of coordinates of points
	fscanf(fp, "%d", &(data->k));
	//Reads increment value of t
	fscanf(fp, "%lf", &(data->dt));
	//Reads maximum value of t
	fscanf(fp, "%lf", &(data->tMax));
	//Reads the conversion ratio
	fscanf(fp, "%lf", &(data->alpha));
	//Reads the the maximum number of iterations
	fscanf(fp, "%d", &(data->limit));
	//Reads the Quality of Classifier to be reached 
	fscanf(fp, "%lf", &(data->qc));

	mallocPointsArray(allCoords,allVelocity,allSet,data->n, data->k);

	for (i = 0; i < data->n; i++)
	{
		readPoint(fp, data->k,*(allCoords) + i * (data->k + 1) ,*(allVelocity) + i * (data->k),*(allSet) + i);
	}

	fclose(fp);

	return 1;
}

/*Function will read a single point from file*/
void readPoint(FILE* fp, int dim, double* allCoords, double* allVelocity,int* allSets)
{
	int i;
	const int ONE = 1;

	allCoords[0] = ONE; //First value of coords is the constant ONE;

	//Reading coords values
	for (i = 1; i < dim + 1; i++)
	{
		fscanf(fp, "%lf", &(allCoords[i]));
	}
	//Reading velocity values
	for (i = 0; i < dim; i++)
	{
		fscanf(fp, "%lf", &(allVelocity[i]));
	}
	
	fscanf(fp, "%d", &(allSets[0]));
}

/*Function will create a new MPI_Datatype*/
void createDataType(MPI_Datatype* dataType)
{
	int blockLengths[DATA_NUM_OF_ATTRIBUTES] = { 1,1,1,1,1,1,1,1,1 };
	MPI_Aint disp[DATA_NUM_OF_ATTRIBUTES];
	MPI_Datatype types[DATA_NUM_OF_ATTRIBUTES] = { MPI_INT, MPI_INT, MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,
		MPI_DOUBLE,MPI_DOUBLE,MPI_INT,MPI_DOUBLE};

	disp[0] = offsetof(Data, n);
	disp[1] = offsetof(Data, k);
	disp[2] = offsetof(Data, alpha);
	disp[3] = offsetof(Data, dt);
	disp[4] = offsetof(Data, tMax);
	disp[5] = offsetof(Data, localCurrentTime);
	disp[6] = offsetof(Data, localMaxTime);
	disp[7] = offsetof(Data, limit);
	disp[8] = offsetof(Data, qc);

	MPI_Type_create_struct(DATA_NUM_OF_ATTRIBUTES, blockLengths, disp, types, dataType);
	MPI_Type_commit(dataType);
}

void checkCudaStatus(cudaError_t* cudaStatus)
{
	if (*cudaStatus != cudaSuccess) {
		fprintf(stderr, "use of Cuda failed!");
		return;
	}
}
int checkAllocation(const void* p)
{
	if (!p)
	{
		printf("ERROR! Not enough memory!");
		return 0;
	}
	return 1;
}

void freeAllAllocations(int count,...)
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