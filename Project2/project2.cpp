/* Xiaoying Li
 * lixiaoyi@oregonstate.edu
 * Project #2 Numeric Integration with OpenMP Reduction */


#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

 // setting the number of tries to discover the maximum performance
#ifndef NUMTRIES
#define NUMTRIES	10
#endif

#define N		 4.	// setting the value of exponent N
#define XMIN	-1.
#define XMAX	 1.
#define YMIN	-1.
#define YMAX	 1.


float Height(int, int);	// function prototype

int NUMT = 1;	// setting the number of threads
int NUMNODES = 1024;	// setting the number of subdivisions


// main program
int main(int argc, char* argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif

	if (argc >= 2)
		NUMT = atoi(argv[1]);
	if (argc >= 3)
		NUMNODES = atoi(argv[2]);

	omp_set_num_threads(NUMT);	// set the number of threads to use in the for-loop

	// the area of a single full-sized tile
	float fullTileArea = (((XMAX - XMIN) / (float)(NUMNODES - 1)) * ((YMAX - YMIN) / (float)(NUMNODES - 1)));

	// get ready to record the maximum performance and the volume
	double maxMegaNodes = 0.;
	float volume;

	// looking for the maximum performance
	for (int t = 0; t < NUMTRIES; t++)
	{
		volume = 0.;
		double time0 = omp_get_wtime();

		// sum up the weighted heights into the variable "volume" using an OpenMP for loop and a reduction
#pragma omp parallel for default(none), shared(NUMNODES), reduction(+: volume)
		for (int i = 0; i < NUMNODES * NUMNODES; i++)
		{
			int iu = i % NUMNODES;
			int iv = i / NUMNODES;
			float z = Height(iu, iv);

			// check if the subdivition is a corner
			if ((iu == 0 && iv == 0) || (iu == 0 && iv == NUMNODES - 1) || 
				(iu == NUMNODES - 1 && iv == 0) || (iu == NUMNODES - 1 && iv == NUMNODES - 1)) {
				z = z / 4;	// tiles in the corners are quarter-sized
			}

			// check if the subdivition is an edge
			else if ((iu == 0 && iv != 0 && iv != NUMNODES - 1) || (iu == NUMNODES - 1 && iv != 0 && iv != NUMNODES - 1) ||
					 (iv == 0 && iu != 0 && iu != NUMNODES - 1) || (iv == NUMNODES - 1 && iu != 0 && iu != NUMNODES - 1)) {
				z = z / 2;	// tiles along the edges are half-sized
			}

			// else, the subdivition is a middle tile, tiles in the middle of the floor are full-sized
			else {
			}

			volume += z;
		}

		double time1 = omp_get_wtime();
		double megaNodes = (double)NUMNODES * NUMNODES / (time1 - time0) / 1000000.;
		if (megaNodes > maxMegaNodes) {
			maxMegaNodes = megaNodes;
		}
	}

	// compute the total volume
	volume = volume * fullTileArea * 2;

	// Print out: (1) the number of threads, (2) the number of subdivisions, 
	//            (3) the total volume, and (4) the maxMegaHeightsPerSecond
	printf("%d\t%d\t%f\t%lf\n", NUMT, NUMNODES, volume, maxMegaNodes);
}


// the function to evaluate the height at a given iu and iv
// provided in the project description
float Height(int iu, int iv)	// iu,iv = 0 .. NUMNODES-1
{
	float x = -1. + 2. * (float)iu / (float)(NUMNODES - 1);	// -1. to +1.
	float y = -1. + 2. * (float)iv / (float)(NUMNODES - 1);	// -1. to +1.

	float xn = pow(fabs(x), (double)N);
	float yn = pow(fabs(y), (double)N);
	float r = 1. - xn - yn;

	if (r < 0.) {
		return 0.;
	}
		
	float height = pow(1. - xn - yn, 1. / (float)N);
	return height;
}