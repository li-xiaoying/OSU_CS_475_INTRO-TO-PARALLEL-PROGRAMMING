/* Xiaoying Li
 * lixiaoyi@oregonstate.edu
 * Project #3 Functional Decomposition */


#define _USE_MATH_DEFINES
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <iostream>


int	NowYear;	// 2020 - 2025
int	NowMonth;	// 0 - 11

float NowPrecip;	// inches of rain per month
float NowTemp;	// temperature this month
float NowHeight;	// grain height in inches
int	NowNumDeer;	// number of deer in the current population
int NowHuntedDeer;	// number of deer hunted by the hunter this month 

const float GRAIN_GROWS_PER_MONTH = 9.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;

const float AVG_PRECIP_PER_MONTH = 7.0; // average
const float AMP_PRECIP_PER_MONTH = 6.0; // plus or minus
const float RANDOM_PRECIP = 2.0;	// plus or minus noise

const float AVG_TEMP = 60.0;	// average
const float AMP_TEMP = 20.0;	// plus or minus
const float RANDOM_TEMP = 10.0;	// plus or minus noise

const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

unsigned int seed = 0;	// put this at the top of the program to make it a global

// function prototypes
float SQR(float);
float Ranf(unsigned int*, float, float);
int Ranf(unsigned int*, int, int);
void GrainDeer();
void Grain();
void Hunter();
void Watcher();


// main program
int main(int argc, char* argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif

	// starting values
	NowMonth = 0;
	NowYear = 2020;
	NowNumDeer = 1;
	NowHeight = 1.;
	NowHuntedDeer = 0;

	// calculate the starting temperature and precipitation
	float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);

	float temp = AVG_TEMP - AMP_TEMP * cos(ang);
	NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);

	float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
	NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
	if (NowPrecip < 0.) {
		NowPrecip = 0.;
	}
	
	// start the threads with a parallel sections directive
	omp_set_num_threads(4);	// same as # of sections
#pragma omp parallel sections
	{
#pragma omp section
		{
			GrainDeer();
		}

#pragma omp section
		{
			Grain();
		}

#pragma omp section
		{
			Hunter();	// my own agent
		}

#pragma omp section
		{
			Watcher();
		}
	}	// implied barrier -- all functions must return in order to allow any of them to get past here

	return 0;
}


void GrainDeer()
{
	// compute a temporary next-value for this quantity based on the current state of the simulation
	int localNumDeer;

	while (NowYear < 2026) {	// return when the year hits 2026
		localNumDeer = NowNumDeer;
		// If the number of graindeer exceeds this value at the end of a month, decrease the number of graindeer by one.
		if (localNumDeer > NowHeight) {
			localNumDeer--;
		}

		// If the number of graindeer is less than this value at the end of a month, increase the number of graindeer by one.
		else if (localNumDeer < NowHeight) {
			localNumDeer++;
		}

		// subtracted the number of graindeer hunted by hunter
		localNumDeer = localNumDeer - NowHuntedDeer;

		// clamp NumDeer against zero
		if (localNumDeer < 0) {
			localNumDeer = 0;
		}

#pragma omp barrier	// DoneComputing barrier
		
		NowNumDeer = localNumDeer;	// copy the local variable into the global version
#pragma omp barrier	// DoneAssigning barrier

#pragma omp barrier	// DonePrinting barrier

	}
}


void Grain()
{
	// compute a temporary next-value for this quantity based on the current state of the simulation
	float localHeight;

	while (NowYear < 2026) {	// return when the year hits 2026
		localHeight = NowHeight;
		float tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10.));
		float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10.));
		localHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
		localHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

		// clamp NumHeight against zero
		if (localHeight < 0.) {
			localHeight = 0.;
		}

#pragma omp barrier	// DoneComputing barrier
		
		NowHeight = localHeight;	// copy the local variable into the global version
#pragma omp barrier	// DoneAssigning barrier

#pragma omp barrier	// DonePrinting barrier

	}
}


void Hunter()
{
	// compute a temporary next-value for this quantity based on the current state of the simulation
	int localHuntedDeer;

	while (NowYear < 2026) {	// return when the year hits 2026
		// If the amount of precipitation is no less than ¡°AVG_PRECIP_PER_MONTH¡±, 1/3 graindeer are hunted by the hunter.
		if (NowPrecip >= AVG_PRECIP_PER_MONTH) {
			localHuntedDeer = NowNumDeer / 3;
		}

		// If the amount of precipitation is more than ¡°AVG_PRECIP_PER_MONTH¡±, 1/2 graindeer are hunted by the hunter.
		else {
			localHuntedDeer = NowNumDeer / 2;
		}

#pragma omp barrier	// DoneComputing barrier
		
		NowHuntedDeer = localHuntedDeer;	// copy the local variable into the global version
#pragma omp barrier	// DoneAssigning barrier

#pragma omp barrier	// DonePrinting barrier
	}
}


void Watcher()
{
	FILE* data = fopen("data.csv", "w+");
	fprintf(data, "Month, Precipitation (cm), Temperature (C), Height (cm), Deer, HuntedDeer\n");

	while (NowYear < 2026) {	// return when the year hits 2026
		
#pragma omp barrier	// DoneComputing barrier
		
#pragma omp barrier	// DoneAssigning barrier

		// print result
		// change the units to ¡ãC and centimeters
		fprintf(data, "%d,%f,%f,%f,%d,%d\n", (NowMonth + 12 * (NowYear - 2020)) + 1, (NowPrecip * 2.54), (5. / 9.) * (NowTemp - 32), (NowHeight * 2.54), NowNumDeer, NowHuntedDeer);
		NowMonth++;	// increment time

		if (NowMonth > 11) {
			NowMonth = 0;
			NowYear++;
		}

		// calculate new environmental parameters
		float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);
		float temp = AVG_TEMP - AMP_TEMP * cos(ang);
		NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);
		float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
		NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);

		if (NowPrecip < 0.) {
			NowPrecip = 0.;
		}

#pragma omp barrier	// DonePrinting barrier
	}

	fclose(data);
}


// function that squares a given number
float SQR(float x)
{
	return x * x;
}


// function that returns a random float between a user-given low value and a high value
float Ranf(unsigned int* seedp, float low, float high)
{
	float r = (float)rand_r(seedp);	// 0 - RAND_MAX
	return(low + r * (high - low) / (float)RAND_MAX);
}


// function that returns a random integer between a user-given low value and a high value
int Ranf(unsigned int* seedp, int ilow, int ihigh)
{
	float low = (float)ilow;
	float high = (float)ihigh + 0.9999f;
	return (int)(Ranf(seedp, low, high));
}