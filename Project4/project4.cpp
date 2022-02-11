/* Xiaoying Li
 * lixiaoyi@oregonstate.edu
 * Project #4 Vectorized Array Multiplication/Reduction using SSE */


#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

#define SSE_WIDTH	4
#define NUMTRIES	10	// setting the number of tries to discover the maximum performance


int ARRAY_SIZE = 1000;	// setting the array size
int NUMT = 1;	// setting the number of threads
int TEST_CASE = 1;	// setting the test case

float SimdMulSum(float*, float*, int);	// function prototype


// main program
int main(int argc, char* argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "OpenMP is not supported here -- sorry.\n");
	return 1;
#endif

	if (argc >= 2) {
		ARRAY_SIZE = atoi(argv[1]);
	}
	if (argc >= 3) {
		NUMT = atoi(argv[2]);
	}
	if (argc >= 4) {
		TEST_CASE = atoi(argv[3]);
	}

	// inialize the arrays
	float* A = new float[ARRAY_SIZE];
	float* B = new float[ARRAY_SIZE];

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		A[i] = sqrt(i);
		B[i] = sqrt(i);
	}

	// get ready to record the maximum performance and the sum for control group (a for-loop with no multicore or SIMD)
	double controlGroupMaxMegaMults = 0.;
	float controlGroupSum;

	// looking for the maximum performance for control group (a for-loop with no multicore or SIMD)
	for (int t = 0; t < NUMTRIES; t++)
	{
		controlGroupSum = 0.;
		double time0 = omp_get_wtime();

		for (int i = 0; i < ARRAY_SIZE; i++) {
			controlGroupSum += A[i] * B[i];
		}

		double time1 = omp_get_wtime();
		double megaMults = (double)ARRAY_SIZE / (time1 - time0) / 1000000.;
		if (megaMults > controlGroupMaxMegaMults)
			controlGroupMaxMegaMults = megaMults;
	}

	// test case 1: array multiplication using SIMD
	if (TEST_CASE == 1) {
		double simdMaxMegaMults = 0.;	// get ready to record the maximum performance

		// looking for the maximum performance
		for (int t = 0; t < NUMTRIES; t++)
		{
			double time0 = omp_get_wtime();
			SimdMulSum(A, B, ARRAY_SIZE);
			double time1 = omp_get_wtime();
			double megaMults = (double)ARRAY_SIZE / (time1 - time0) / 1000000.;
			if (megaMults > simdMaxMegaMults)
				simdMaxMegaMults = megaMults;
		}

		// compute the speedup over the performance of control group
		float simdSpeedup = simdMaxMegaMults / controlGroupMaxMegaMults;
		// print out the results
		printf("ARRAY SIZE = %d\tNUMT = %d\tSIMD\n", ARRAY_SIZE, NUMT);
		printf("Control Group Peak Performance = %lf MegaMults/Sec\n", controlGroupMaxMegaMults);
		printf("SIMD Peak Performance = %lf MegaMults/Sec\n", simdMaxMegaMults);
		printf("SIMD Speedup = %f\n", simdSpeedup);
	}

	// test case 2: array multiplication using multithreading and SIMD
	else if (TEST_CASE == 2) {
		omp_set_num_threads(NUMT);	// set the number of threads to use in the for-loop
		int NUM_ELEMENTS_PER_CORE = ARRAY_SIZE / NUMT;
		double multicoreSimdMaxMegaMults = 0.;	// get ready to record the maximum performance

		// looking for the maximum performance
		for (int t = 0; t < NUMTRIES; t++)
		{
			double time0 = omp_get_wtime();

#pragma omp parallel
			{
				int first = omp_get_thread_num() * NUM_ELEMENTS_PER_CORE;
				SimdMulSum(&A[first], &B[first], NUM_ELEMENTS_PER_CORE);
			}

			double time1 = omp_get_wtime();
			double megaMults = (double)ARRAY_SIZE / (time1 - time0) / 1000000.;
			if (megaMults > multicoreSimdMaxMegaMults)
				multicoreSimdMaxMegaMults = megaMults;
		}

		// compute the speedup over the performance of control group
		float multicoreSimdSpeedup = multicoreSimdMaxMegaMults / controlGroupMaxMegaMults;
		// print out the results
		printf("ARRAY SIZE = %d\tNUMT = %d\tMuticore SIMD\n", ARRAY_SIZE, NUMT);
		printf("Control Group Peak Performance = %lf MegaMults/Sec\n", controlGroupMaxMegaMults);
		printf("Multicore SIMD Peak Performance = %lf MegaMults/Sec\n", multicoreSimdMaxMegaMults);
		printf("Multicore SIMD Speedup = %f\n", multicoreSimdSpeedup);
	}

	// test case 3: array multiplication using multithreading
	else if (TEST_CASE == 3) {
		omp_set_num_threads(NUMT);	// set the number of threads to use in the for-loop
		double multicoreMaxMegaMults = 0.;	// get ready to record the maximum performance and the sum
		float multicoreSum;

		// looking for the maximum performance
		for (int t = 0; t < NUMTRIES; t++)
		{
			multicoreSum = 0.;
			double time0 = omp_get_wtime();

#pragma omp parallel for default(none), shared(A, B, ARRAY_SIZE), reduction(+: multicoreSum)
			for (int i = 0; i < ARRAY_SIZE; i++) {
				multicoreSum += A[i] * B[i];
			}

			double time1 = omp_get_wtime();
			double megaMults = (double)ARRAY_SIZE / (time1 - time0) / 1000000.;
			if (megaMults > multicoreMaxMegaMults)
				multicoreMaxMegaMults = megaMults;
		}

		// compute the speedup over the performance of control group
		float multicoreSpeedup = multicoreMaxMegaMults / controlGroupMaxMegaMults;
		// print out the results
		printf("ARRAY SIZE = %d\tNUMT = %d\tMulticore\n", ARRAY_SIZE, NUMT);
		printf("Control Group Peak Performance = %lf MegaMults/Sec\n", controlGroupMaxMegaMults);
		printf("Multicore Peak Performance = %lf MegaMults/Sec\n", multicoreMaxMegaMults);
		printf("Multicore Speedup = %f\n", multicoreSpeedup);
		//printf("Multicore Sum = %f\n", multicoreSum);
	}
	
	//float simdSum = SimdMulSum(A, B, ARRAY_SIZE);
	//printf("Control Group Sum = %f\n", controlGroupSum);
	//printf("SIMD Sum = %f\n", simdSum);

	delete[] A;
	delete[] B;

	return 0;
}


// supplied SIMD SSE intrinsics code to run an array multiplication
float SimdMulSum(float* a, float* b, int len)
{
	float sum[4] = { 0., 0., 0., 0. };
	int limit = (len / SSE_WIDTH) * SSE_WIDTH;
	register float* pa = a;
	register float* pb = b;

	__m128 ss = _mm_loadu_ps(&sum[0]);
	for (int i = 0; i < limit; i += SSE_WIDTH)
	{
		ss = _mm_add_ps(ss, _mm_mul_ps(_mm_loadu_ps(pa), _mm_loadu_ps(pb)));
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps(&sum[0], ss);

	for (int i = limit; i < len; i++)
	{
		sum[0] += a[i] * b[i];
	}

	return sum[0] + sum[1] + sum[2] + sum[3];
}