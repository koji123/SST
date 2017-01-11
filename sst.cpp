/*
 * sst.cpp
 *
 *  Created on: 2011/01/07
 *      Author: ueno
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <sys/time.h>
#include "lapackw.h"

#define FELIX_SST_EVR 0
#define FELIX_SST_PRODUCT_H_OPT 2

#include "felixsst.h"

// ret : in microsecond
inline long long sub_timeval(struct timeval l, struct timeval r)
{
	return ((long long)l.tv_sec*1000000 + l.tv_usec) - ((long long)r.tv_sec*1000000 + r.tv_usec);
}

template<typename REAL>
void make_sample_data(REAL* s, int len, double tick, int cp)
{
	double r[] = { 0.057, 0.0851, 0.063, 0.45  };
	int state = 0;
	double x = 0.0f;

	for(int i = 0; i < len; i++){
		s[i] = sin(x);

		if( (i % cp) == 0) {
			if( ++state == sizeof(r)/sizeof(r[0])){
				state = 0;
			}
		}

		x += r[state] * tick;
	}
}

#define PRINT_SCORE 1
#define FELIX_SST 1
#define STREAM_OFFSET 0
#define STREAM_WAVE 1500

template<typename REAL>
int sst_test(int window_size, int iter)
{
  int gap = 8;
#if FELIX_SST
	FelixSSTWorkspace<REAL> work;
#else
	NaiveSSTWorkspace<REAL> work;
#endif
  REAL *stream;
  REAL a[window_size]; // to hold state
  int len = (4 * window_size + gap + iter + STREAM_OFFSET);

  fprintf(stderr, "WND=%d,STREAM=%d,precision=%d,method=%s\n", window_size, iter, (int)sizeof(REAL), FELIX_SST ? "IKA-SST" : "SVD-SST");

	stream = (REAL*)malloc(len*sizeof(REAL));
  make_sample_data(stream, len, 1, STREAM_WAVE);
	if( STREAM_OFFSET ) memmove(stream, stream + STREAM_OFFSET, (len-STREAM_OFFSET)*sizeof(REAL));

	struct timeval before, after;
	long long gpu_time;
	gettimeofday(&before, NULL);

	for(int i = 0; i < iter; i++){
    REAL score = work.computeScore(stream + i, window_size, gap, 3, a, i == 0);
#if PRINT_SCORE
		printf("score %f\n", score);
#endif
	}

	gettimeofday(&after, NULL);
	gpu_time = sub_timeval(after, before);
  fprintf(stderr, "CPU(total):%f\n", (double)gpu_time / 1000000);

	free(stream);

	return 0;
}

int main(int argc, char *argv[])
{
  int window_size = 0;
  int iter = 0;
  if (argc >= 2){
    window_size = atoi(argv[1]);
  }
  if (window_size == 0){
    window_size = 250;
  }
  if (argc >= 3){
    iter = atoi(argv[2]);
  }
  if (iter == 0){
    iter = 10000;
  }
  return sst_test<float>(window_size, iter);
}
