#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#define NUM_THREADS 3
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow

//CPU specs:
// Architecture:        x86_64
// CPU op-mode(s):      32-bit, 64-bit
// Core(s):  6
// Model name:          Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
//Results with different NUM_THREADS:
//NUM_THREADS 12:
// sequential-scan = 0.393995s
// parallel-scan   = 0.295449s
// error = 0
// NUM_THREADS 6:
// sequential-scan = 0.411884s
// parallel-scan   = 0.275807s
// error = 0
//NUM_THREADS 3:
// sequential-scan = 0.383975s
// parallel-scan   = 0.340615s
// error = 0



void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  int chunksize = ceil(double(n)/double(NUM_THREADS));
  int numOfChunksize = NUM_THREADS-NUM_THREADS*chunksize+n-1;
  //printf("chunksize:%d numOfChunksize:%d\n",chunksize,numOfChunksize);
  long int sum=0;
  if (n == 0) return;
  //1. Parallel Computing
  prefix_sum[0]=0;
  #pragma omp parallel
  {
    long int localTemp = 0;
    #pragma omp for 
    for (long i=1; i<n; i++){
      localTemp += A[i-1];
      prefix_sum[i] = localTemp;
    }
  }

  //2. Plus offset. 
  int count=0;
  int temp = chunksize;
  for (long i=0; i<n; i+=chunksize){
        count++;
        if(count>numOfChunksize){
          chunksize=temp-1;
        }
       for (long j = i+1; j<i+chunksize+1 and j<n; j++){
         prefix_sum[j] += prefix_sum[i];
         //printf("i:%ld j:%ld prefix_sumi:%ld chunksize:%d\n",i,j,prefix_sum[i],chunksize);
      }
    }

}

int main() {
  long N = 100000000;
  omp_set_num_threads(NUM_THREADS);
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand(); 
  //for (long i = 0; i < N; i++)  printf("A:%ld\n",A[i] );

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  //for (long i = 0; i < N; i++) printf("B0:%ld\n",B0[i] );
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  //for (long i = 0; i < N; i++) printf("B1:%ld\n",B1[i] );
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}

