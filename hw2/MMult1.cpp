// g++ -std=c++11 -O3 -march=native MMult1.cpp && ./a.out
#include <stdio.h>
#include <math.h>
// #include <omp.h> 
#include "utils.h"
#include "cstring"
#define BLOCK_SIZE 64

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  for (long e = 0; e < n; e+=BLOCK_SIZE) {
    for (long f = 0; f < k; f+=BLOCK_SIZE) {
      for (long g = 0; g < m; g+=BLOCK_SIZE){


          //Block Multiplication
          for (long j = e; j < e + BLOCK_SIZE && j < n; j++) {
            for (long p = f; p < f + BLOCK_SIZE && p < k; p++) {
              for (long i = g; i < g + BLOCK_SIZE && i < m; i++) {
                double A_ip = a[i+p*m];
                double B_pj = b[p+j*k];
                double C_ij = c[i+j*m];
                C_ij = C_ij + A_ip * B_pj;
                c[i+j*m] = C_ij;
              }
            }
          }


      }
        
      }
    }
  }



int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = (m*2*n*k)*NREPEATS/time/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = 4*m*n*k*sizeof(double)*NREPEATS/time/1e9; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10ld %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP:
// To do that, you have to add -fopenmp to the compiler and comment in the
// omp.h header file
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.

//Result:
  //-O3
    // Dimension       Time    Gflop/s       GB/s        Error
    //       64   0.446847   4.476157  71.618519 0.000000e+00
    //      128   0.447588   4.469922  71.518745 0.000000e+00
    //      192   0.450416   4.462808  71.404921 0.000000e+00
    //      256   0.454951   4.425240  70.803845 0.000000e+00
    //      320   0.455015   4.464943  71.439086 0.000000e+00
    //      384   0.456610   4.464269  71.428302 0.000000e+00
    //      448   0.486041   4.439890  71.038238 0.000000e+00
    //      512   0.513937   4.178497  66.855953 0.000000e+00
    //      576   0.514322   4.458752  71.340030 0.000000e+00
    //      640   0.470855   4.453927  71.262830 0.000000e+00
    //      704   0.470033   4.453901  71.262419 0.000000e+00
    //      768   0.615731   4.414118  70.625885 0.000000e+00
    //      832   0.517707   4.449858  71.197729 0.000000e+00
    //      896   0.649347   4.431052  70.896838 0.000000e+00
    //      960   0.796290   4.444288  71.108602 0.000000e+00
    //     1024   0.573213   3.746400  59.942400 0.000000e+00
    //     1088   0.590353   4.363195  69.811126 0.000000e+00
    //     1152   0.689699   4.433306  70.932899 0.000000e+00
    //     1216   0.812577   4.425539  70.808621 0.000000e+00
    //     1280   0.953600   4.398390  70.374247 0.000000e+00
    //     1344   1.096349   4.428726  70.859623 0.000000e+00
    //     1408   1.261168   4.426548  70.824769 0.000000e+00
    //     1472   1.441023   4.426726  70.827618 0.000000e+00
    //     1536   1.716755   4.221777  67.548427 0.000000e+00
    //     1600   1.852331   4.422535  70.760559 0.000000e+00
    //     1664   2.082697   4.424497  70.791957 0.000000e+00
    //     1728   2.343350   4.403765  70.460232 0.000000e+00
    //     1792   2.793011   4.120704  65.931265 0.000000e+00
    //     1856   2.891552   4.422144  70.754304 0.000000e+00
    //     1920   3.203793   4.418443  70.695093 0.000000e+00
    //     1984   3.530121   4.424512  70.792192 0.000000e+00

  //-O0
      //  Dimension       Time    Gflop/s       GB/s        Error
      //   64   4.170312   0.479618   7.673895 0.000000e+00
      //  128   4.245484   0.471250   7.539995 0.000000e+00
      //  192   4.145991   0.484835   7.757354 0.000000e+00
      //  256   4.113669   0.489409   7.830541 0.000000e+00
      //  320   4.234083   0.479824   7.677189 0.000000e+00
      //  384   4.326581   0.471141   7.538263 0.000000e+00
      //  448   4.606692   0.468442   7.495076 0.000000e+00
      //  512   4.556119   0.471341   7.541449 0.000000e+00
      //  576   4.828667   0.474921   7.598738 0.000000e+00
      //  640   4.384402   0.478321   7.653138 0.000000e+00
      //  704   4.310609   0.485658   7.770529 0.000000e+00
      //  768   5.932143   0.458166   7.330663 0.000000e+00
      //  832   4.810885   0.478856   7.661697 0.000000e+00
      //  896   6.103426   0.471423   7.542761 0.000000e+00
      //  960   7.345805   0.481764   7.708223 0.000000e+00
      // 1024   4.609828   0.465849   7.453584 0.000000e+00
      // 1088   5.327009   0.483541   7.736656 0.000000e+00
      // 1152   6.381898   0.479113   7.665802 0.000000e+00
      // 1216   7.442676   0.483172   7.730749 0.000000e+00
      // 1280   8.681287   0.483143   7.730289 0.000000e+00
      // 1344  10.238775   0.474220   7.587519 0.000000e+00
      // 1408  11.529032   0.484223   7.747562 0.000000e+00
      // 1472  13.297851   0.479702   7.675240 0.000000e+00
      // 1536  15.701804   0.461588   7.385401 0.000000e+00
      // 1600  17.298794   0.473559   7.576944 0.000000e+00
      // 1664  19.196334   0.480034   7.680538 0.000000e+00
      // 1728  21.947186   0.470200   7.523196 0.000000e+00
      // 1792  23.992956   0.479690   7.675033 0.000000e+00
      // 1856  26.735317   0.478276   7.652416 0.000000e+00
      // 1920  28.841551   0.490812   7.852990 0.000000e+00
      // 1984  32.350217   0.482812   7.724987 0.000000e+00
