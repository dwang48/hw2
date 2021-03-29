
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <time.h>
#define N 1000


void initialize(double *u0, double *f);
double* iterate_jacobi(double *u0, double *f, double h);
void printState(double *u0);
double* parallel_jacobi(double *u0, double *f, double h);
int main() {
	int i;
	double h = 1/double(N-1);
	double *u = (double *) malloc(sizeof(double)*N*N);
	double *f = (double *) malloc(sizeof(double)*N*N);
	initialize(u,f);
	#ifdef _OPENMP
	double tt = omp_get_wtime();
	for(i=0;i<100;i++){
		u=parallel_jacobi(u,f,h);
		
	}
	// printf("u_parallel:\n");
	// printState(u);
	
	printf("parallel_jacobi:%fs\n",omp_get_wtime()-tt);
	#endif
	initialize(u,f);
	clock_t tic = clock();
	for(i=0;i<100;i++){
		u=iterate_jacobi(u,f,h);
		
	}
	clock_t toc = clock();
	printf("iterate_jacobi:%fs\n",(double) (toc-tic)/CLOCKS_PER_SEC);
	// printf("u_iter:\n");
	// printState(u);
	

}

void initialize(double *u0, double *f){
	int i;
	for(i = 0; i < N*N; i++){
		u0[i] = 0;
	}

	for(i = 0; i < N*N; i++){
		f[i] = 1;
	}

}

double* iterate_jacobi(double *u_k, double *f, double h){
	int i,j;
	double h_squared = h*h;
	double *u_k1 = (double *) malloc(sizeof(double)*N*N);
	for(i=1;i<N-1;i++){
		for(j=1;j<N-1;j++){
			u_k1[i*N+j] = 0.25*(h_squared*f[i*N+j] + u_k[(i-1)*N + j] + u_k[i*N + j-1] + u_k[(i+1)*N + j] + u_k[i*N + j+1]);
		} 
	} 
	return u_k1;
}

double* parallel_jacobi(double *u_k, double *f, double h){
	int i,j;
	double h_squared = h*h;
	double *u_k1 = (double *) malloc(sizeof(double)*N*N);
	#pragma omp parallel private(i,j)
	{
		#pragma omp for
		for(i=1;i<N-1;i++){
			for(j=1;j<N-1;j++){
				u_k1[i*N+j] = 0.25*(h_squared*f[i*N+j] + u_k[(i-1)*N + j] + u_k[i*N + j-1] + u_k[(i+1)*N + j] + u_k[i*N + j+1]);
			} 
		} 
	}
	
	return u_k1;
}
void printState(double *u0){
	int i,j;
	for(i=1;i<N-1;i++){
		for(j=1;j<N-1;j++){
			printf("%f ",u0[i*N+j]);
		} 
		printf("\n");
	} 
	printf("-----------------\n");
}
