#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <omp.h>

// original LU code:
// https://en.wikipedia.org/wiki/LU_decomposition

// sets type of matrices
#define dfloat double

// hostLU:
// a. compute the LU decompositions of N matrices of size MxM
// b. assumes each matrix stored in column major

// void hostMGS(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat tol){

//   // loop over matrices
//   for(int n=0;n<N;++n){

//     int i, j, k;

//     // pointer to nth A and LU matrix storage
//     const dfloat *An = A+n*M*M;
//     dfloat *Qn = Q+n*M*M;
//     dfloat *Rn = R+n*M*M;

//     // i will be column
//     // j will be row
    
//     // column major storage
//     for (j = 0; j < M; ++j) {
//       for (i = 0; i < M; ++i) {
// 	      Qn[j + M*i] = An[j + M*i];
//       }
//     }

//     Rn[1+] = 

//     // loop over columns
//     for (i = 0; i < M; ++i) {
//       // loop over rows starting from diagonal
//       for (j = i + 1; j < M; ++j) {

	
//         LUn[j+M*i] /= LUn[i*M+i];
        
//         for (k = i + 1; k < M; k++)
//           LUn[j+M*k] -= LUn[j+M*i] * LUn[i+M*k];
//         }
//     }
//   }
// }
void printMatrix(dfloat *A, int M){
  for(int j=0;j<M;++j){ // row
    for(int i=0;i<M;++i){ // column
      printf("%e ", A[j+i*M]);
      if(i == M-1) printf("\n");
    }
  }
}

void hostMGSnoN(int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat tol){

  int i, j, j2;

  // tmp helper
  dfloat *V  = (dfloat*) calloc(M*M, sizeof(dfloat));
  // i will be column
  // j will be row
  
  // column major storage
  for (j = 0; j < M; ++j) {
    for (i = 0; i < M; ++i) {
      V[j + M*i] = A[j + M*i];
    }
  }
  // loop over columns
  for (i = 0; i < M; ++i){
    dfloat tmp = 0;
    for (j = 0; j<M; ++j){
      // TODO: is this allowed?
      tmp += V[j + M*i]*V[j + M*i];
      //R[i + M*i] += abs(V[j + M*i]); // using 1 norm to avoid sqrt
    }
    R[i + M*i] = sqrt(tmp);
    // printMatrix(V,M);
    // printf("i:%d, j:%d, %f\n", i, j, tmp);
    for (j = 0; j<M; ++j){
      Q[j + M*i] = V[j + M*i]/R[i + M*i];
    }
    //loop over rows starting from diagonal
    for (j = i + 1; j < M; ++j){
      for (j2 = 0; j2 < M; ++j2){
        R[i + M*j] += Q[j2 + M*i]*V[j2 + M*j];
      }
      for (j2 = 0; j2 < M; ++j2){
        V[j2 + M*j] = V[j2 + M*j]-R[i + M*j]*Q[j2 + M*i];
      }
    }
  }
}

int main(int argc, char **argv){

  if(argc<3) {
    printf("usage: ./cudaQR totalMatrixEntries matrixWidth\n");
    exit(-1);
  }
  
  int Ntotal = atoi(argv[1]);
  int M = atoi(argv[2]);

  int N = (Ntotal+M*M-1)/(M*M);

  dfloat *A  = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *Q  = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *R  = (dfloat*) calloc(N*M*M, sizeof(dfloat));

  
//   dfloat *h_A  = (dfloat*) calloc(N*M*M, sizeof(dfloat));
//   dfloat *h_LU = (dfloat*) calloc(N*M*M, sizeof(dfloat));
//   int    *h_P  = (int*)    calloc(N*M,   sizeof(int));

  for(int n=0;n<N;++n){
    //dfloat *An = h_A + n*M*M;
    for(int j=0;j<M;++j){ // row
      for(int i=0;i<M;++i){ // column
	      A[j+i*M] = i+1;
      }
    }
  }

  printf("A:\n");
  printMatrix(A, M);

  dfloat tol = 1e-14;
  hostMGSnoN(M, A, Q, R, tol);

  printf("Q:\n");
  printMatrix(Q, M);
  printf("R:\n");
  printMatrix(R, M);
//   

//   // 1. test hostLU
//   double ticLU = omp_get_wtime();
//   hostLU(N, M, h_A, h_LU, tol);  
//   double tocLU = omp_get_wtime();
//   double elapsedLU = tocLU-ticLU;
//   dfloat bandwidthLU = (2.*N*M*M/(elapsedLU*1.e9))*sizeof(dfloat); // matrix bandwidth in GB/s (ignoring caching)
  
//   printf("hostLU,    M:%02d, time:%3.2e s, diff:%3.2e, throughput:%3.2f GB/s\n",
// 	 M, elapsedLU, 0., bandwidthLU);

//   // 1.0 v0 kernel run
//   dfloat elapsedTimes[1000];
//   dfloat *c_A, *c_LU;
//   dfloat *h_gpuLU = (dfloat*) calloc(N*M*M, sizeof(dfloat));
//   cudaMalloc(&c_A, N*M*M*sizeof(dfloat));
//   cudaMalloc(&c_LU, N*M*M*sizeof(dfloat));

//   cudaMemcpy(c_A, h_A, N*M*M*sizeof(dfloat), cudaMemcpyHostToDevice);

//   // warm up
//   int Nops = 5;
//   for(int op=0;op<Nops;++op){
//     runDeviceLU(op, N, M, c_A, c_LU, tol);

//     // zero c_LU on DEVICE
//     cudaMemset(c_LU, 0, N*M*M*sizeof(dfloat));
    
//     elapsedTimes[op] = runDeviceLU(op, N, M, c_A, c_LU, tol);
    
//     cudaMemcpy(h_gpuLU, c_LU, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
    
//     dfloat maxDiff =0;
//     for(int n=0;n<N*M*M;++n){
//       dfloat diff = fabs(h_gpuLU[n]-h_LU[n]);
//       maxDiff = (maxDiff>diff) ? maxDiff:diff;
//     }

//     // matrix bandwidth in GB/s (ignoring caching)
//     dfloat bandwidthLUop = (2.*N*M*M/(elapsedTimes[op]*1.e9))*sizeof(dfloat); 
    
//     printf("kernel:%02d, M:%02d, time:%3.2e s, diff:%3.2e, throughput:%3.2f GB/s\n",
// 	   op, M, elapsedTimes[op], maxDiff, bandwidthLUop);
//   }
  
// #if 0
//   // 2. test hostPLU (pivoted LU)
//   double ticPLU = omp_get_wtime();
//   hostPLU(N, M, h_A, h_LU, h_P, tol);
//   double tocPLU = omp_get_wtime();
//   double elapsedPLU = tocPLU-ticPLU;
//   dfloat bandwidthPLU = (2.*N*M*M/(elapsedPLU*1.e9))*sizeof(dfloat); // matrix bandwidth in GB/s (ignoring caching)
  
//   printf("hostPLU took %g to factorize %d matrices of size %d x %d with throughput %g GB/s\n",
// 	 elapsedPLU, N, M, M, bandwidthPLU);
// #endif
  
  return 0;
}