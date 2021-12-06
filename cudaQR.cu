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

__host__ __device__ void printMatrix(dfloat *A, int M){
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
  
  // i will be column
  // j will be row
  dfloat *V  = (dfloat*) calloc(M*M, sizeof(dfloat));

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
      tmp += V[j + M*i]*V[j + M*i];
      //printf("%e\n", V[j + M*i]);
      //printf("i:%d, j:%d, %e\n", i, j, tmp);
      //R[i + M*i] += abs(V[j + M*i]); // using 1 norm to avoid sqrt
    }
    
    R[i + M*i] = sqrt(tmp);
    //printf("i:%d, j:%d, %e\n", i, j, sqrt(tmp));
    // printMatrix(V,M);
    // printf("i:%d, j:%d, %f\n", i, j, tmp);
    for (j = 0; j<M; ++j){
      Q[j + M*i] = V[j + M*i]/R[i + M*i];
    }
    //loop over rows starting from diagonal
    // for (j = i + 1; j < M; ++j){
    //   for (j2 = 0; j2 < M; ++j2){ // super ugly
    //     R[i + M*j] += Q[j2 + M*i]*V[j2 + M*j];
    //   }
    //   for (j2 = 0; j2 < M; ++j2){ //super ugly
    //     V[j2 + M*j] = V[j2 + M*j]-R[i + M*j]*Q[j2 + M*i];
    //     printf("i:%d, j:%d, %e\n", i, j, V[j2 + M*j]);
    //   }
    // }

    
    for (j2 = 0; j2 < M; ++j2){ // super ugly
      for (j = i + 1; j < M; ++j){
        R[i + M*j] += Q[j2 + M*i]*V[j2 + M*j];
        printf("2i:%d, j:%d, %e\n", i, j, R[i + M*j]);
      }
    }
    for (j2 = 0; j2 < M; ++j2){ //super ugly
      for (j = i + 1; j < M; ++j){
        V[j2 + M*j] = V[j2 + M*j]-R[i + M*j]*Q[j2 + M*i];
        printf("3i:%d, j:%d, %e\n", i, j, V[j + M*j2] );
        //printf("i:%d, j:%d, %e\n", i, j, V[j2 + M*j]);
      }
    }
    
  }
}

// ___device___ void sumReduction(dfloat *s_V, int sM, int j){
//   int alive = (sM+1)/2;
//   while(alive>=1){
  
//     if(j<alive){
//       if (fabs(s_V[j]) < fabs(s_V[j+alive])){
//         s_V[j] = s_V[j+alive];
//       } 
//     }
//     if(alive>32)
//       __syncthreads();
    
//     alive /= 2;
//   }
// }

// __global__ void printMatrixKernel(dfloat *A, int M){
//   for(int j=0;j<M;++j){ // row
//     for(int i=0;i<M;++i){ // column
//       printf("%e ", A[j+i*M]);
//       if(i == M-1) printf("\n");
//     }
//   }
// }

// I would like to take V out of the parameters

__global__ void deviceMGSNoNv0(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat *V, dfloat *ans, dfloat *ans2){

  
  // loop over matrices
  //  for(int n=0;n<N;++n)
  int n = blockIdx.x;
  int j = threadIdx.x;

  int i, k, j2;

  //dfloat *V = (dfloat*) malloc(N*M*M*sizeof(dfloat));
  //__shared__ dfloat s_V[sM][sM];
  // // pointer to nth A and LU matrix storage
  // const dfloat *An = A+n*M*M;
  // dfloat *LUn = LU+n*M*M;

  // i will be column
  // j will be row
  
  // note we assume column major storage (for coalescing)
  for (i = 0; i < M; ++i) {
    V[j + M*i] = A[j + M*i];
  }

  // loop over columns
  //if(j<M){
  for (i = 0; i < M; ++i) {
    __syncthreads();
    //printf("3.55%e\n", V[j + M*i]);
    dfloat an = V[j + M*i]*V[j + M*i];
    //printf("%e\n", an);
    __syncthreads();
    // an uninterruptible increment
    atomicAdd(ans, an);
    //printf("3.5%e\n", ans[0]);

    //printf("i:%d, j:%d, %e\n", i, j, sqrt(ans[0]));
    R[i + M*i] = sqrt(ans[0]);
    //printf("4%e\n",sqrt(ans[0]));
    Q[j + M*i] = V[j + M*i]/R[i + M*i];
    //printf("5%e\n",R[i + M*i]);
    __syncthreads();
    //loop over rows starting from diagonal
          //for (j = i + 1; j < M; ++j)
    
    //R[i + M*j] += Q[j2 + M*i]*V[j2 + M*j];
    __syncthreads();
    // dfloat tmp;
    // tmp = V[j + M*j]-R[i + M*j]*Q[j + M*i];
    
    // dfloat an2 = Q[j + M*i]*V[j + M*i];
    // tmp = atomicAdd(ans2, an2);

    // if (j>i){
    //   R[i + M*j] = ans2[0];
    // }
    //printf("%e\n",Q[j + M*i]);
    __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      dfloat an2 = Q[j + M*i]*V[j + M*j2];
      //printf("1%e\n",an2);
      atomicAdd(ans2, an2);
      R[i + M*j2] = ans2[0];
    }
      __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      printf("2i:%d, j:%d, %e\n", i, j, ans2[0]);
      V[j + M*j2] = V[j + M*j2]-R[i + M*j2]*Q[j + M*i];
      printf("3i:%d, j:%d, %e\n", i, j, V[j + M*j2] );
    }

  }//}
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
  
  dfloat *h_A  = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *h_Q = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *h_R = (dfloat*) calloc(N*M*M, sizeof(dfloat));

  for(int n=0;n<N;++n){
    dfloat *A = h_A + n*M*M;
    for(int j=0;j<M;++j){ // row
      for(int i=0;i<M;++i){ // column
	      A[j+i*M] = j+(i*M)+1;
        if(i==2 & j==2){
          A[j+i*M] = 10;
        }
      }
    }
  }
  A[2+2*M]=10;

  printf("A:\n");
  printMatrix(h_A, M);

  dfloat tol = 1e-14;
  hostMGSnoN(M, h_A, Q, R, tol);

  printf("Host MGS\n-----------\n");
  printf("Q:\n");
  printMatrix(Q, M);
  printf("R:\n");
  printMatrix(R, M);

  dfloat *c_A, *c_Q, *c_R, *c_V;
  dfloat *h_gpuQ = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *h_gpuR = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  cudaMalloc(&c_A, N*M*M*sizeof(dfloat));
  cudaMalloc(&c_Q, N*M*M*sizeof(dfloat));
  cudaMalloc(&c_R, N*M*M*sizeof(dfloat));
  cudaMalloc(&c_V, N*M*M*sizeof(dfloat));

  cudaMemcpy(c_A, h_A, N*M*M*sizeof(dfloat), cudaMemcpyHostToDevice);
  dfloat *ans, *ans2;
  cudaMalloc(&ans, N*M*M*sizeof(dfloat));
  cudaMalloc(&ans2, N*M*M*sizeof(dfloat));
  deviceMGSNoNv0<<<N,M>>>(N, M, c_A, c_Q, c_R, c_V, ans, ans2);
  cudaGetLastError();
  cudaMemcpy(h_gpuQ, c_Q, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gpuR, c_R, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
  
  printf("Host gpuMGS\n-----------\n");
  printf("Q:\n");
  printMatrix(h_gpuQ, M);
  printf("R:\n");
  printMatrix(h_gpuR, M);
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