#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <omp.h>

// sets type of matrices
#define dfloat double

__host__ __device__ void printMatrix(dfloat *A, int M){
  printf("[");
  for(int j=0;j<M;++j){ // row
    for(int i=0;i<M;++i){ // column
      printf("%17.15e ", A[j+i*M]);
      if(i == M-1) 
        if (j == M-1) printf("]\n");
        else printf(";\n");
    }
  }
}

void hostMGS(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat tol){

  for(int n=0;n<N;++n){
    int i, j, j2;

    const dfloat *An = A+n*M*M;
    dfloat *Qn = Q+n*M*M;
    dfloat *Rn = R+n*M*M;

    // i will be column
    // j will be row
    dfloat *V  = (dfloat*) calloc(M*M, sizeof(dfloat));
    dfloat *Vn = V+n*M*M;

    // column major storage
    for (j = 0; j < M; ++j) {
      for (i = 0; i < M; ++i) {
        Vn[j + M*i] = An[j + M*i];
      }
    }
    // loop over columns
    for (i = 0; i < M; ++i){
      dfloat tmp = 0;
      for (j = 0; j<M; ++j){
        tmp += Vn[j + M*i]*Vn[j + M*i];
      }
      
      R[i + M*i] = sqrt(tmp);
      for (j = 0; j<M; ++j){
        Qn[j + M*i] = Vn[j + M*i]/Rn[i + M*i];
      }

      for (j2 = 0; j2 < M; ++j2){ 
        for (j = i + 1; j < M; ++j){
          Rn[i + M*j] += Qn[j2 + M*i]*Vn[j2 + M*j];
        }
      }
      for (j2 = 0; j2 < M; ++j2){
        for (j = i + 1; j < M; ++j){
          Vn[j2 + M*j] = Vn[j2 + M*j]-Rn[i + M*j]*Qn[j2 + M*i];
        }
      }
      
    }
  }
}

__global__ void deviceMGSv0(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat *V, dfloat *ans, dfloat *ans2){
  
  // loop over matrices
  int n = blockIdx.x;
  int j = threadIdx.x;

  int i, j2;

  const dfloat *An = A+n*M*M;
  dfloat *Qn = Q+n*M*M;
  dfloat *Rn = R+n*M*M;
  dfloat *Vn = V+n*M*M;

  // i will be column
  // j will be row
  
  // note we assume column major storage (for coalescing)
  for (i = 0; i < M; ++i) {
    Vn[j + M*i] = An[j + M*i];
  }

  // loop over columns
  for (i = 0; i < M; ++i) {
    __syncthreads();
    if(j==0) ans[0] = 0;
    if(j==0) ans2[0] = 0;
    __syncthreads();

    dfloat an = Vn[j + M*i]*Vn[j + M*i];
    // an uninterruptible increment
    atomicAdd(ans, an);
    __syncthreads();

    if(j==0)
    Rn[i + M*i] = sqrt(ans[0]);

    Qn[j + M*i] = Vn[j + M*i]/Rn[i + M*i];
  
    __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      dfloat an2 = Qn[j + M*i]*Vn[j + M*j2];
    
      __syncthreads();
      if(j==0) ans2[0] = 0;
      __syncthreads();

      atomicAdd(ans2, an2); // will change to warp or BT reduction
      __syncthreads();
      if(j==0)
      Rn[i + M*j2] = ans2[0];
    }
      __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      Vn[j + M*j2] = Vn[j + M*j2]-Rn[i + M*j2]*Qn[j + M*i];
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
  
  dfloat *h_A  = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *h_Q = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *h_R = (dfloat*) calloc(N*M*M, sizeof(dfloat));

  for(int n=0;n<N;++n){
    dfloat *An = h_A + n*M*M;
    for(int j=0;j<M;++j){ // row
      for(int i=0;i<M;++i){ // column
	      An[j+i*M] = drand48();
      }
    }
  }

  printf("A:\n");
  printMatrix(h_A, M);

  dfloat tol = 1e-14;
  hostMGS(N, M, h_A, h_Q, h_R, tol);

  printf("Host MGS\n-----------\n");
  printf("Q:\n");
  printMatrix(h_Q, M);
  printf("R:\n");
  printMatrix(h_R, M);

  dfloat *c_A, *c_Q, *c_R, *c_V;
  dfloat *h_gpuQ = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *h_gpuR = (dfloat*) calloc(N*M*M, sizeof(dfloat));

  cudaMalloc(&c_A, N*M*M*sizeof(dfloat));
  cudaMalloc(&c_Q, N*M*M*sizeof(dfloat));
  cudaMalloc(&c_R, N*M*M*sizeof(dfloat));
  // won't need temporary V matrix in global memory in next version:
  cudaMalloc(&c_V, N*M*M*sizeof(dfloat));

  cudaMemcpy(c_A, h_A, N*M*M*sizeof(dfloat), cudaMemcpyHostToDevice);

  // will be avoided in next version with shared memory:
  dfloat *ans, *ans2;
  dfloat *h_ans = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  dfloat *h_ans2 = (dfloat*) calloc(N*M*M, sizeof(dfloat));
  cudaMalloc(&ans, N*M*M*sizeof(dfloat));
  cudaMalloc(&ans2, N*M*M*sizeof(dfloat));
  cudaMemcpy(ans, h_ans, N*M*M*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(ans2, h_ans2, N*M*M*sizeof(dfloat), cudaMemcpyHostToDevice);

  deviceMGSv0<<<N,M>>>(N, M, c_A, c_Q, c_R, c_V, ans, ans2);
  cudaGetLastError();
  
  cudaMemcpy(h_gpuQ, c_Q, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gpuR, c_R, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
  
  printf("Device MGS\n-----------\n");
  printf("Q:\n");
  printMatrix(h_gpuQ, M);
  printf("R:\n");
  printMatrix(h_gpuR, M);

  dfloat maxDiff =0;
  for(int n=0;n<N*M*M;++n){
    dfloat diff = fabs(h_gpuQ[n]-h_Q[n]);
    maxDiff = (maxDiff>diff) ? maxDiff:diff;
  }
  printf("Q diff: %3.2e \n", maxDiff);

  maxDiff =0;
  for(int n=0;n<N*M*M;++n){
    dfloat diff = fabs(h_gpuR[n]-h_R[n]);
    maxDiff = (maxDiff>diff) ? maxDiff:diff;
  }
  printf("R diff: %3.2e \n", maxDiff);
  
  return 0;
}