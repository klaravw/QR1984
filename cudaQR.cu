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

void hostQR(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat *V){

  for(int n=0;n<N;++n){
    int i, j, j2;

    const dfloat *An = A+n*M*M;
    dfloat *Qn = Q+n*M*M;
    dfloat *Rn = R+n*M*M;
    dfloat *Vn = V+n*M*M;

    // i will be column
    // j will be row

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

      // 2-norm of V column
      Rn[i + M*i] = sqrt(tmp);
      // normalize q
      for (j = 0; j<M; ++j){
        Qn[j + M*i] = Vn[j + M*i]/Rn[i + M*i];
      }

      // projection steps
      for (j2 = 0; j2 < M; ++j2){ 
        for (j = i + 1; j < M; ++j){
          Rn[i + M*j] += Qn[j2 + M*i]*Vn[j2 + M*j];
        }
      }
      for (j2 = 0; j2 < M; ++j2){
        // start from diagonal
        for (j = i + 1; j < M; ++j){
          Vn[j2 + M*j] = Vn[j2 + M*j]-Rn[i + M*j]*Qn[j2 + M*i];
        }
      } 
    }
  }
}

// no shared with atomics
__global__ void deviceQRv0(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat *V, dfloat *ans, dfloat *ans2){
  
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

// uses shared and atomics
template < int sM >
__global__ void deviceQRv1(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat *ans, dfloat *ans2){
  
  // loop over matrices
  int n = blockIdx.x;
  int j = threadIdx.x;

  int i, j2;

  __shared__ dfloat s_Qn[sM][sM];
  __shared__ dfloat s_Rn[sM][sM];
  __shared__ dfloat s_Vn[sM][sM];

  const dfloat *An = A+n*M*M;
  dfloat *Qn = Q+n*M*M;
  dfloat *Rn = R+n*M*M;

  // i will be column
  // j will be row
  
  // note we assume column major storage (for coalescing)
  for (i = 0; i < M; ++i) {
    s_Vn[j][i] = An[j + M*i];
  }

  // loop over columns
  for (i = 0; i < M; ++i) {
    __syncthreads();
    if(j==0) ans[0] = 0;
    if(j==0) ans2[0] = 0;
    __syncthreads();

    dfloat an = s_Vn[j][i]*s_Vn[j][i];
    // an uninterruptible increment
    atomicAdd(ans, an);
    __syncthreads();

    if(j==0)
    s_Rn[i][i] = sqrt(ans[0]);
    __syncthreads();
    s_Qn[j][i] = s_Vn[j][i]/s_Rn[i][i];

    __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      dfloat an2 = s_Qn[j][i]*s_Vn[j][j2];
    
      __syncthreads();
      if(j==0) ans2[0] = 0;
      __syncthreads();

      atomicAdd(ans2, an2); // will change to warp or BT reduction
      __syncthreads();
      if(j==0)
      s_Rn[i][j2] = ans2[0];
    }
      __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      s_Vn[j][j2] = s_Vn[j][j2]-s_Rn[i][j2]*s_Qn[j][i];
    }
    Qn[j+M*i] = s_Qn[j][i];
    Rn[j+M*i] = s_Rn[j][i];
  }
}

template < int sM >
__global__ void deviceQRv2(int N, int M, const dfloat *A, dfloat *Q, dfloat *R, dfloat *ans, dfloat *ans2){
  
  // loop over matrices
  int n = blockIdx.x;
  int j = threadIdx.x;

  int i, j2;

  __shared__ dfloat s_Qn[sM][sM];
  __shared__ dfloat s_Rn[sM][sM];
  __shared__ dfloat s_Vn[sM][sM];
  __shared__ dfloat tmps_Vn[sM];

  const dfloat *An = A+n*M*M;
  dfloat *Qn = Q+n*M*M;
  dfloat *Rn = R+n*M*M;

  // i will be column
  // j will be row
  
  // note we assume column major storage (for coalescing)
  for (i = 0; i < M; ++i) {
    s_Vn[j][i] = An[j + M*i];
  }

  // loop over columns
  for (i = 0; i < M; ++i) {
    __syncthreads();
    if(j==0) ans[0] = 0;
    if(j==0) ans2[0] = 0;
    __syncthreads();

    tmps_Vn[j] = s_Vn[j][i]*s_Vn[j][i];
    // an uninterruptible increment
    int alive = sM;
    while(alive>1){
      __syncthreads();
      int oldAlive = alive;
      alive = (alive+1)/2;
      if (j<alive && j+alive<oldAlive){
        tmps_Vn[j] += s_Vn[j+alive];
      }
    }

    if(j==0)
    s_Rn[i][i] = sqrt(tmps_Vn[0]);
    __syncthreads();
    s_Qn[j][i] = s_Vn[j][i]/s_Rn[i][i];

    __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      dfloat an2 = s_Qn[j][i]*s_Vn[j][j2];
    
      __syncthreads();
      if(j==0) ans2[0] = 0;
      __syncthreads();

      atomicAdd(ans2, an2); // will change to warp or BT reduction
      __syncthreads();
      if(j==0)
      s_Rn[i][j2] = ans2[0];
    }
      __syncthreads();
    for (j2 = i + 1; j2 < M; ++j2){
      s_Vn[j][j2] = s_Vn[j][j2]-s_Rn[i][j2]*s_Qn[j][i];
    }
    Qn[j+M*i] = s_Qn[j][i];
    Rn[j+M*i] = s_Rn[j][i];
  }
}

template <int tM>
void runKernelQR(int op, int N, int M, const dfloat *c_a, dfloat *c_Q, dfloat *c_R, dfloat *c_V, dfloat *ans, dfloat *ans2){
  switch(op){
  case 0: deviceQRv0    <<<N,M>>> (N,M,c_a, c_Q, c_R, c_V, ans, ans2); break;
  case 1: deviceQRv1<tM><<<N,M>>> (N,M,c_a, c_Q, c_R, ans, ans2); break;
  //case 2: deviceQRv2<tM><<<N,M>>> (N,M,c_a, c_Q, c_R, ans, ans2); break;
  }
}

void runKernel(int op, int N, int M, const dfloat *c_A, dfloat *c_Q, dfloat *c_R, dfloat *c_V, dfloat *ans, dfloat *ans2){
  switch(M){
  case  1: runKernelQR< 1>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  2: runKernelQR< 2>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  3: runKernelQR< 3>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  4: runKernelQR< 4>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  5: runKernelQR< 5>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  6: runKernelQR< 6>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  7: runKernelQR< 7>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  8: runKernelQR< 8>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case  9: runKernelQR< 9>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 10: runKernelQR<10>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;
  case 11: runKernelQR<11>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 12: runKernelQR<12>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 13: runKernelQR<13>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 14: runKernelQR<14>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 15: runKernelQR<15>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 16: runKernelQR<16>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 17: runKernelQR<17>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 18: runKernelQR<18>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 19: runKernelQR<19>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;
  case 20: runKernelQR<20>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;
  case 21: runKernelQR<21>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 22: runKernelQR<22>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 23: runKernelQR<23>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 24: runKernelQR<24>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 25: runKernelQR<25>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 26: runKernelQR<26>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 27: runKernelQR<27>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 28: runKernelQR<28>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  case 29: runKernelQR<29>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;
  case 30: runKernelQR<30>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;
  case 31: runKernelQR<31>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;
  case 32: runKernelQR<32>(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2); break;	
  }
}

double runDeviceQR(int op, int N, int M, const dfloat *c_A, dfloat *c_Q, dfloat *c_R, dfloat *c_V, dfloat *ans, dfloat *ans2){


  cudaMemset(c_Q, 0, N*M*M*sizeof(dfloat));
  cudaMemset(c_R, 0, N*M*M*sizeof(dfloat));
  
  cudaDeviceSynchronize();

  cudaEvent_t tic, toc;
  cudaEventCreate(&tic);
  cudaEventCreate(&toc);

  cudaEventRecord(tic);
  // launched (possibly templated kernel)
  runKernel(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2);
  cudaEventRecord(toc);

  cudaDeviceSynchronize();
  
  float elapsed;
  cudaEventElapsedTime(&elapsed, tic, toc);
  elapsed /= 1000.;
  return elapsed;
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
  dfloat *h_V = (dfloat*) calloc(N*M*M, sizeof(dfloat));

  for(int n=0;n<N;++n){
    dfloat *An = h_A + n*M*M;
    for(int j=0;j<M;++j){ // row
      for(int i=0;i<M;++i){ // column
	      An[j+i*M] = drand48();
      }
    }
    //printMatrix(An,M);
  }

  // UNCOMMENT TO SEE MATRIX OUTPUT:
//   printf("A:\n");
//   printMatrix(h_A, M);

//   hostQR(N, M, h_A, h_Q, h_R);

//  // UNCOMMENT TO SEE MATRIX OUTPUT:
//   printf("Host QR\n-----------\n");
//   printf("Q:\n");
//   printMatrix(h_Q, M);
//   printf("R:\n");
//   printMatrix(h_R, M);

  dfloat elapsedTimes[1000];
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

  // deviceQRv1<3><<<N,M>>>(N, M, c_A, c_Q, c_R, ans, ans2);
  // cudaGetLastError();
  
  // cudaMemcpy(h_gpuQ, c_Q, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
  // cudaMemcpy(h_gpuR, c_R, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
  

  // // UNCOMMENT TO SEE MATRIX OUTPUT:
  // printf("Device QR\n-----------\n");
  // printf("Q:\n");
  // printMatrix(h_gpuQ, M);
  // printf("R:\n");
  // printMatrix(h_gpuR, M);

  // dfloat maxDiff =0;
  // for(int n=0;n<N*M*M;++n){
  //   dfloat diff = fabs(h_gpuQ[n]-h_Q[n]);
  //   maxDiff = (maxDiff>diff) ? maxDiff:diff;
  // }
  // printf("Q diff: %3.2e \n", maxDiff);

  // maxDiff =0;
  // for(int n=0;n<N*M*M;++n){
  //   dfloat diff = fabs(h_gpuR[n]-h_R[n]);
  //   maxDiff = (maxDiff>diff) ? maxDiff:diff;
  // }
  // printf("R diff: %3.2e \n", maxDiff);
  // -------------------------------------------------------
  // 1. test host
  double tic = omp_get_wtime();
  hostQR(N, M, h_A, h_Q, h_R, h_V);
  double toc = omp_get_wtime();
  double elapsed = toc-tic;
  dfloat bandwidth = (2.*N*M*M/(elapsed*1.e9))*sizeof(dfloat); // matrix bandwidth in GB/s (ignoring caching)
  
  // printf("host,    M:%02d, time:%3.2e s, diff:%3.2e, throughput:%3.2f GB/s\n",
	//   M, elapsed, 0., bandwidth);

  //deviceQRv2(N, M,  *A, *Q, *R,*ans,  *ans2);

  int Nops = 2;
  for(int op=0;op<Nops;++op){
    // warm up
    runDeviceQR(2, N, M, c_A, c_Q, c_R, c_V, ans, ans2);
    elapsedTimes[op] = runDeviceQR(op, N, M, c_A, c_Q, c_R, c_V, ans, ans2);
    
    cudaMemcpy(h_gpuQ, c_Q, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpuR, c_R, N*M*M*sizeof(dfloat), cudaMemcpyDeviceToHost);

    dfloat maxDiff =0;
    for(int n=0;n<N*M*M;++n){
      dfloat diff = 0;
      diff = fabs(h_gpuQ[n]-h_Q[n]);
      
      //printf("%e\n", h_gpuQ[n]);
      //printf("%e\n", h_Q[n]);
      maxDiff = (maxDiff>diff) ? maxDiff:diff;
    }
    // printMatrix(h_Q,3);
    // printMatrix(h_gpuQ,3);
    // printf("Q diff: %3.2e\n", maxDiff);

    maxDiff = 0;
    for(int n=0;n<N*M*M;++n){
      dfloat diff = 0;
      diff = fabs(h_gpuR[n]-h_R[n]);
      //printf("%e\n",h_R[n]);
      maxDiff = (maxDiff>diff) ? maxDiff:diff;
    }
    //printf("R diff: %3.2e\n", maxDiff);
    // matrix bandwidth in GB/s (ignoring caching)
    dfloat bandwidthOp = (2.*N*M*M/(elapsedTimes[op]*1.e9))*sizeof(dfloat); 
    
    printf("kernel:%02d, M:%02d, time:%3.2e s, diff:%3.2e, throughput:%3.2f GB/s\n",
	   op, M, elapsedTimes[op], maxDiff, bandwidthOp);

     //printf("%02d %3.2e;\n", M,elapsedTimes[op]);
     //printf("%02d %3.2f;\n", M,bandwidthOp);
  }

  
  return 0;
}