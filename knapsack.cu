#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define THREADS 	256

// Knapsack parameters 
#define N   100 
#define W   500000


void CudaTimerStart(cudaEvent_t* startGPU, cudaEvent_t* stopGPU) {
	// Create the cuda events
	cudaEventCreate(startGPU);
	cudaEventCreate(stopGPU);
	// Record event on the default stream
	cudaEventRecord(*startGPU, 0);
}

void CudaTimerStop(cudaEvent_t* startGPU, cudaEvent_t *stopGPU) {
    // Stop and destroy the timer
    float elapsed_gpu = 0.0;
	cudaEventRecord(*stopGPU,0);
	cudaEventSynchronize(*stopGPU);
	cudaEventElapsedTime(&elapsed_gpu, *startGPU, *stopGPU);
	cudaEventDestroy(*startGPU);
    cudaEventDestroy(*stopGPU);
    printf("\nGPU time: %f (msec)\n", elapsed_gpu);
}

void initializeZerosFirstRow(float *arr) {
    for(int i =0; i<W; i++) {
        arr[i]= 0.0;
    }
}

void initializeValues(float *arr, int seed) {
	int i;
	srand(seed);
	for (i = 0; i < N; i++) {
        arr[i] = (float) (rand()%1000);
	}
}

void initializeWeights(int *arr, int seed) {
	int i;	
	srand(seed);
	for (i = 0; i < N; i++) {        
        arr[i] = (int) (rand()%20000);;
	}
}

void hostKnapsack(int *w, float* v, float *m, int *chosen) {
    int i, j;
    float with = 0, without = 0;

    for (i = 1; i < N; i++) {
        for (j = 1; j < W; j++) {
            if(j < w[i-1]) {
                // Skip
                m[i*W + j] = m[(i-1)*W + j]; 
                chosen[i*W + j] = 0;
            } else {
                // Should I take it or not
                without = m[(i-1)*W+j];
                with = m[(i-1)*W+(j-w[i-1])]+v[i-1];
                if(without >= with) {
                    m[i*W + j] = without;
                    chosen[i*W + j] = 0;
                } else {
                    m[i*W + j] = with;
                    chosen[i*W+j] = 1;
                }
            }
        }
    }
}

__global__ void GPU_knap(
    const int i,
    const int curv,
    const int curw,
    const int*__restrict__ DP_old,
    int* __restrict__ DP_new,
    int* __restrict__ Path,
    const int capacity) {

    const int offset = threadIdx.x+blockIdx.x*blockDim.x;
    const int which_warp = threadIdx.x>>5;
    const int tid_in_warp = threadIdx.x%32;

    __shared__ int s_cache[9][33];//do not really need to use __shared__ memory for this but did it anyway
    __shared__ int s_or[9][33];//ditto

    if(offset >= capacity) {
        return;
    }

    const int v1 = (offset >= curw) ? (DP_old[(offset-curw)] + curv) : -1;
    const int v0 = DP_old[offset];

    s_cache[which_warp][tid_in_warp] = (v1>=0 && v1>v0) ? v1 : v0;
    s_or[which_warp][tid_in_warp] = (v1>=0 && v1>v0) ? 1 : 0;
    __syncthreads();

    atomicExch(&DP_new[offset],s_cache[which_warp][tid_in_warp]);
    __syncthreads();

    if(s_or[which_warp][tid_in_warp] > 0) {
        atomicOr(&Path[offset],1);
    }
}

int main(int argc, char **argv) {
    // Select device
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // GPU Timing variables
    cudaEvent_t startGPU, stopGPU;

    int dp_arr_size = N*W*sizeof(float);
    int chosen_arr_size = N*W*sizeof(int);
    int values_arr_size = N*sizeof(float);

    // Arrays on GPU global memory
    float *device_values, *device_DP;
	int *device_weights, *device_chosen;
    CUDA_SAFE_CALL(cudaMalloc((void **)&device_weights, values_arr_size));
	CUDA_SAFE_CALL(cudaMalloc((void **)&device_values, values_arr_size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&device_DP, dp_arr_size));
    CUDA_SAFE_CALL(cudaMalloc((void **)&device_chosen, chosen_arr_size));
    
    // 2D arrays on host memory
    float *host_values, *host_DP;
    int *host_weights, *host_chosen;
    host_weights = (int *) malloc(values_arr_size);
    host_chosen = (int *)malloc(chosen_arr_size);
	host_values = (float *) malloc(values_arr_size);
	host_DP = (float *)malloc(dp_arr_size);
    
    // Initialize the arrays on CPU
    initializeValues(host_values, 1251);
    initializeWeights(host_weights, 1251);
    initializeZerosFirstRow(host_DP); // Marks the entire first row as zeros
    
    // Transfer the 2d-arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(device_values, host_values, N*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_weights, host_weights, N*sizeof(int), cudaMemcpyHostToDevice));

    CudaTimerStart(&startGPU, &stopGPU);

	dim3 dimGrid(((W+THREADS-1)/THREADS),1,1);
    dim3 dimBlock((THREADS, 1, 1);

    for(; ii<N; ii++) {
        GPU_knap<<<dimGrid,dimBlock>>>(
            ii,
            device_values[ii-1],
            device_weights[ii-1],
            (int*)(&d_DP[(ii-1)*W]),
            (int*)(&d_DP[ii*W]),
            (int*)(&d_Path[ii*W]),
            W);       
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

	// Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    float *device_result = malloc(N*W*sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(device_result, device_DP, N*W*sizeof(float)));
    printf("GPU Result %f\n", device_result[(N*W) + W - 1]);    

    // Transfer the results back to the host
    //CUDA_SAFE_CALL(cudaMemcpy(host_deviceResCopy, device_res, allocSize2D, cudaMemcpyDeviceToHost));
    CudaTimerStop(&startGPU, &stopGPU);

    // **************** CPU BASELINE **************************************
    // Calculate time
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    // Compute on CPU
    hostKnapsack(host_weights, host_values, host_DP, host_chosen);
    gettimeofday(&t2, 0);
    double total_cpu_time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to generate:  %3.1f ms \n", total_cpu_time);


    printf("Result %f\n", host_DP[(N*W) + W - 1]);

	// Free-up device and host memory
    CUDA_SAFE_CALL(cudaFree(device_weights));
    CUDA_SAFE_CALL(cudaFree(device_values));
    CUDA_SAFE_CALL(cudaFree(device_chosen));
    CUDA_SAFE_CALL(cudaFree(device_DP));
    free(host_weights);
    free(host_values);
    free(host_chosen);
    free(host_DP);
	return 0;
}
