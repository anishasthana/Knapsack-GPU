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

// TODO
__global__ void Knapsack_Kernel(
    const int i,
    const int current_val,
    const int current_weight,
    float *__restrict__ DP_old,
    float *__restrict__ DP_new,
    int *__restrict__ Path,
    const int capacity) {
    const int offset = threadIdx.x + blockIdx.x * blockDim.x;

    if (offset >= capacity)
        return;

    const float v1 = (offset >= current_weight) ? (DP_old[(offset - current_weight)] + current_val) : -1;
    const float v0 = DP_old[offset];

    float max_val = (v1 >= 0 && v1 > v0) ? v1 : v0;
    int chosen = (v1 >= 0 && v1 > v0) ? 1 : 0;

    atomicExch(&DP_new[offset], max_val);
    atomicOr(&Path[offset], chosen);
}

int main(int argc, char **argv) {
    // Select device
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // GPU Timing variables
    cudaEvent_t startGPU, stopGPU;
    // + 1 for 0th rows 
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
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&device_DP, dp_arr_size, cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&device_chosen, dp_arr_size, cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&device_values, values_arr_size, cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc((void **)&device_weights, values_arr_size, cudaHostAllocDefault));

    // Transfer the 2d-arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(device_values, host_values, N*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(device_weights, host_weights, N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(device_DP, 0.0, dp_arr_size));
    CUDA_SAFE_CALL(cudaMemset(device_chosen, 0.0, dp_arr_size));
   
    CudaTimerStart(&startGPU, &stopGPU);

	dim3 dimGrid((((W+1)+THREADS-1)/THREADS),1,1);
    dim3 dimBlock(THREADS, 1, 1);

    for(int i = 1; i<N; i++) {        
        Knapsack_Kernel<<<dimGrid,dimBlock>>>(
            i, host_values[i - 1], host_weights[i - 1], (float *)(&device_DP[(i - 1) * W]),
            (float *)(&device_DP[i * W]), (int *)(&device_chosen[i * W]), W);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    printf("GPU DONE\n");
    fflush(stdout);
    
	// Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    float *device_result = (float*)malloc(dp_arr_size);
    printf("device_result malloc'd\n");
    fflush(stdout);
    CUDA_SAFE_CALL(cudaMemcpy(device_result, device_DP, dp_arr_size, cudaMemcpyDeviceToHost));
    printf("device_result cuda memcopy done\n");
    fflush(stdout);
    printf("GPU Result %f\n", device_result[N*W - 1]);  // Segfault here...   
    fflush(stdout);
    CudaTimerStop(&startGPU, &stopGPU);
    // Transfer the results back to the host
    //CUDA_SAFE_CALL(cudaMemcpy(host_deviceResCopy, device_res, allocSize2D, cudaMemcpyDeviceToHost));
    

    // **************** CPU BASELINE **************************************
    // Calculate time
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    // Compute on CPU
    hostKnapsack(host_weights, host_values, host_DP, host_chosen);
    gettimeofday(&t2, 0);
    double total_cpu_time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to generate:  %3.1f ms \n", total_cpu_time);


    printf("Result %f\n", host_DP[N*(W-5)]);

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
