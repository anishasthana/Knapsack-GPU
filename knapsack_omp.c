#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define THREADS 	256

// Knapsack parameters
#define N   100
#define W   500000


void initializeZerosFirstRow(float *arr) {
    int i;
    for(i =0; i<W; i++) {
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
        #pragma omp parallel for
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


int main(int argc, char **argv) {

    // + 1 for 0th rows 
    int dp_arr_size = N*W*sizeof(float);
    int chosen_arr_size = N*W*sizeof(int);
    int values_arr_size = N*sizeof(float);
    
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
    
    printf("CPU Time: %3.1f ms\n", total_cpu_time);
    printf("CPU Result %f\n", host_DP[N*(W-5)]);
	// Free-up device and host memory
    free(host_weights);
    free(host_values);
    free(host_chosen);
    free(host_DP);
	return 0;
}
