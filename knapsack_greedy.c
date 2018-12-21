#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#define THREADS     256

// Knapsack parameters
#define N   100
#define W   500000

typedef struct item item;

struct item{
    float ratio;
    float value;
    int weight;
};

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

void initializeRatios(float *ratio, float *value, int *weight){
    int i;
    for (i=0; i < N; i++) {
        ratio[i] = (float)value[i]/(float)weight[i];
    }
}

void initializeStruct(float *ratio, float *value, int *weight, item *items){
    int i;
    for (i=0; i < N; i++) {
        item a = {.ratio = ratio[i], .value = value[i], .weight = weight[i]};
        items[i] = a;

    }
}

int comparator(const void *p, const void *q) 
{ 
    float l = ((struct item *)p)->ratio; 
    float r = ((struct item *)q)->ratio; 
    return ((int)(l - r)); 
} 

void hostKnapsack(item* items, int sum, int worth) {
    int i;

    for (i = N-1; i >= 0; i--) {
        if (sum+items[i].weight < W){
            sum += items[i].weight;
            worth += items[i].value;
        }
    }
}


int main(int argc, char **argv) {

    // + 1 for 0th rows 
    int dp_arr_size = N*W*sizeof(float);
    int chosen_arr_size = N*W*sizeof(int);
    int values_arr_size = N*sizeof(float);
    int sum = 0;
    int worth = 0;
    
    // 2D arrays on host memory
    float *host_values, *host_DP, *host_ratio;
    int *host_weights, *host_chosen;
    item *host_items;
    host_weights = (int *) malloc(values_arr_size);
    host_chosen = (int *)malloc(chosen_arr_size);
    host_values = (float *) malloc(values_arr_size);
    host_DP = (float *)malloc(dp_arr_size);
    host_ratio = (float *)malloc(values_arr_size);
    host_items = (item *)malloc(values_arr_size);
    
    // Initialize the arrays on CPU
    initializeValues(host_values, 1251);
    initializeWeights(host_weights, 1251);
    initializeZerosFirstRow(host_DP); // Marks the entire first row as zeros
    initializeRatios(host_ratio,host_values,host_weights);
    initializeStruct(host_ratio,host_values,host_weights,host_items);
    //qsort(host_items, values_arr_size, sizeof(host_items[0]), comparator);

    // Transfer the results back to the host
    //CUDA_SAFE_CALL(cudaMemcpy(host_deviceResCopy, device_res, allocSize2D, cudaMemcpyDeviceToHost));

    // **************** CPU BASELINE **************************************
    // Calculate time
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    // Compute on CPU
    hostKnapsack(host_items, sum, worth);
    gettimeofday(&t2, 0);
    double total_cpu_time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    
    printf("CPU Time: %3.1f ms\n", total_cpu_time);
    printf("CPU Result %f\n", host_DP[N*(W-5)]);
    // Free-up device and host memory
    return 0;
}
