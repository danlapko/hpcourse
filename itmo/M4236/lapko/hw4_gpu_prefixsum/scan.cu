// for compile run: nvcc -o scan scan.cu
#include <iostream>
#include <fstream>

#define BLOCK_SIZE 256

using namespace std;

__global__
void scan(int length, float *arr) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int pos = bid * BLOCK_SIZE + tid;

    __shared__ float a[BLOCK_SIZE]; //

    if (pos >= length)
        return;

    a[tid] = arr[pos];

    __syncthreads();

    for (int shift = 1; shift < BLOCK_SIZE; shift <<= 1) {
        float cur = a[tid];
        float prev = 0;
        if (tid - shift >= 0)
            prev = a[tid - shift];

        __syncthreads();

        a[tid] = cur + prev;

        __syncthreads();
    }

    arr[pos] = a[tid];
    if (tid == BLOCK_SIZE - 1)
        arr[length + bid] = a[tid];
}

__global__
void sum(int length, float *d_arr) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int pos = bid * BLOCK_SIZE + tid;

    float *deltas = d_arr + length;

    if (pos >= length || bid == 0)
        return;

    d_arr[pos] += deltas[bid - 1];
}

void solve_recursivly(int length, float *d_arr) {
    if (length == 1)
        return;

    int n_blocks = (length + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    scan << < n_blocks, BLOCK_SIZE >> > (length, d_arr);

    solve_recursivly(n_blocks, d_arr + length);
    sum << < n_blocks, BLOCK_SIZE >> > (length, d_arr);

}

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int N;
    float *h_arr;
    float *d_arr;

    fin >> N;

    // ------- memory allocation --------

    h_arr = (float *) malloc(2 * N * sizeof(float));

    cudaMalloc(&d_arr, 2 * N * sizeof(float));

    // ------- read arrays to host --------

    for (int i = 0; i < N; ++i) {
        fin >> h_arr[i];
        h_arr[i + N] = 0;
    }

    // ------- copy arrays to device --------

    cudaMemcpy(d_arr, h_arr, 2 * N * sizeof(float), cudaMemcpyHostToDevice);

    // ------- execute kernel ---------
    solve_recursivly(N, d_arr);

    // ------- copy arrays to device --------
    cudaMemcpy(h_arr, d_arr, 2 * N * sizeof(float), cudaMemcpyDeviceToHost);

    // ------- print out -------
    for (int i = 0; i < N; ++i)
        fout << h_arr[i] << " ";

    // ------- deallocate memory -----
    cudaFree(d_arr);
    free(h_arr);

    fin.close();
    fout.close();
}
