// for compile run: nvcc -o conv conv.cu
#include <iostream>
#include <fstream>

using namespace std;

__global__
void conv(int N, int M, float *a, float *b, float *c) {
    int pad = M / 2;
    int Npad = pad + N + pad;

    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= N * N)
        return;

    int x = pad + pos / N;
    int y = pad + pos % N;

    c[pos] = 0;

    for (int i = -pad; i <= pad; ++i)
        for (int j = -pad; j <= pad; ++j)
            c[pos] += a[(x + i) * Npad + (y + j)] * b[(i + pad) * M + (j + pad)];

}

int main() {
    ifstream fin("input.txt");
    ofstream fout("output.txt");

    int N, M;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    fin >> N >> M;

    int pad = M / 2;
    int Npad = N + 2 * pad;
    // ------- memory allocation --------

    h_a = (float *) malloc(Npad * Npad * sizeof(float));
    h_b = (float *) malloc(M * M * sizeof(float));
    h_c = (float *) malloc(N * N * sizeof(float));

    cudaMalloc(&d_a, Npad * Npad * sizeof(float));
    cudaMalloc(&d_b, M * M * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));


    // ------- read arrays to host --------

    for (int i = 0; i < Npad; ++i)
        for (int j = 0; j < Npad; ++j) {
            if (i < pad || i >= (pad + N) || j < pad || j >= (pad + N))
                h_a[i * Npad + j] = 0;
            else
                fin >> h_a[i * Npad + j];
        }


    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            fin >> h_b[i * M + j];

//    for (int i = 0; i < Npad; ++i) {
//        for (int j = 0; j < Npad; ++j)
//            cout << h_a[i * Npad + j] << " ";
//        cout << "\n";
//    }

    // ------- copy arrays to device --------

    cudaMemcpy(d_a, h_a, Npad * Npad * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, M * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * N * sizeof(float), cudaMemcpyHostToDevice);


    // ------- execute kernel ---------
    conv << < (N * N + 255) / 256, 256 >> > (N, M, d_a, d_b, d_c);

    // ------- copy arrays to device --------
    cudaMemcpy(h_c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // ------- print out -------
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            fout << h_c[i * N + j] << " ";
        fout << "\n";
    }

    // ------- deallocate memory -----
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    fin.close();
    fout.close();
}
