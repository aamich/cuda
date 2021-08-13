// nvcc -std=c++11 myMatMul.cu -o myMatMul

#include <vector>

using namespace std;

__global__ void mulMat(float* a, float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = row*n + col;
    if(tid < n*n) {
        for(int k = 0; k < n; ++k) {
            c[tid] += a[row*n + k]*b[col + n*k];
            //c[tid] += a[row*n + k]*b[n*col + k]; 
        }
    }
}

int main() {
    const int n = 1<<13;
    auto vv = vector<vector<float>>(n, vector<float>(n, sqrt(10)));

    vector<float> v1;
    for(int i = 0; i < n; ++i)
        v1.insert(v1.end(), vv[i].begin(), vv[i].end());

    vector<float> c = v1;

    float* da;
    float* db;
    float* dc;

    size_t bytes = n*n*sizeof(float);
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    int BlockSize = 32;
    int GridSize = (n + BlockSize - 1)/BlockSize;
    cout << "BlockSize_1d:" << BlockSize <<" GridSize_1d:" << GridSize << endl;

    dim3 BS(BlockSize, BlockSize);
    dim3 GS(GridSize, GridSize);

    cudaMemcpy(da, v1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, v1.data(), bytes, cudaMemcpyHostToDevice);
    mulMat<<<GS, BS>>>(da, da, dc, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    cout << c[0] << endl;
    cout << c[n*n/2] << endl;
}
