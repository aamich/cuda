// test cuda programming
// nvcc myAddVec.cu -o myAddVec

#include <iostream>
#include <vector>
#include <assert.h>

using namespace std;

__global__ void addVec(int* da, int* db, int* dc, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n) {
        dc[tid] = da[tid] + db[tid];
    }
}

int main() {
    cout << "Demo: CUDA add vector" << endl;
    const int n = 1000;
    size_t bytes = n*sizeof(int);
    vector<int> a = vector<int>(n, 1);
    vector<int> b = vector<int>(n, 2);
    vector<int> c = vector<int>(n, 0);

    int* da;
    int* db;
    int* dc;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice);
    err = cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice);

    int BlockSize = 256;
    int GridSize = (n + BlockSize - 1)/BlockSize;
    cout << "GridSize=" << GridSize << endl;
    cout << "BlockSize=" << BlockSize << endl;
    addVec<<<GridSize, BlockSize>>>(da, db, dc, n);

    cudaDeviceSynchronize();
    err = cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost);
    if(err == cudaSuccess)
        cout << "cudaMemcpyDeviceToHost ok." << endl;
    else
        cout << err << " cudaMemcpyDeviceToHost failed." << endl;
  
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    cout << "c[0]:" << c[0] << endl;
    cout << "c[100]:" << c[100] << endl;
    assert(c[0] == 3);
    assert(c[500] == 3);
    cout << "CUDA add vector successfully!" << endl;
}
                                                              
