#include <stdio.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#define BLOCK_SIZE 256
#define GRID_SIZE 648

__global__
void zeroOutQsgdDecomKernel(float* d_Output, int* d_LayerSizePrefixSum, int* d_BitmapInput, int* d_QuanInput, float* d_LayersMinMaxInput, int bins, int size, int layerNum, int qsgdLoopTimes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int shraedCurrentLayerIndex;
    __shared__ int sharedThreadsNumToNextLayer;
    extern __shared__ int sharedLayersSizePrefixSum[];

    // load layer size and block numbers into shared memory
    int count = 0;
    int sharedId = count * BLOCK_SIZE + threadIdx.x;
    while(sharedId < layerNum) {
        int globalId = count * BLOCK_SIZE + threadIdx.x;

        sharedLayersSizePrefixSum[sharedId] = d_LayerSizePrefixSum[globalId];

        count++;
        sharedId = count * BLOCK_SIZE + threadIdx.x;
    }

    __syncthreads();

    

    for(int i = 0; i < qsgdLoopTimes; i++) {
        int blockFirstGid = blockIdx.x * blockDim.x + i * gridDim.x * blockDim.x;
        int currentGid = tid + i * gridDim.x * blockDim.x;

        if(blockFirstGid < size) {
            int bitGIndex = currentGid / 32;
            int bitLIndex = (currentGid - (currentGid / 32) * 32);
            int bitmapData = d_BitmapInput[bitGIndex];
            int zeroOutBitRes = (0x80000000 >> bitLIndex) & bitmapData;

            if(threadIdx.x == 0) {
                int left = 0, right = layerNum - 1;
                int mid = 0;
                while(left < right) {
                    mid = left + (right - left) / 2;
                    int tempLayersSize = sharedLayersSizePrefixSum[mid];
                    if(blockFirstGid + 1 < tempLayersSize) {
                        if(mid == 0 || blockFirstGid + 1 > sharedLayersSizePrefixSum[mid - 1]) {
                            break;
                        }else {
                            right = mid - 1;
                        }
                    }else if(blockFirstGid + 1 > tempLayersSize) {
                        if(mid == layerNum - 1 || blockFirstGid + 1 <= sharedLayersSizePrefixSum[mid + 1]) {
                            mid = mid + 1;
                            break;
                        }else {
                            left = mid + 1;
                        }
                    }else {
                        break;
                    }
                }

                shraedCurrentLayerIndex = mid;
                sharedThreadsNumToNextLayer = sharedLayersSizePrefixSum[mid] - blockFirstGid;
            }

            __syncthreads();

            if(zeroOutBitRes == 0x00000000) {
                d_Output[currentGid] = 0.0f;
            }else {
                if(currentGid < size) {
                    int currentLayerIndex = shraedCurrentLayerIndex;
                    
                    if(currentGid - blockFirstGid >= sharedThreadsNumToNextLayer) {
                        while(currentGid + 1 > sharedLayersSizePrefixSum[currentLayerIndex]) {  
                            currentLayerIndex += 1;
                        }
                    }
                    float currentLayerMin = d_LayersMinMaxInput[currentLayerIndex * 2];
                    float currrentLayerMax = d_LayersMinMaxInput[currentLayerIndex * 2 + 1];
                    float range = currrentLayerMax - currentLayerMin;
                    
                    int quanData = d_QuanInput[currentGid];
                    float dequanData = ((quanData + (bins / 2)) / (1.0 * bins)) * range + currentLayerMin;
                    /*if(currentGid == 6064) {
                        printf("id%d: quandata%d range%f currentLayerMin%f dequandata%f currentLayerIndex%d\n", currentGid, quanData, range, currentLayerMin, dequanData, currentLayerIndex);
                    }*/

                    d_Output[currentGid] = dequanData;
                }
            }
        }
    }
}

int runZeroPlusQsgdDecom(float* d_Output, int* d_LayerSizePrefixSum, int* d_BitmapInput, int* d_QuanInput, float* d_LayersMinMaxInput, int size, int layerNum, int bins) {
    // Define grid sizes for CUDA kernel launch
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(GRID_SIZE, 1, 1);

    int qsgdLoopTimes = ceil((size * 1.0) / (648 * 256));

    zeroOutQsgdDecomKernel<<<dimGrid, dimBlock>>>(d_Output, d_LayerSizePrefixSum, d_BitmapInput, d_QuanInput, d_LayersMinMaxInput, bins, size, layerNum, qsgdLoopTimes);

    return 0;
}

int runQsgdDecom(float* d_Input, int* d_LayerSize, int* d_LayerSizePrefixSum, int* d_LayerBlocksNeedPrefixSum, int* d_totalBlocksNum, int* d_QuanOutput, float* d_LayersMinMaxOutput, int size, int layerNum, int reductionTimes, int maxBlocksNeed, float relEb, int bins) {
    return 0;
}

extern "C"
{
    void ZQGPUDECOM(float* d_Output, int* d_LayerSizePrefixSum, int* d_BitmapInput, int* d_QuanInput, float* d_LayersMinMaxInput, int size, int layerNum, int bins)
    {
        runZeroPlusQsgdDecom(d_Output, d_LayerSizePrefixSum, d_BitmapInput, d_QuanInput, d_LayersMinMaxInput, size, layerNum, bins);
        return;
    }
}

extern "C"
{
    void QGPUDECOM(float* d_Input, int* d_LayerSize, int* d_LayerSizePrefixSum, int* d_LayerBlocksNeedPrefixSum, int* d_totalBlocksNum, int* d_QuanOutput, float* d_LayersMinMaxOutput, int size, int layerNum, int reductionTimes, int maxBlocksNeed, float relEb, int bins)
    {
        runQsgdDecom(d_Input, d_LayerSize, d_LayerSizePrefixSum, d_LayerBlocksNeedPrefixSum, d_totalBlocksNum, d_QuanOutput, d_LayersMinMaxOutput, size, layerNum, reductionTimes, maxBlocksNeed, relEb, bins);
        return;
    }
}