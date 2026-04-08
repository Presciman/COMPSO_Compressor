#include <stdio.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#define FULL_MASK 0xffffffff
#define BLOCK_SIZE 256
#define GRID_SIZE 648

// helper function to ceil in cuda c kernel
__device__ int myCeil(float a) {
    int intPart = (int)a;

    if (a == (float)intPart) {
        return a;
    } else {
        return intPart + 1;
    }
}

__inline__ __device__
float2 warpShuffleMinMax(float2 val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float neighbor_min = __shfl_down_sync(FULL_MASK, val.x, offset);
        float neighbor_max = __shfl_down_sync(FULL_MASK, val.y, offset);
        float min = fmin(val.x, neighbor_min);
        float max = fmax(val.y, neighbor_max);
        val = make_float2(min, max);
    }

    return val;
}

__global__
void minMaxKernel(float* d_Input, int* d_LayerSize, int* d_LayerSizePrefixSum, int* d_LayerBlocksNeedPrefixSum, int* d_totalBlocksNum, float2* d_MinMaxTempHolder1, float2* d_MinMaxTempHolder2, int layerNum, int maxBlocksNeed, int reductionCount, int reductionTimes) {
    __shared__ float2 partialMinMax[8];
    __shared__ int shraedCurrentLayerIndex;
    extern __shared__ int sharedLayersBlocksNeedPrefixSum[];

    // load layer size and block numbers into shared memory
    int count = 0;
    int sharedId = count * BLOCK_SIZE + threadIdx.x;
    while(sharedId < layerNum) {
        int globalId = (reductionCount - 1) * layerNum + count * BLOCK_SIZE + threadIdx.x;

        sharedLayersBlocksNeedPrefixSum[sharedId] = d_LayerBlocksNeedPrefixSum[globalId];

        count++;
        sharedId = count * BLOCK_SIZE + threadIdx.x;
    }

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread index in the grid 
    int lane = threadIdx.x - (threadIdx.x / warpSize) * warpSize; // lane index in one warp, euqal to threadIdx.x % warpSize, for faster speed version
    int wid = threadIdx.x / warpSize; // warp index in one block
    int warpNum = 8; // how many warps in one block
    int currentTotalBlocksNum = d_totalBlocksNum[reductionCount - 1];
    int minMaxLoopTimes = myCeil((currentTotalBlocksNum * 1.0) / GRID_SIZE);
    float minMaxData;
    float2 dataPack;

    for(int i = 0; i < minMaxLoopTimes; i++) {
        int currentGid = i * gridDim.x * blockDim.x + tid;
        int currrentBlockId = i * gridDim.x + blockIdx.x;

        if(currrentBlockId < currentTotalBlocksNum) {
            // get whole block's layer index
            if(threadIdx.x == 0) {
                int left = 0, right = layerNum - 1;
                int mid = 0;
                while(left < right) {
                    mid = left + (right - left) / 2;
                    int tempBlocksNedd = sharedLayersBlocksNeedPrefixSum[mid];
                    if(currrentBlockId + 1 < tempBlocksNedd) {
                        if(mid == 0 || currrentBlockId + 1 > sharedLayersBlocksNeedPrefixSum[mid - 1]) {
                            break;
                        }else {
                            right = mid - 1;
                        }
                    }else if(currrentBlockId + 1 > tempBlocksNedd) {
                        if(mid == layerNum - 1 || currrentBlockId + 1 <= sharedLayersBlocksNeedPrefixSum[mid + 1]) {
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
            }

            __syncthreads();
            
            int currentLayerIndex = shraedCurrentLayerIndex;
            int prevsLayersThreadsNeed;
            int prevLayersBlocksNeed;
            if(currentLayerIndex == 0) {
                prevsLayersThreadsNeed = 0;
                prevLayersBlocksNeed = 0;
            }else {
                prevsLayersThreadsNeed = d_LayerSizePrefixSum[(reductionCount - 1) * layerNum + currentLayerIndex - 1];
                prevLayersBlocksNeed = sharedLayersBlocksNeedPrefixSum[currentLayerIndex - 1];
            }
            int currentLayerThreadId =  currentGid - BLOCK_SIZE * prevLayersBlocksNeed;

            // shuffle and reduction operation
            // load data
            if(currentLayerThreadId < d_LayerSize[(reductionCount - 1) * layerNum + currentLayerIndex]) {
                if(reductionCount == 1) { // first time
                    minMaxData = d_Input[prevsLayersThreadsNeed + currentLayerThreadId]; 
                    dataPack = make_float2(minMaxData, minMaxData);
                }else if(reductionCount - (reductionCount / 2) * 2 == 0){ // even times
                    dataPack = d_MinMaxTempHolder1[currentLayerIndex * maxBlocksNeed + currentLayerThreadId];
                }else { // odd times
                    dataPack = d_MinMaxTempHolder2[currentLayerIndex * maxBlocksNeed + currentLayerThreadId];
                }
            }else {
                dataPack = make_float2(1000.0f, -1000.0f);
            }

            dataPack = warpShuffleMinMax(dataPack);

            if(lane == 0) {
                partialMinMax[wid] = dataPack;
            }
        
            __syncthreads();

            if(threadIdx.x < warpNum) {
                dataPack = partialMinMax[threadIdx.x];
            }

            if(wid == 0) {
                dataPack = warpShuffleMinMax(dataPack);
            }

            // output data
            if(threadIdx.x == 0) {
                if(reductionCount - (reductionCount / 2) * 2 != 0) { // odd times
                    d_MinMaxTempHolder1[currentLayerIndex * maxBlocksNeed + currrentBlockId - prevLayersBlocksNeed] = dataPack;
                }else { // even times
                    d_MinMaxTempHolder2[currentLayerIndex * maxBlocksNeed + currrentBlockId - prevLayersBlocksNeed] = dataPack;
                }
            }
        }
    }
}

__global__ 
void zeroOutKernel(float *d_Input, int* d_LayerSizePrefixSum, float2* d_MinMaxTempHolder, int *d_BitmapOutput, int size, int layerNum, int maxBlocksNeed, float relEb, int zeroOutLoopTimes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread index

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

    for(int i = 0; i < zeroOutLoopTimes; i++) {
        int blockFirstGid = blockIdx.x * blockDim.x * 32 + i * 32 * gridDim.x * blockDim.x;
        int startGid = tid * 32 + i * 32 * gridDim.x * blockDim.x;
        int endGid = startGid + 32;

        if(blockFirstGid < size) {
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

            if(startGid < size) {
                int currentGid = startGid;
                int blockPartialBits = 0;
                int currentLayerIndex = shraedCurrentLayerIndex;

                while(currentGid < size && currentGid < endGid) {
                    float bitmapData = d_Input[currentGid];

                    if(currentGid - blockFirstGid >= sharedThreadsNumToNextLayer) {
                        while(currentGid + 1 > sharedLayersSizePrefixSum[currentLayerIndex]) {    
                            currentLayerIndex += 1;
                        }
                    }

                    float2 currentLayerMinMax = d_MinMaxTempHolder[currentLayerIndex * maxBlocksNeed];
                    float currentLayerRange = currentLayerMinMax.y - currentLayerMinMax.x;
                    float absEb = currentLayerRange * relEb;

                    if (bitmapData > absEb || bitmapData < -absEb) {
                        blockPartialBits |= 0x80000000 >> (currentGid - (currentGid / 32) * 32); // euqal to currentGid % 32, for faster speed
                    }
                    currentGid += 1;
                }

                d_BitmapOutput[tid + i * gridDim.x * blockDim.x] = blockPartialBits;
            }
        }
    }
}

__global__
void qsgdKernel(float* d_Input, int* d_LayerSizePrefixSum, float2* d_MinMaxTempHolder, int* d_QuanOutput, float* d_LayersMinMaxOutput, int bins, int size, int layerNum, int maxBlocksNeed, int qsgdLoopTimes) {
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

            if(currentGid < size) {
                // store layer min max in output array
                if(currentGid < layerNum) {
                    float2 tempLayerRange = d_MinMaxTempHolder[currentGid * maxBlocksNeed];
                    d_LayersMinMaxOutput[currentGid * 2] = tempLayerRange.x;
                    d_LayersMinMaxOutput[currentGid * 2 + 1] = tempLayerRange.y; 
                }

                int currentLayerIndex = shraedCurrentLayerIndex;
                
                if(currentGid - blockFirstGid >= sharedThreadsNumToNextLayer) {
                    while(currentGid + 1 > sharedLayersSizePrefixSum[currentLayerIndex]) {    
                        currentLayerIndex += 1;
                    }    
                }

                float2 currentLayerRange = d_MinMaxTempHolder[currentLayerIndex * maxBlocksNeed];
                float range = currentLayerRange.y - currentLayerRange.x;
                
                float qsgdData = d_Input[currentGid];
                float renormalize_p = ((qsgdData - currentLayerRange.x) / range) * bins;

                float floor_p = floorf(renormalize_p);
                float final_p = renormalize_p - floor_p;

                float rand_edge = ((__cosf((renormalize_p - floor_p) * CUDART_PI_F) + 1) / 2);
                float edge = floorf(final_p - rand_edge + 1);

                /*if(currentGid == 4267) {
                    printf("id%d: range%f countrange%f renormalize_p%f floor_p%f, edge%f", currentGid, range, qsgdData - currentLayerRange.x, renormalize_p, floor_p, edge);
                }*/

                d_QuanOutput[currentGid] = -(bins / 2) + (int)(floor_p + edge);
            }
        }
    }
}

int runZeroPlusQsgd(float* d_Input, int* d_LayerSize, int* d_LayerSizePrefixSum, int* d_LayerBlocksNeedPrefixSum, int* d_totalBlocksNum, int* d_BitmapOutput, int* d_QuanOutput, float* d_LayersMinMaxOutput, int size, int layerNum, int reductionTimes, int maxBlocksNeed, float relEb, int bins) {
    // Define grid sizes for CUDA kernel launch
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(GRID_SIZE, 1, 1);

    int zeroOutLoopTimes = ceil((size * 1.0) / (32 * 648 * 256));
    int qsgdLoopTimes = ceil((size * 1.0) / (648 * 256));

    float2* d_MinMaxTempHolder1 = NULL;
    float2* d_MinMaxTempHolder2 = NULL;
    float2* d_MinMaxTempHolder = NULL;
    cudaMalloc((void**)&d_MinMaxTempHolder1, maxBlocksNeed * layerNum * sizeof(float2));
    cudaMalloc((void**)&d_MinMaxTempHolder2, maxBlocksNeed * layerNum * sizeof(float2));

    int reductionCount = 1;
    minMaxKernel<<<dimGrid, dimBlock>>>(d_Input, d_LayerSize, d_LayerSizePrefixSum, d_LayerBlocksNeedPrefixSum, d_totalBlocksNum, d_MinMaxTempHolder1, d_MinMaxTempHolder2, layerNum, maxBlocksNeed, reductionCount, reductionTimes);

    while(reductionCount != reductionTimes) {
        reductionCount++;
        minMaxKernel<<<dimGrid, dimBlock>>>(d_Input, d_LayerSize, d_LayerSizePrefixSum, d_LayerBlocksNeedPrefixSum, d_totalBlocksNum, d_MinMaxTempHolder1, d_MinMaxTempHolder2, layerNum, maxBlocksNeed, reductionCount, reductionTimes);
    }

    if(reductionTimes % 2 == 0) {
        d_MinMaxTempHolder = d_MinMaxTempHolder2;
    }else {
        d_MinMaxTempHolder = d_MinMaxTempHolder1;
    }
    
    zeroOutKernel<<<dimGrid, dimBlock>>>(d_Input, d_LayerSizePrefixSum, d_MinMaxTempHolder, d_BitmapOutput, size, layerNum, maxBlocksNeed, relEb, zeroOutLoopTimes);

    qsgdKernel<<<dimGrid, dimBlock>>>(d_Input, d_LayerSizePrefixSum, d_MinMaxTempHolder, d_QuanOutput, d_LayersMinMaxOutput, bins, size, layerNum, maxBlocksNeed, qsgdLoopTimes);

    cudaFree(d_MinMaxTempHolder1);
    cudaFree(d_MinMaxTempHolder2);

    return 0;
}

int runQsgd(float* d_Input, int* d_LayerSize, int* d_LayerSizePrefixSum, int* d_LayerBlocksNeedPrefixSum, int* d_totalBlocksNum, int* d_QuanOutput, float* d_LayersMinMaxOutput, int size, int layerNum, int reductionTimes, int maxBlocksNeed, float relEb, int bins) {
    // Define grid sizes for CUDA kernel launch
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(GRID_SIZE, 1, 1);

    int qsgdLoopTimes = ceil((size * 1.0) / (648 * 256));

    float2* d_MinMaxTempHolder1 = NULL;
    float2* d_MinMaxTempHolder2 = NULL;
    float2* d_MinMaxTempHolder = NULL;
    cudaMalloc((void**)&d_MinMaxTempHolder1, maxBlocksNeed * layerNum * sizeof(float2));
    cudaMalloc((void**)&d_MinMaxTempHolder2, maxBlocksNeed * layerNum * sizeof(float2));

    int reductionCount = 1;
    minMaxKernel<<<dimGrid, dimBlock>>>(d_Input, d_LayerSize, d_LayerSizePrefixSum, d_LayerBlocksNeedPrefixSum, d_totalBlocksNum, d_MinMaxTempHolder1, d_MinMaxTempHolder2, layerNum, maxBlocksNeed, reductionCount, reductionTimes);

    while(reductionCount != reductionTimes) {
        reductionCount++;
        minMaxKernel<<<dimGrid, dimBlock>>>(d_Input, d_LayerSize, d_LayerSizePrefixSum, d_LayerBlocksNeedPrefixSum, d_totalBlocksNum, d_MinMaxTempHolder1, d_MinMaxTempHolder2, layerNum, maxBlocksNeed, reductionCount, reductionTimes);
    }

    if(reductionTimes % 2 == 0) {
        d_MinMaxTempHolder = d_MinMaxTempHolder2;
    }else {
        d_MinMaxTempHolder = d_MinMaxTempHolder1;
    }

    qsgdKernel<<<dimGrid, dimBlock>>>(d_Input, d_LayerSizePrefixSum, d_MinMaxTempHolder, d_QuanOutput, d_LayersMinMaxOutput, bins, size, layerNum, maxBlocksNeed, qsgdLoopTimes);

    cudaFree(d_MinMaxTempHolder1);
    cudaFree(d_MinMaxTempHolder2);

    return 0;
}

extern "C"
{
    void ZQGPU(float* d_Input, int* d_LayerSize, int* d_LayerSizePrefixSum, int* d_LayerBlocksNeedPrefixSum, int* d_totalBlocksNum, int* d_BitmapOutput, int* d_QuanOutput, float* d_LayersMinMaxOutput, int size, int layerNum, int reductionTimes, int maxBlocksNeed, float relEb, int bins)
    {
        runZeroPlusQsgd(d_Input, d_LayerSize, d_LayerSizePrefixSum, d_LayerBlocksNeedPrefixSum, d_totalBlocksNum, d_BitmapOutput, d_QuanOutput, d_LayersMinMaxOutput, size, layerNum, reductionTimes, maxBlocksNeed, relEb, bins);
        return;
    }
}

extern "C"
{
    void QGPU(float* d_Input, int* d_LayerSize, int* d_LayerSizePrefixSum, int* d_LayerBlocksNeedPrefixSum, int* d_totalBlocksNum, int* d_QuanOutput, float* d_LayersMinMaxOutput, int size, int layerNum, int reductionTimes, int maxBlocksNeed, float relEb, int bins)
    {
        runQsgd(d_Input, d_LayerSize, d_LayerSizePrefixSum, d_LayerBlocksNeedPrefixSum, d_totalBlocksNum, d_QuanOutput, d_LayersMinMaxOutput, size, layerNum, reductionTimes, maxBlocksNeed, relEb, bins);
        return;
    }
}