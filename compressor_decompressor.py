import torch
import ctypes
import numpy as np
from ctypes import *

# compute aggregation mapping Info
def computeAggregationMappingInfo(layers_size, num_layers, layers_size_prefixSum, layers_blocks_need_prefixSum, reductions_blocks_need):
    layers_blocks_need = []
    BLOCK_SIZE = 256
    max_layer_size = 0
    current_reduction_blocks_need = 0

    for i in range(num_layers):
        current_layer_size = layers_size[i]
        current_layer_blocks_need = np.ceil(layers_size[i] / BLOCK_SIZE)
        max_layer_size = max(max_layer_size, current_layer_size)
        layers_blocks_need.append(int(current_layer_blocks_need))
        current_reduction_blocks_need += current_layer_blocks_need

    reductions_blocks_need.append(int(current_reduction_blocks_need))
    max_blocks_need = int(np.ceil(max_layer_size / BLOCK_SIZE))
    reduction_times = int(np.ceil(np.log(max_blocks_need) / np.log(256))) + 1

    for i in range(reduction_times - 1):
        current_reduction_blocks_need = 0

        for j in range(num_layers):
            current_layer_blocks_need = np.ceil(layers_blocks_need[i * num_layers + j] / 256)
            current_reduction_blocks_need += current_layer_blocks_need
            

            layers_size.append(int(np.ceil(layers_size[i * num_layers + j] / 256)))
            layers_blocks_need.append(int(current_layer_blocks_need))
        
        reductions_blocks_need.append(int(current_reduction_blocks_need))

    for i in range(reduction_times):
        for j in range(num_layers):
            if j == 0:
                layers_size_prefixSum.append(layers_size[i * num_layers + j])
                layers_blocks_need_prefixSum.append(layers_blocks_need[i * num_layers + j])
            else:
                layers_size_prefixSum.append(layers_size[i * num_layers + j] + layers_size_prefixSum[i * num_layers + j - 1])
                layers_blocks_need_prefixSum.append(layers_blocks_need[i * num_layers + j] + layers_blocks_need_prefixSum[i * num_layers + j - 1])

    return max_blocks_need, reduction_times

# python-c interface
def runZeroOutPlusQsgdC():
    dll = ctypes.CDLL('./zoutPlusQsgd.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.ZQGPU
    func.argtypes = [POINTER(c_float), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_float), c_int, c_int, c_int, c_int, c_float, c_int]
    return func

def runQsgdC():
    dll = ctypes.CDLL('./zoutPlusQsgd.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.QGPU
    func.argtypes = [POINTER(c_float), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_float), c_int, c_int, c_int, c_int, c_float, c_int]
    return func

def runZeroOutPlusQsgdDecomC():
    dll = ctypes.CDLL('./zoutPlusQsgdDecom.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.ZQGPUDECOM
    func.argtypes = [POINTER(c_float), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_float), c_int, c_int, c_int]
    return func

def runQsgdDecomC():
    dll = ctypes.CDLL('./zoutPlusQsgdDecom.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.QGPUDECOM
    func.argtypes = [POINTER(c_float), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_float), c_int, c_int, c_int, c_int, c_float, c_int]
    return func

# compressor function
def compressor(input_tsr_gpu_ptr, output_bitmap_gpu_ptr, output_quan_data_gpu_ptr, output_layers_min_max_gpu_ptr, num_elements, layers_size, num_layers, rel_eb, is_contain_zero_out, bins):
    layers_size_prefixSum = []
    layers_blocks_need_prefixSum = []
    reductions_blocks_need = []

    max_blocks_need, reduction_times = computeAggregationMappingInfo(layers_size, num_layers, layers_size_prefixSum, layers_blocks_need_prefixSum, reductions_blocks_need)

    layers_size = torch.tensor(layers_size, dtype=torch.int32).to('cuda')
    layers_size_prefixSum = torch.tensor(layers_size_prefixSum, dtype=torch.int32).to('cuda')
    layers_blocks_need_prefixSum = torch.tensor(layers_blocks_need_prefixSum, dtype=torch.int32).to('cuda')
    reductions_blocks_need = torch.tensor(reductions_blocks_need, dtype=torch.int32).to('cuda')

    layers_size_gpu_ptr = layers_size.data_ptr()
    layers_size_gpu_ptr = cast(layers_size_gpu_ptr, ctypes.POINTER(c_int))

    layers_size_prefixSum_gpu_ptr = layers_size_prefixSum.data_ptr()
    layers_size_prefixSum_gpu_ptr = cast(layers_size_prefixSum_gpu_ptr, ctypes.POINTER(c_int))

    layers_blocks_need_prefixSum_gpu_ptr = layers_blocks_need_prefixSum.data_ptr()
    layers_blocks_need_prefixSum_gpu_ptr = cast(layers_blocks_need_prefixSum_gpu_ptr, ctypes.POINTER(c_int))

    reductions_blocks_need_gpu_ptr = reductions_blocks_need.data_ptr()
    reductions_blocks_need_gpu_ptr = cast(reductions_blocks_need_gpu_ptr, ctypes.POINTER(c_int))

    if is_contain_zero_out:
        __zoutQsgd = runZeroOutPlusQsgdC()
        try:
            __zoutQsgd(input_tsr_gpu_ptr, layers_size_gpu_ptr, layers_size_prefixSum_gpu_ptr, layers_blocks_need_prefixSum_gpu_ptr, reductions_blocks_need_gpu_ptr, output_bitmap_gpu_ptr, output_quan_data_gpu_ptr, output_layers_min_max_gpu_ptr, num_elements, num_layers, reduction_times, max_blocks_need, rel_eb, bins)
        except Exception as e:
            print(f"CUDA Error: {str(e)}")
    else:
        __qsgd = runQsgdC()
        try:
            __qsgd(input_tsr_gpu_ptr, layers_size_gpu_ptr, layers_size_prefixSum_gpu_ptr, layers_blocks_need_prefixSum_gpu_ptr, reductions_blocks_need_gpu_ptr, output_quan_data_gpu_ptr, output_layers_min_max_gpu_ptr, num_elements, num_layers, reduction_times, max_blocks_need, rel_eb, bins)
        except Exception as e:
            print(f"CUDA Error: {str(e)}")

# decompressor function
def decompressor(input_bitmap_gpu_ptr, input_quan_data_gpu_ptr, input_layers_min_max_gpu_ptr, output_tsr_gpu_ptr, num_elements, layers_size, num_layers, is_contain_zero_out, bins):
    layers_size_prefixSum = []
    layers_blocks_need_prefixSum = []
    reductions_blocks_need = []

    max_blocks_need, reduction_times = computeAggregationMappingInfo(layers_size, num_layers, layers_size_prefixSum, layers_blocks_need_prefixSum, reductions_blocks_need)

    layers_size = torch.tensor(layers_size, dtype=torch.int32).to('cuda')
    layers_size_prefixSum = torch.tensor(layers_size_prefixSum, dtype=torch.int32).to('cuda')
    layers_blocks_need_prefixSum = torch.tensor(layers_blocks_need_prefixSum, dtype=torch.int32).to('cuda')
    reductions_blocks_need = torch.tensor(reductions_blocks_need, dtype=torch.int32).to('cuda')

    layers_size_gpu_ptr = layers_size.data_ptr()
    layers_size_gpu_ptr = cast(layers_size_gpu_ptr, ctypes.POINTER(c_int))

    layers_size_prefixSum_gpu_ptr = layers_size_prefixSum.data_ptr()
    layers_size_prefixSum_gpu_ptr = cast(layers_size_prefixSum_gpu_ptr, ctypes.POINTER(c_int))

    layers_blocks_need_prefixSum_gpu_ptr = layers_blocks_need_prefixSum.data_ptr()
    layers_blocks_need_prefixSum_gpu_ptr = cast(layers_blocks_need_prefixSum_gpu_ptr, ctypes.POINTER(c_int))

    reductions_blocks_need_gpu_ptr = reductions_blocks_need.data_ptr()
    reductions_blocks_need_gpu_ptr = cast(reductions_blocks_need_gpu_ptr, ctypes.POINTER(c_int))

    if is_contain_zero_out:
        __zoutQsgdDecom = runZeroOutPlusQsgdDecomC()
        try:
            __zoutQsgdDecom(output_tsr_gpu_ptr, layers_size_prefixSum_gpu_ptr, input_bitmap_gpu_ptr, input_quan_data_gpu_ptr, input_layers_min_max_gpu_ptr, num_elements, num_layers, bins)
        except Exception as e:
            print(f"CUDA Error: {str(e)}")
    else:
        __qsgdDecom = runQsgdDecomC()
        try:
            __qsgdDecom(output_tsr_gpu_ptr, layers_size_gpu_ptr, layers_size_prefixSum_gpu_ptr, layers_blocks_need_prefixSum_gpu_ptr, reductions_blocks_need_gpu_ptr, input_quan_data_gpu_ptr, input_layers_min_max_gpu_ptr, num_elements, num_layers, reduction_times, max_blocks_need, rel_eb, bins)
        except Exception as e:
            print(f"CUDA Error: {str(e)}")

# resnet 50 step 100 dataset test
if __name__ == '__main__':
    torch.cuda.set_device(0)

    # parameters setting
    rel_eb = 4e-3
    bins = 250 # rel_eb and bins are connected
    is_contain_zero_out = True

    # assign data holders
    aggregated_data = np.array([])
    layers_size = []
    num_layers = int(53)

    # Files input
    for i in range(num_layers):    
        file_name = './grad_res50/layer' + str(i) + '_step100.npy'
        data = np.load(file_name)
        aggregated_data = np.append(aggregated_data, data)
        layers_size.append(int(np.size(data)))

    input_tsr = torch.tensor(aggregated_data, dtype=torch.float32).to('cuda')
    num_elements = int(input_tsr.numel())

    output_tsr = torch.zeros(input_tsr.shape, dtype=torch.float32).cuda()
    bitmap = torch.zeros(int(np.ceil(input_tsr.numel() / 32)), dtype=torch.int32).cuda()
    quan_data = torch.zeros(input_tsr.shape, dtype=torch.int32).cuda()
    layers_min_max = torch.zeros(num_layers * 2, dtype=torch.float32).cuda()

    input_tsr_gpu_ptr = input_tsr.data_ptr()
    input_tsr_gpu_ptr = cast(input_tsr_gpu_ptr, ctypes.POINTER(c_float))

    output_tsr_gpu_ptr = output_tsr.data_ptr()
    output_tsr_gpu_ptr = cast(output_tsr_gpu_ptr, ctypes.POINTER(c_float))

    bitmap_gpu_ptr = bitmap.data_ptr()
    bitmap_gpu_ptr = cast(bitmap_gpu_ptr, ctypes.POINTER(c_int))

    quan_data_gpu_ptr = quan_data.data_ptr()
    quan_data_gpu_ptr = cast(quan_data_gpu_ptr, ctypes.POINTER(c_int))

    layers_min_max_gpu_ptr = layers_min_max.data_ptr()
    layers_min_max_gpu_ptr = cast(layers_min_max_gpu_ptr, ctypes.POINTER(c_float))

    compressor(input_tsr_gpu_ptr, bitmap_gpu_ptr, quan_data_gpu_ptr, layers_min_max_gpu_ptr, num_elements, layers_size, num_layers, rel_eb, is_contain_zero_out, bins)

    decompressor(bitmap_gpu_ptr, quan_data_gpu_ptr, layers_min_max_gpu_ptr, output_tsr_gpu_ptr, num_elements, layers_size, num_layers, is_contain_zero_out, bins)

    print(input_tsr[:100])

    print(output_tsr[:100])

    input_tsr_cpu = input_tsr.to('cpu')
    input_tsr_cpu = input_tsr_cpu.numpy()

    output_tsr_cpu = output_tsr.to('cpu')
    output_tsr_cpu = output_tsr_cpu.numpy()

    layers_min_max_cpu = layers_min_max.to('cpu')
    layers_min_max_cpu = layers_min_max_cpu.numpy()

    size_count = 0
    count = 0
    for i in range(53):
        size = layers_size[i]
        data_range = layers_min_max_cpu[i * 2 + 1] - layers_min_max_cpu[i * 2]
        eb = data_range / bins
        for j in range(size):
            if abs(output_tsr_cpu[size_count + j] - input_tsr_cpu[size_count + j]) > eb:
                print(str(size_count + j) + 'is out of eb' + ' and dequan data is ' + str(output_tsr_cpu[size_count + j]) + ' and origin data is ' + str(input_tsr_cpu[size_count + j]) +'!!!')
                count += 1
        
        size_count += size

    print('count:' + str(count))
    print('end of check!')

    print("finished")

    # abs(output_tsr_cpu[size_count + j] - input_tsr_cpu[size_count + j]) > eb
    # output_tsr_cpu[size_count + j] == 0
    