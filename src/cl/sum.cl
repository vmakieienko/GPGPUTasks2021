#if defined(__CLION_IDE__) || defined(__CLION_IDE_)

#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned int *input,
                           const unsigned int n,
                  __global unsigned int *output) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int mem[WORK_GROUP_SIZE];
    if (globalId >= n) {
        mem[localId] = 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        return;
    }

    mem[localId] = input[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            sum += mem[i];
        }
        atomic_add(output, sum);
    }
}
