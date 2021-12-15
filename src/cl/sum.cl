#if defined(__CLION_IDE__) || defined(__CLION_IDE_)

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#define WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned int *input,
                  const unsigned int n,
                  __global unsigned int *output) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);
    const bool exec = globalId < n;

    __local unsigned int mem[WORK_GROUP_SIZE];
    mem[localId] = exec ? input[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = WORK_GROUP_SIZE >> 1; i != 0; i >>= 1) {
        if (exec && localId < i) {
            mem[localId] = mem[localId] + mem[localId + i];
        }
        if (i != 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (exec && localId == 0) {
        atomic_add(output, mem[0]);
    }
}
