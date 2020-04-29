/**
 * Auto correlation computation
 * f(tau) = sum f(x) * f(x+tau) for x from 0 to N-tau
 */
__kernel void acorr(
    const    int     N,
    __global float* sample,
    __global float* output)
{
    const int global_id = get_global_id(0);

    const int tau = global_id;
    float sum     = 0.0f;
    for(int i = 0; i < N - tau; i++) {
        //printf("[%d]+=[%d]*[%d]\n", global_id, i, i+tau);
        sum += (sample[i] * sample[i + tau]);
    }
    // copy local buf to ouput buffer
    output[global_id] = sum;
}

__kernel void acorr_local(
    const    int     N,
    __global float* sample,
    __global float* output,
    __local  float* locbuf)
{
    const int global_id  = get_global_id(0);
    const int group_id    = get_group_id(0);
    const int local_size  = get_local_size(0);
    const int local_id    = get_local_id(0);

    event_t evt_copy_in = async_work_group_copy(locbuf,
                                                sample + group_id * local_size,
                                                local_size,
                                                0);

    const int tau = global_id;
    float sum     = 0.0f;
    for(int i = 0; i < N - tau; i++) {
        //printf("[%d]+=[%d]*[%d]\n", global_id, i, i+tau);
        sum += (locbuf[i] * locbuf[i + tau]);
    }
    // copy local buf to ouput buffer
    locbuf[global_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    event_t evt_copy_out = async_work_group_copy(sample + local_size * group_id,
                                                 locbuf,
                                                 local_size,
                                                 0);
    wait_group_events(1, &evt_copy_out);
}

__kernel void acorr_vec4(
    const    int    N,
    __global float* sample,
    __global float* output)
{
    const int global_id = get_global_id(0);

    const int tau = global_id;
    float     sum = 0.0f;
    for(int i = 0; i < N - tau; i+=4) {
        //printf("[%d]+=[%d]*[%d]\n", global_id, i, i+tau);
        float4 a = vload4(i, sample);
        float4 b = vload4(i+tau, sample);
        sum += dot(a, b);
        //printf("[%d]*[%d]=(%f,%f,%f,%f)*(%f,%f,%f,%f)=%f\n", i, i+tau,
        //       a->x, a->y, a->z, a->w, b->x, b->y, b->z, b->w, sum);
    }
    // copy to ouput buffer
    output[global_id] = sum;
}

uint us8_dot(ushort8 a, ushort8 b)
{
    ushort8 c = a * b;
    uint sum = c.s0 + c.s1 + c.s2 + c.s3 + c.s4 + c.s5 + c.s6 + c.s7;
    return sum;
}

__kernel void acorr_vec8(
    const    int    N,
    __global ushort* sample,
    __global ushort* output)
{
    const int global_id = get_global_id(0);

    const int tau = global_id;
    uint    sum = 0.0f;
    for(int i = 0; i < N - tau; i+=8) {
        ushort8 a = vload8(i, sample);
        ushort8 b = vload8(i+tau, sample);

        sum += us8_dot(a, b);
    }
    // copy to ouput buffer
    output[global_id] = sum;
}
