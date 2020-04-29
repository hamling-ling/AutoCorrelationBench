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
