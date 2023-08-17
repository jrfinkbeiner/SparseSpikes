#include "../cuda_utils.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__<110
#error compute capability 1.1 is required for atomic operations
#endif

__device__ float superspike_surrogate_gpu(const float *state, const float* threshold) {
  float beta{10};
  return std::pow((beta * std::abs(state[0]-threshold[0]) + 1.0f), -2);
}

__global__ void gen_spike_vector_from_dense_gpu(const float* states, unsigned int* spike_ids, float* spike_grads, unsigned int* num_spikes,
                     //  unsigned int* spike_ids_fwd, unsigned int* spike_ids_grad,
                    //  const float threshold_spike, const float threshold_grad,
                     const float* thresholds,
                     const unsigned int num_max_spikes, const unsigned int batchsize, const unsigned int num_states,
                     const bool precalc_surrogate){
  // TODO use `precalc_surrogate` flag or just separate function?

  // only one shared memory per block, even if it is processing multiple batches
  extern __shared__ char shared[];
  unsigned int *spike_ids_local = (unsigned int *)shared; // TODO think about bank alignment...
  float *spike_grads_local;
  unsigned int *num_spikes_local_stage1;
  
  if (precalc_surrogate){
    spike_grads_local = (float *)&spike_ids_local[2*num_max_spikes];
    num_spikes_local_stage1 = (unsigned int *)&spike_grads_local[2*num_max_spikes];
  } else {
    num_spikes_local_stage1 = (unsigned int *)&spike_ids_local[2*num_max_spikes];
  }
  unsigned int *num_spikes_local = (unsigned int *)&num_spikes_local_stage1[2];  
  
  const float threshold_spike = thresholds[0]; 
  const float threshold_grad = thresholds[1];

  for (unsigned ibatch=blockIdx.x; ibatch<batchsize; ibatch+=blockDim.x){
    if (threadIdx.x < 2){
      num_spikes_local_stage1[threadIdx.x] = 0;
      num_spikes_local[threadIdx.x] = 0;
    };
    __syncthreads();

    for(int i=threadIdx.x; i<num_states; i+=blockDim.x) {
        int idx_batch_offset_states = ibatch * num_states;
        float state = states[i+idx_batch_offset_states];
        if ((state > threshold_spike) && (atomicAdd(&num_spikes_local_stage1[0], 1) < num_max_spikes)) {
          const unsigned int idx = atomicAdd(&num_spikes_local[0], 1);
          spike_ids_local[idx] = i;
          if (precalc_surrogate){
            spike_grads_local[idx] = superspike_surrogate_gpu(&state, &threshold_spike);
          }
        } else if ((state > threshold_grad) && (atomicAdd(&num_spikes_local_stage1[1], 1) < num_max_spikes)) {
          const unsigned int idx = atomicAdd(&num_spikes_local[1], 1);
          spike_ids_local[idx + num_max_spikes] = i;
          if (precalc_surrogate){
            spike_grads_local[idx + num_max_spikes] = superspike_surrogate_gpu(&state, &threshold_spike);
          }
        }
        // // TODO another if statement with break ?
        // if (num_spikes_local[0]+num_spikes_local[1]){
        //   break;
        // }
    }
    __syncthreads();

    // copy data to global memory
    int idx_batch_offset_spikes = ibatch * num_max_spikes;
    for(int i=threadIdx.x; i<min((num_spikes_local[0]+num_spikes_local[1]), num_max_spikes); i+=blockDim.x) {
      // TODO use addditional shared_memory for combine operation?
      // TODO why not just write everything to global memory ?
      if (i < num_spikes_local[0]) {
        spike_ids[i+idx_batch_offset_spikes] = spike_ids_local[i];
        if (precalc_surrogate){
          spike_grads[i+idx_batch_offset_spikes] = spike_grads_local[i];
        }
      } else {
        spike_ids[i+idx_batch_offset_spikes] = spike_ids_local[i-num_spikes_local[0]+num_max_spikes]; // TODO NO MINUS!!!
        if (precalc_surrogate){
          spike_grads[i+idx_batch_offset_spikes] = spike_grads_local[i-num_spikes_local[0]+num_max_spikes];
        }
      }
    }
    if (threadIdx.x == 0){
      // int idx_batch_offset_num_spikes = ibatch * 2;
      // num_spikes[0+idx_batch_offset_num_spikes] = num_spikes_local[threadIdx.x];
      // num_spikes[1+idx_batch_offset_num_spikes] = min((num_spikes_local[0]+num_spikes_local[1]), num_max_spikes);
      num_spikes[ibatch] = num_spikes_local[threadIdx.x];
      num_spikes[ibatch+batchsize] = min((num_spikes_local[0]+num_spikes_local[1]), num_max_spikes);
    }
  }
}

// ----------------------- grad ------------------------------

__global__ void gen_spike_vector_from_dense_grad_gpu_precalculated_surrogate(const unsigned int* spike_ids, const unsigned int* num_grad_spikes,
                     const float* spike_grads, float* state_grads,
                     const unsigned int num_max_spikes, const unsigned int batchsize, const unsigned int num_states){

  extern __shared__ char shared[];
  // float* state_local = (float *)&shared[0];
  // float* state_grad_local = (float *)&state_local[blockDim.x];
  float* state_grad_local = (float *)&shared[0];
  unsigned int* spike_ids_local = (unsigned int *)&state_grad_local[blockDim.x];

  for (unsigned ibatch=blockIdx.x; ibatch<batchsize; ibatch+=gridDim.x){
    const unsigned int num_grad_spikes_thisThread{num_grad_spikes[ibatch]}; // or use shared memory or is it cached somehow anyway?
    const float* spike_grads_ibatch = spike_grads + ibatch*num_max_spikes;
    for(unsigned i=threadIdx.x; i<num_grad_spikes_thisThread; i+=blockDim.x) {
      spike_ids_local[i] = spike_ids[i+ibatch*num_max_spikes];
    }
    __syncthreads();

    unsigned start{0};
    unsigned end{blockDim.x};
    // TODO change to loop ((num_states+1) / blockDim.x) times
    for(unsigned i=threadIdx.x; i<num_states; i+=blockDim.x) {
      state_grad_local[threadIdx.x] = 0.0;
      // state_local[threadIdx.x] = states[i+ibatch*num_states];
      __syncthreads();
      for(unsigned ispk=threadIdx.x; ispk<num_grad_spikes_thisThread; ispk+=blockDim.x) {
        if ((spike_ids_local[ispk] < end) && (start <= spike_ids_local[ispk])) {
          // atomicAdd(&state_grad_local[spike_ids_local[ispk]-start], 1.0f); // or atomicSet ?
          atomicAdd(&state_grad_local[spike_ids_local[ispk]-start], spike_grads_ibatch[ispk]); // or atomicSet ?
        }
      }
      __syncthreads();
      state_grads[i+ibatch*num_states] = state_grad_local[threadIdx.x];
      start = end;
      end += blockDim.x;
    }
  }
}
