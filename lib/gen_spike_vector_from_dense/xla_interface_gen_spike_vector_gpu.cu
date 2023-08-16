#include "spike_vector_from_dense_gpu.h"

void get_sizes_from_opaque(void* opaque, unsigned int* sizes) {
  unsigned int* sizes_ptr = (unsigned int*) opaque;
  sizes[0] = sizes_ptr[0];
  sizes[1] = sizes_ptr[1];
  sizes[2] = sizes_ptr[2];
}

// compile using 
// nvcc -shared --compiler-options '-fPIC' -o libgen_sparse_spikes_gpu.so xla_interface_gen_spike_vector_gpu.cu
// TODO consider using --use_fast_math (which implies --ftz=true --prec-div=false --prec-sqrt=false)
// TODO consider using -ptx -o <filename>.ptx for "assembly" output

// template <typename T> 
extern "C" 
void gen_spike_vector_gpu_f32(cudaStream_t stream, void** buffers, 
                    const char* opaque, size_t opaque_len) {


//   printf("\ngen_spike_vector_gpu_f32\n");
//   printf("%zu\n", opaque_len);
//   for (unsigned i=0; i<opaque_len; ++i) {
//     printf("opaque val %i: |%c|\n", i, opaque[i]);
//     // printf("%d ", opaque[i]);
//   }
  
  unsigned int sizes[3];
  get_sizes_from_opaque((void*) opaque, sizes);
  // TODO long or int?
  const unsigned long batchsize = sizes[0];
  const unsigned long num_states = sizes[1];
  const unsigned long max_num_spikes = sizes[2];

  // printf("sizes: %i %i %i\n", sizes[0], sizes[1], sizes[2]);
  // printf("batchsize: %lu\n", batchsize);
  // printf("num_states: %lu\n", num_states);
  // printf("max_num_spikes: %lu\n", max_num_spikes);

  const float *states = (const float *)(buffers[0]);
  const float *thresholds = (const float*)(buffers[1]);

  unsigned int *spike_ids = (unsigned int *)(buffers[2]);
  float *spike_grads = (float *)(&spike_ids[batchsize*max_num_spikes]);
  unsigned int *num_spikes = (unsigned int *)(&spike_grads[batchsize*max_num_spikes]);

  const bool precalc_surrogate = true;

  dim3 grid_dim(batchsize);
  dim3 block_dim(512);
  unsigned int shared_mem_size = (2*max_num_spikes+2+2)*sizeof(unsigned int);
  if (precalc_surrogate){
    shared_mem_size += 2*max_num_spikes*sizeof(float);
  }

  gen_spike_vector_from_dense_gpu<<<grid_dim, block_dim,
                      shared_mem_size, stream>>>(states, spike_ids, spike_grads, num_spikes,
                    //  threshold_spike, threshold_grad,
                      thresholds,
                     max_num_spikes, batchsize, num_states, precalc_surrogate);  
}



// template <typename T> 
extern "C" 
void gen_spike_vector_gpu_f32_grad(cudaStream_t stream, void** buffers, 
                    const char* opaque, size_t opaque_len) {

  unsigned int sizes[3];
  get_sizes_from_opaque((void*) opaque, sizes);
  // TODO long or int?
  const unsigned long batchsize = sizes[0];
  const unsigned long num_states = sizes[1];
  const unsigned long max_num_spikes = sizes[2];

  // const float *states = (const float *)(buffers[0]);
  // const float *thresholds = (const float*)(buffers[1]);

  unsigned int *spike_ids = (unsigned int *)(buffers[0]);
  float* spike_grads = (float *)(&spike_ids[batchsize*max_num_spikes]);
  unsigned int *num_spikes_grad = (unsigned int *)(&spike_grads[batchsize*max_num_spikes+batchsize]);
  
  float *state_grads = (float *)(buffers[1]);

  dim3 grid_dim(batchsize);
  dim3 block_dim(512);

  gen_spike_vector_from_dense_grad_gpu_precalculated_surrogate<<<grid_dim, block_dim, (block_dim.x*sizeof(float)+max_num_spikes*sizeof(unsigned int))>>>
                    (spike_ids,
                     num_spikes_grad, spike_grads, state_grads,
                     max_num_spikes, batchsize, num_states);
}