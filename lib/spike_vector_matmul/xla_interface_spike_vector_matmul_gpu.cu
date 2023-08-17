#include "stdio.h"
#include "spike_vector_matmul_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void get_sizes_from_opaque(void* opaque, unsigned int* sizes, const size_t opaque_len) {
  unsigned int* sizes_ptr = (unsigned int*) opaque;
  for (int i = 0; i < opaque_len; ++i) {
    sizes[i] = sizes_ptr[i];
  }
}

// compile using 
// nvcc -shared --compiler-options '-fPIC' -o libspike_vector_matmul_gpu.so xla_interface_spike_vector_matmul_gpu.cu
// TODO consider using --use_fast_math (which implies --ftz=true --prec-div=false --prec-sqrt=false)
// TODO consider using -ptx -o <filename>.ptx for "assembly" output

// template <typename T> 
void spike_vector_matmul_gpu_f32(cudaStream_t stream, void** buffers, 
                    const char* opaque, size_t opaque_len) {


//   printf("\ngen_spike_vector_gpu_f32\n");
//   printf("%zu\n", opaque_len);
//   for (unsigned i=0; i<opaque_len; ++i) {
//     printf("opaque val %i: |%c|\n", i, opaque[i]);
//     // printf("%d ", opaque[i]);
//   }
  
  unsigned int sizes[opaque_len];
  get_sizes_from_opaque((void*) opaque, sizes, opaque_len);
  // TODO long or int?
  const unsigned long batchsize = sizes[0];
  const unsigned long num_cols = sizes[1];
  const unsigned long max_num_spikes = sizes[2];
  const unsigned long use_grad_spikes = sizes[3];

  // printf("spike_vector_matmul_gpu_f32\n");
  // printf("batchsize: %lu\n", batchsize);
  // printf("num_cols: %lu\n", num_cols);
  // printf("max_num_spikes: %lu\n", max_num_spikes);
  // printf("use_grad_spikes: %lu\n", use_grad_spikes);
  const unsigned int num_spikes_offset = use_grad_spikes? batchsize : 0;

  const float *matrix = (const float *)(buffers[0]);
  const unsigned int *spike_ids = (unsigned int *)(buffers[1]);
  const unsigned int *num_spikes = (unsigned int *)(&spike_ids[2*batchsize*max_num_spikes+num_spikes_offset]);
  float *result_vector = (float *)(buffers[2]);

  dim3 grid_dim(batchsize);
  dim3 block_dim(256);

  spike_vector_matmul_gpu<<<grid_dim,block_dim,max_num_spikes*sizeof(unsigned int),stream>>>(matrix, spike_ids, num_spikes, result_vector, batchsize, num_cols, max_num_spikes);
}

void spike_vector_matmul_gpu_f32_matrix_grad(cudaStream_t stream, void** buffers, 
                    const char* opaque, size_t opaque_len) {
  
  unsigned int sizes[opaque_len];
  get_sizes_from_opaque((void*) opaque, sizes, opaque_len);
  // TODO long or int?
  const unsigned long num_rows = sizes[0];
  const unsigned long num_cols = sizes[1];
  const unsigned long batchsize = sizes[2];
  const unsigned long max_num_spikes = sizes[3];

  // printf("spike_vector_matmul_gpu_f32_matrix_grad\n");
  // printf("num_rows: %lu\n", num_rows);
  // printf("num_cols: %lu\n", num_cols);
  // printf("batchsize: %lu\n", batchsize);
  // printf("max_num_spikes: %lu\n", max_num_spikes);

  const unsigned int *spike_ids = (unsigned int *)(buffers[0]);
  const unsigned int *num_spikes = (unsigned int *)(&spike_ids[2*batchsize*max_num_spikes]);
  const float *result_grads = (const float *)(buffers[1]);

  float *matrix_grad = (float *)(buffers[2]);

  float fill_val = 0.0;
  int fill_val_int_view = *((int*) &fill_val);
  cudaMemset(matrix_grad, fill_val_int_view, num_rows*num_cols*sizeof(float));
  // TODO sync stream here?
  dim3 grid_dim(batchsize);
  dim3 block_dim(256);
  spike_vector_matmul_matrix_grad_gpu<<<grid_dim,block_dim,max_num_spikes*sizeof(unsigned int),stream>>>(matrix_grad, spike_ids, num_spikes, result_grads, batchsize, num_rows, num_cols, max_num_spikes);
}

extern "C" 
void spike_vector_matmul_gpu_f32_spikes_grad(cudaStream_t stream, void** buffers, 
                    const char* opaque, size_t opaque_len) {
  
  unsigned int sizes[opaque_len];
  get_sizes_from_opaque((void*) opaque, sizes, opaque_len);
  // TODO long or int?
  const unsigned long num_cols = sizes[0];
  const unsigned long batchsize = sizes[1];
  const unsigned long max_num_spikes = sizes[2];

  // printf("spike_vector_matmul_gpu_f32_spikes_grad\n");
  // printf("num_cols: %lu\n", num_cols);
  // printf("batchsize: %lu\n", batchsize);
  // printf("max_num_spikes: %lu\n", max_num_spikes);

  const float *matrix = (const float *)(buffers[0]);
  const float *result_grads = (const float *)(buffers[1]);
  const unsigned int *spike_ids = (unsigned int *)(buffers[2]);
  const float *spike_grads_precalc_surr = (float *)(&spike_ids[batchsize*max_num_spikes]);
  const unsigned int *num_spikes_all = (unsigned int *)(&spike_grads_precalc_surr[batchsize*max_num_spikes]);
  const unsigned int *num_spikes = (unsigned int *)(&num_spikes_all[batchsize]);
  
  unsigned int *spike_ids_res = (unsigned int *)(buffers[3]);
  float *spike_grads_res = (float *)(&spike_ids_res[batchsize*max_num_spikes]);
  unsigned int *num_spikes_all_res = (unsigned int *)(&spike_grads_res[batchsize*max_num_spikes]);
  // unsigned int *num_spikes_res = (unsigned int *)(&num_spikes_all_res[batchsize]);
  
  cudaMemcpyAsync(spike_ids_res, spike_ids, batchsize*max_num_spikes*sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(num_spikes_all_res, num_spikes_all, 2*batchsize*sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream);

  dim3 grid_dim(batchsize);
  dim3 block_dim(64, 8);
  if (block_dim.x * block_dim.y > 1024) {
    printf("Error: block_dim.x * block_dim.y > 1024");
    exit(-1);
  } else if (block_dim.x % 32 > 0) {
    printf("Warning: block_dim.x is not a multiple of 32 which will lead to supoptimal performance.");
  }
  spike_vector_grad_gpu<<<grid_dim,block_dim,max_num_spikes*sizeof(float),stream>>>(matrix, spike_ids, spike_grads_precalc_surr, num_spikes, result_grads, spike_grads_res, batchsize, num_cols, max_num_spikes);
  // TODO necessary due to cudaMemcpyAsync ?
  cudaStreamSynchronize(stream);
}

void sparse_vector_matmul_gpu_f32(cudaStream_t stream, void** buffers, 
                    const char* opaque, size_t opaque_len) {

  unsigned int sizes[opaque_len];
  get_sizes_from_opaque((void*) opaque, sizes, opaque_len);
  // TODO long or int?
  const unsigned long batchsize = sizes[0];
  const unsigned long num_cols = sizes[1];
  const unsigned long max_num_spikes = sizes[2];
  const unsigned long use_grad_spikes = sizes[3];

  // printf("sparse_vector_matmul_gpu_f32\n");
  // printf("batchsize: %lu\n", batchsize);
  // printf("num_cols: %lu\n", num_cols);
  // printf("max_num_spikes: %lu\n", max_num_spikes);
  // printf("use_grad_spikes: %lu\n", use_grad_spikes);
  const unsigned int num_spikes_offset = use_grad_spikes? batchsize : 0;

  const float *matrix = (const float *)(buffers[0]);
  const float *vals = (float *)(buffers[1]);
  const unsigned int *spike_ids = (unsigned int *)(buffers[2]);
  const unsigned int *num_spikes = (unsigned int *)(&spike_ids[2*batchsize*max_num_spikes+num_spikes_offset]);
  float *result_vector = (float *)(buffers[3]);

  dim3 grid_dim(batchsize);
  dim3 block_dim(256);

  sparse_vals_vector_matmul_gpu<<<grid_dim,block_dim,2*max_num_spikes*sizeof(unsigned int),stream>>>(matrix, vals, spike_ids, num_spikes, result_vector, batchsize, num_cols, max_num_spikes);
}

#ifdef __cplusplus
}
#endif