// for now one kernel per batch
__global__ void spike_vector_matmul_gpu(const float* matrix, const unsigned int* spike_ids, const unsigned int* num_spikes, float* result, const unsigned int batchsize, const unsigned int num_cols, const unsigned int max_num_spikes) {
  // extern __shared__ char shared[];
  // float *sum_values = (float *)shared; // TODO think about bank alignment...
  // float *results_local = (float *)sum_values; // TODO think about bank alignment...
  
  // const int num_elements = num_cols*num_rows;
  // const int start_matrix = threadIdx.x + blockIdx.x * num_elements; // assuming one batch per block
  // const int step_matrix =  * gridDim.x;

  // const unsigned int* spike_ids_this_batch = spike_ids + ibatch * max_num_spikes;

  extern __shared__ char shared[];
  unsigned int* spike_ids_this_batch = (unsigned int*)shared;
  
  const int ibatch = blockIdx.x;

  for(unsigned ispk=threadIdx.x;ispk<num_spikes[ibatch];ispk+=blockDim.x) {
    spike_ids_this_batch[ispk] = spike_ids[ibatch*max_num_spikes+ispk];
  }
  __syncthreads();


  for (unsigned icol=threadIdx.x; icol<num_cols; icol+=blockDim.x){
    float sum = 0;
    // // TODO split summation over threads which will require shared memory...
    // for(unsigned ispk=threadIdx.y;ispk<num_spikes[ibatch];ispk+=blockDim.y) { 
    for(unsigned ispk=0;ispk<num_spikes[ibatch];++ispk) {
      sum+= matrix[spike_ids_this_batch[ispk]*num_cols+icol]; // do hierachical summation...
      // sum+= matrix[icol]; // do hierachical summation...
    }
    result[icol+ibatch*num_cols] = sum;
  }
}

// for now one kernel per batch
__global__ void sparse_vals_vector_matmul_gpu(const float* matrix, const float* vals, const unsigned int* spike_ids, const unsigned int* num_spikes, float* result, const unsigned int batchsize, const unsigned int num_cols, const unsigned int max_num_spikes) {
  const int ibatch = blockIdx.x;

  // const float* vals_this_batch = vals + ibatch * max_num_spikes;
  // const unsigned int* spike_ids_this_batch = spike_ids + ibatch * max_num_spikes;
  
  extern __shared__ char shared[];
  float* vals_this_batch = (float*)shared;
  unsigned int* spike_ids_this_batch = (unsigned int*)(&vals_this_batch[max_num_spikes]);
  
  for(unsigned ispk=threadIdx.x;ispk<num_spikes[ibatch];ispk+=blockDim.x) {
    vals_this_batch[ispk] = vals[ibatch*max_num_spikes+ispk];
  }
  for(unsigned ispk=threadIdx.x;ispk<num_spikes[ibatch];ispk+=blockDim.x) {
    spike_ids_this_batch[ispk] = spike_ids[ibatch*max_num_spikes+ispk];
  }  
  __syncthreads();

  for (unsigned icol=threadIdx.x; icol<num_cols; icol+=blockDim.x){
    float sum = 0;
    // // TODO split summation over threads which will require shared memory...
    // for(unsigned ispk=threadIdx.y;ispk<num_spikes[ibatch];ispk+=blockDim.y) { 
    for(unsigned ispk=0;ispk<num_spikes[ibatch];++ispk) {
      sum+= matrix[spike_ids_this_batch[ispk]*num_cols+icol] * vals_this_batch[ispk]; // TODO spikes and vals should be written to shared memory first...
      // sum+= matrix[icol]; // do hierachical summation...
    }
    result[icol+ibatch*num_cols] = sum;
  }
}

__global__ void spike_vector_matmul_matrix_grad_gpu(float* matrix_grad, const unsigned int* spike_ids, const unsigned int* num_spikes, const float* result_grads, const unsigned int batchsize, const unsigned int num_rows, const unsigned int num_cols, const unsigned int max_num_spikes) {

  extern __shared__ char shared[];
  unsigned int* spike_ids_this_batch = (unsigned int*)shared;
  
  const int ibatch = blockIdx.x;

  // for(unsigned i=threadIdx.x+ibatch*blockDim.x;i<num_rows*num_cols;i+=blockDim.x*gridDim.x) {
  //   matrix_grad[i] = 0.f;
  // }

  for(unsigned ispk=threadIdx.x;ispk<num_spikes[ibatch];ispk+=blockDim.x) {
    spike_ids_this_batch[ispk] = spike_ids[ibatch*max_num_spikes+ispk];
  }
  __syncthreads();

  for (unsigned icol=threadIdx.x; icol<num_cols; icol+=blockDim.x){
    float res_grad = result_grads[icol+ibatch*num_cols]; // loads it once to L1 cache
    for(unsigned ispk=0;ispk<num_spikes[ibatch];++ispk) {
      atomicAdd(&matrix_grad[spike_ids_this_batch[ispk]*num_cols+icol], res_grad);
    }
  }

  // TODO just flatten outer product (can be done by max_num_spikes*num_cols in parallel) the operation and use shared memory for both the spike_ids and result_grads
  // TODO when using shared memory for result_grads, might get close on architectures with less L1 cache (RTX3090s and less)
  
  // TODO use warp shuffle to reduce the number of atomicAdd calls ? what does this mean ?!
}


__global__ void spike_vector_grad_gpu(const float* matrix, const unsigned int* spike_ids, const float* spike_grads_precalc_surr, const unsigned int* num_grad_spikes, const float* result_grads, float* spike_grads, const unsigned int batchsize, const unsigned int num_cols, const unsigned int max_num_spikes) {

  // make sure blockDim.x is multiple of 32

  // threadIdx.x threads handle a column of the matrix
  // threadIdx.y threads handle different spike ids
  // blockIdx.x threads handle different batches

  extern __shared__ char shared[];
  float * sum_shared = (float*)shared;

  const unsigned int ibatch = blockIdx.x;
  const unsigned thread_idx = threadIdx.y * blockDim.x + threadIdx.x;


  // if (thread_idx < num_grad_spikes[ibatch]){
  //   sum_shared[thread_idx] = 0.f;
  // }
  // initialize shared memory
  for (unsigned int ispk = thread_idx; ispk < num_grad_spikes[ibatch]; ispk += blockDim.y*blockDim.x) {
    sum_shared[ispk] = 0.f;
  }
  __syncthreads();

  const unsigned int* spike_ids_ibatch = spike_ids + ibatch * max_num_spikes;
  float* spike_grads_ibatch = spike_grads + ibatch * max_num_spikes;
  const float* result_grads_ibatch = result_grads + ibatch * num_cols;
  const float* spike_grads_precalc_surr_ibatch = spike_grads_precalc_surr + ibatch * max_num_spikes;

  const unsigned start_result_idx = threadIdx.x;
  const unsigned step_result_idx = blockDim.x;
  
  for (unsigned int ispk = threadIdx.y; ispk < num_grad_spikes[ibatch]; ispk += blockDim.y) {
    float sum = 0.0f;
    const unsigned int spk_idx = spike_ids_ibatch[ispk];
    const float* matrix_row = matrix + spk_idx * num_cols;
    for (unsigned int i = start_result_idx; i < num_cols; i += step_result_idx) {
      sum += matrix_row[i] * result_grads_ibatch[i];
    }
    if (threadIdx.x < num_cols) {
      // probably slow... better to use parallel threads to sum from separate spikes (as above)
      // doing that would require more shared memory tho (as above)
      atomicAdd(&sum_shared[ispk], sum); 
    }
  }
  // could also write transpose here with shifted acess pattern to simply read in sum operation
  
  __syncthreads();

  // write result to global memory
  for (unsigned int ispk = thread_idx; ispk < num_grad_spikes[ibatch]; ispk += blockDim.y*blockDim.x) {
    spike_grads_ibatch[ispk] = spike_grads_precalc_surr_ibatch[ispk] * sum_shared[ispk];
    // spike_grads[ibatch + thread_idx] = sum_shared[thread_idx];
  }
}