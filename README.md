# SparseSpikes

Repository to implement cuda kernels for sparse spikes vectors for matrix multiply. This includes the kernels for forward and gradient calculation using surrogate gradients (forward and backward mode) for:
1. sparse spike vector times dense matrix matmul
2. sparse spike vector generaten from states and sthresholds

The repository structure is as follows:
- `"lib/"` contains cuda kernels and neccessary c interface files
- `"jax_interface"` contains the python interface for jax and some test files 



## Install guide

To use the code, first clode the repository and move into the directory

```
git clone []
cd SparseSPikes
```

Then complie the C and cuda code by running
```
make
```

This will create a new directory called `build`.
In order to use the python functions add the repository to the python path (as we don't yet support pip). THe easiest way is to add the repository path to the PYTHONPATH in the bashrc and sourcing the bashrc:
```
echo 'export PYTHONPATH="'$(pwd)':$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

You need to install jax and jaxlib version 0.4.7 for the GPU. Instructions see at https://github.com/google/jax#installation.
You can directly install jax with the other requirements via:

```
pip install --upgrade pip
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

where the `-f` flag is needed for jax cuda installation. 

Now you can just run the code. Try by running:

```
python3 tests/test.py
```


## Training convergence and accuracy 

If you want to run the example code and SNN training scripts under `examples/accuracy_sweeps` you need to install additional requirements via:

```
pip install -r requirements_examples.txt
pip install tonic
```

## Citation

If you found this repository useful, please cite: ([arxiv link](https://arxiv.org/abs/2311.04386))

```
@article{finkbeiner2023harnessing,
      title={Harnessing Manycore Processors with Distributed Memory for Accelerated Training of Sparse and Recurrent Models}, 
      author={Jan Finkbeiner and Thomas Gmeinder and Mark Pupilli and Alexander Titterton and Emre Neftci},
      year={2023},
      eprint={2311.04386},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

