# GNN Inference (Paper Artifact)

This repository accompanies the paper in `Degree-Based Scheduling and Memory Management for Large-Scale Exact Online GNN Inference`. The change set to the DGL repository lives in `DGL modifications`, which needs to be applied to an upstream DGL checkout and built from source.

## Apply the DGL modifications
- Clone DGL: `git clone https://github.com/dmlc/dgl.git <path-to-dgl>`
- Copy the patched files from this repo into the cloned DGL repo, replacing the originals (adjust paths if the DGL layout changes):
  ```bash
  DGL_ROOT=<path-to-dgl>
  cp "DGL modifications/array.cc"                "$DGL_ROOT/src/array/array.cc"
  cp "DGL modifications/array_index_select_uvm.cu"  "$DGL_ROOT/src/array/array_index_select_uvm.cu"
  cp "DGL modifications/array_index_select_uvm.cuh" "$DGL_ROOT/src/array/array_index_select_uvm.cuh"
  cp "DGL modifications/uvm_array_op.h"          "$DGL_ROOT/src/array/uvm_array_op.h"
  cp "DGL modifications/rowwise_sampling.cu"     "$DGL_ROOT/src/graph/sampling/rowwise_sampling.cu"
  cp "DGL modifications/neighbor_sampler.py"     "$DGL_ROOT/python/dgl/sampling/neighbor_sampler.py"
  cp "DGL modifications/pin_memory.py"           "$DGL_ROOT/python/dgl/sampling/pin_memory.py"
  ```

## Build DGL from source
- Install build prerequisites per the DGL docs (CMake, a C++ toolchain, Python, PyTorch, and optionally CUDA/CUDNN).
- From the DGL repo:
  ```bash
  cd "$DGL_ROOT"
  mkdir -p build && cd build
  cmake -DUSE_CUDA=ON ..    # drop USE_CUDA or set OFF if building CPU-only
  make -j"$(nproc)"
  cd ..
  pip install -e python     # installs the patched DGL into your Python environment
  ```

Once built and installed, use the patched DGL in your experiments; all scripts in this repository expect that environment.

## Running the experiments
- From the repo root, run the `find_*` scripts first, then the `lin_*` scripts, and then the `*_load.py` scripts; these prepare and load results before diving into specific experiments.
- After those complete, enter the subdirectories (e.g., `ablation_study`, `e2e`, `experimental_analysis`, `overhead`, `sensitivity_analysis`, etc.) to run the experiments reported in the paper.

## Script naming quick guide
- `_B`: baseline
- `_P`: proposed method (GSM)
- `_O`: full knowledge scheduler
- `_M`, `_S`: ablations from the paper
