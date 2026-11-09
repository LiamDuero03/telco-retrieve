# Installation

From a clone of this repository:

```bash
git clone https://github.com/Znbne/telco-retrieve
cd telco-retrieve
pip install -e ".[local-embed,pdf,docx]"
```

## NVIDIA, AMD ROCm or CPU: install PyTorch first

The `local-embed` extra depends on `torch`. If you install the package first, pip pulls the default PyTorch build from
PyPI, which on Linux is a CUDA build of several GB. That is the wrong build on an AMD GPU and a waste on a machine
without a GPU, and replacing it afterwards means a forced reinstall. So install PyTorch **first**, from the index that
matches your hardware, then the package, then check:

```bash
# 0. Update pip first
pip install --upgrade pip

# 1. PyTorch first. The index URL names the build; take the exact name for your CUDA or ROCm version from
#    https://pytorch.org/get-started/locally/ (the version numbers below are examples)
pip install torch --index-url https://download.pytorch.org/whl/cu128      # NVIDIA, CUDA 12.8 (cu126, cu130, ... to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2    # AMD, ROCm 7.2 (the index must match the ROCm on the host, see below)
pip install torch --index-url https://download.pytorch.org/whl/cpu        # no GPU

# 2. then the package: it finds torch already installed and leaves it alone
pip install -e ".[local-embed,pdf,docx]"

# 3. check which build you have
python -c "import torch; print('gpu:', torch.cuda.is_available(), '| cuda:', torch.version.cuda, '| hip:', torch.version.hip)"
```

Run `pip install --upgrade pip` before any torch install, then only the line of step 1 that matches your machine. NVIDIA prints `gpu: True | cuda: 12.x | hip: None`. An AMD ROCm
build also answers `torch.cuda.is_available()`, so it prints `gpu: True | cuda: None | hip: 6.x`. A CPU build prints
`gpu: False | cuda: None | hip: None`. If you have a GPU and see `gpu: False`, or the wrong one of `cuda` and `hip`, the wrong
build is installed: run the matching line of step 1 again with `--force-reinstall`.

**ROCm: match the index to the host.** `rocm-smi` and `rocminfo` show the ROCm version and the GPU of the machine. Install the PyTorch build with the same ROCm number: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/rocm<version>`. The PyTorch page above lists which build suits which GPU.

## API keys

Create a `.env` file in your project root:

```bash
OPENROUTER_API_KEY=your_key_here      # for LLM generation via OpenRouter (Llama, Claude, GPT-4o, etc.)
ANTHROPIC_API_KEY=your_key_here       # optional — for direct Claude access
DATALAB_API_KEY=your_key_here         # optional — for Marker PDF preprocessing
```
