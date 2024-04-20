# BRAINBYTE_SEARCH_ENGINE
## Introduction

A free, opensource, and offline research engine based on `BM25` and `BERT` models. 
Built using `PyTorch` and `PyQt6`.

## Usage
### Conda environment

First, install the Python dependencies:

```
conda create -n brainbyte python==3.9
conda activate brainbyte
pip install -r environment.txt
```

### Setting `PYTHONPATH` in Conda Environment

Setting the `PYTHONPATH` environment variable after activating your environment allows Python to find and import modules from directories not installed in the standard library or virtual environment paths.

#### For Unix/Linux/macOS:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/module"
```

#### For Windows:

```cmd
set PYTHONPATH=%PYTHONPATH%;C:\path\to\your\module
```

### To run the UI:

```bash
python search_enging_UI.py
```

Note that you can only input queries listed in `/sample_data/sample_queries.tsv`.
