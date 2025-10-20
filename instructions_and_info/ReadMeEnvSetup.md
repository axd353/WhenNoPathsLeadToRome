# README

## Environment Setup

To set up the Python environment for this project, follow these steps:

1. Ensure you're using Python 3:
   ```bash
   alias python='python3'

2. pip install --upgrade pip setuptools wheel

3. pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.6.0+cu118

4. pip install \
  torch-scatter==2.1.2 \
  --find-links https://data.pyg.org/whl/torch-2.6.0+cu118.html

5. pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.tx


