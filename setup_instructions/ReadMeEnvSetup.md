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


# Build the benchmark dataset
python benchmark_builders/BenchmarkDatasetBuilderASP.py
Configuring Ambiguity

By default, the maximum ambiguity level is set to 20. To change this value:

    Open the file:

benchmark_builders/world_descriptors/WorldGendersLocationsNoHornAmbFacts.py

Locate the parameter max_ambiguity and set it to your desired integer value:

max_ambiguity = 20  # change this to your desired level

Re-run the builder script to generate a dataset with the new ambiguity setting.