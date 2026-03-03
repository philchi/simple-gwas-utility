# Simple GWAS Utility

## Overview

The goal of this project is to implement GWAS as a command-line utility written in Python.
Specifically, it aims to perform GWAS on quantitative traits, such as height, blood pressure,
cholesterol levels, etc. The tool would rely largely on highly optimized Python packages, like
NumPy, to efficiently process vectorized matrix operations. The utility will be benchmarked
against plink to compare their computational efficiency, memory usage, and accuracy.

## Installation Steps

1. Create a Python environment with version `>= 3.11`
> Note: Windows users may need to set up the environment in WSL to avoid errors when installing package cyvcf2
2. Install the required packages with `pip install -r requirements.txt`
3. Run the utility with `python gwas.py`

## Next Steps