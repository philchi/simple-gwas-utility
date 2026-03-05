# Simple GWAS Utility

## Overview

The goal of this project is to implement GWAS as a command-line utility written in Python.
Specifically, it aims to perform GWAS on quantitative traits, such as height, blood pressure,
cholesterol levels, etc. The tool would rely largely on highly optimized Python packages, like
NumPy, to efficiently process vectorized matrix operations. The utility will be benchmarked
against plink to compare their computational efficiency, memory usage, and accuracy.

## Installation Steps

1. Create a Python environment with version `>= 3.11` (older versions may work, but not guaranteed)
> Note: Windows users may need to set up the environment in WSL to avoid errors when installing package cyvcf2
2. Install the required packages with `pip install -r requirements.txt`
3. Run the utility with `python gwas.py`

## Next Steps

0. Complete GWAS utility basic functionality
1. Verify GWAS utility's result with plink's result
2. Optimize GWAS utility's efficiency by vectorizing the loop over SNPs
3. Implement additional advanced features, such as MAF filtering, missingness filtering, assigning principal components as covariates, etc.
4. Create a larger dataset with 1000G chromosome 1 pruned vcf by simulating the phenotypes
5. Evaluate GWAS utility's various performance against plink's performance