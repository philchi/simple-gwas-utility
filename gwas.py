import argparse
import numpy as np
from cyvcf2 import VCF
from scipy import stats
from utils.vcf_utils import read_vcf, get_genotypes, get_phenotype

def run_gwas(genotypes, phenotype, covars=None):
    """
    genotypes: (num samples x 1)
    phenotype: (num samples x 1)
    covars: (num samples x num covars)
    """
    if covars:
        X = np.column_stack((np.ones(genotypes.shape), genotypes, covars))
    else:
        X = np.column_stack((np.ones(genotypes.shape), genotypes))

    y = phenotype

    coeffs, residuals, rank, _ = np.linalg.lstsq(X, y)

    effectSize = coeffs[1]
    dof = y.shape[0] - rank

    mse = residuals[0] / dof
    effectSizeVar = mse * (np.linalg.inv(X.T @ X)[1,1])
    coeffStdErr = np.sqrt(effectSizeVar)
    
    tStat = effectSize / coeffStdErr
    pVal = stats.t.sf(np.abs(tStat), dof) * 2

    return effectSize, tStat, pVal

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simple GWAS Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    io_group = parser.add_argument_group("Input/Output Options")
    
    io_group.add_argument(
        "--vcf",
        type=str,
        required=True,
        help="Path to the input VCF file (can be .vcf or .vcf.gz)"
    )
    io_group.add_argument(
        "--pheno",
        type=str,
        required=True,
        help="Path to the phenotype file"
    )
    io_group.add_argument(
        "--out",
        type=str,
        default="pygwas_results",
        help="Prefix for the output results files"
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    
    vcfName = args.vcf
    phenName = args.pheno

    vcf = read_vcf(vcfName)

    genotypes = get_genotypes(vcf)
    phenotype = get_phenotype(vcf, phenName)

    vcf.close()

    index = 0
    for _ in range(3):
        effectSize, tStat, pVal = run_gwas(genotypes=genotypes[:, index:index+1], phenotype=phenotype[:, 0:1])
        print(f"effect size: {effectSize}, t-stat: {tStat}, p value: {pVal}")
        index += 1
    

if __name__ == "__main__":
    main()