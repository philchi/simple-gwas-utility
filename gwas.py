import argparse
import os
import numpy as np
from cyvcf2 import VCF
from tqdm import tqdm
from scipy import stats
from utils.vcf_utils import read_vcf, get_genotypes, get_phenotype, get_covars

HEADERS = ["ID", "BETA", "T-STAT", "P-VALUE"]

def run_gwas(genotypes, phenotype, covars=None):
    """
    genotypes: (num samples x 1)
    phenotype: (num samples x 1)
    covars: (num samples x num covars)
    """
    if covars is not None:
        X = np.column_stack((np.ones(genotypes.shape), genotypes, covars))
    else:
        X = np.column_stack((np.ones(genotypes.shape), genotypes))

    y = phenotype

    coeffs, residuals, rank, _ = np.linalg.lstsq(X, y)

    if residuals.size == 0:
        return np.nan, np.nan, np.nan

    effectSize = coeffs[1]
    dof = y.shape[0] - rank

    mse = residuals[0] / dof
    effectSizeVar = mse * (np.linalg.pinv(X.T @ X)[1,1])
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
        default="gwas_out",
        help="Prefix for the output results files"
    )

    processing_group = parser.add_argument_group("Processing/Filtering Options")

    processing_group.add_argument(
        "--covar",
        type=str,
        help="Path to plink eigenvec file from pca"
    )

    processing_group.add_argument(
        "--maf",
        type=float,
        default=0,
        help="MAF threshold for filtering rare variants"
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()
    
    # load io args
    vcfName = args.vcf
    phenName = args.pheno
    outName = args.out

    # load processing args
    covarName = args.covar
    maf = args.maf

    # load genotypes, phenotype, and covariates
    vcf = read_vcf(vcfName)

    genotypes, ids = get_genotypes(vcf, maf)
    phenotype = get_phenotype(vcf, phenName)
    covars = get_covars(vcf, covarName) if covarName else None

    vcf.close()

    os.makedirs("output", exist_ok=True)

    # run gwas
    with open(f"output/{outName}", "w+") as f:
        f.write("\t".join(HEADERS)+"\n")
        
        index = 0
        for i in tqdm(range(genotypes.shape[1])):
            effectSize, tStat, pVal = run_gwas(genotypes[:, index:index+1], phenotype[:, 0:1], covars)
            f.write(f"{ids[i]}\t{effectSize}\t{tStat}\t{pVal}\n")
            index += 1


if __name__ == "__main__":
    main()