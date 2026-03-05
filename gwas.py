import numpy as np
from cyvcf2 import VCF
from scipy import stats

def run_gwas(genotypes, phenotype, covars):
    """
    genotypes: (num samples x 1)
    phenotype: (num samples x 1)
    covars: (num samples x num covars)
    """
    X = np.column_stack((np.ones(genotypes.shape), genotypes, covars))
    y = phenotype

    coeffs, residuals, rank, _ = np.linalg.lstsq(X, y)

    effectSize = coeffs[0]
    dof = y.shape[0] - rank - 1

    mse = residuals[0] / dof
    effectSizeVar = mse * (np.linalg.inv(X.T @ X)[1,1])
    coeffStdErr = np.sqrt(effectSizeVar)
    
    tStat = effectSize / coeffStdErr
    pVal = stats.t.sf(np.abs(tStat), dof) * 2

    return effectSize, pVal