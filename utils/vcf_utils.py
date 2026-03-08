from cyvcf2 import VCF
from tqdm import tqdm
import numpy as np

def read_vcf(vcfName: str):
    return VCF(vcfName)

def get_genotypes(vcf, mafHighPass):
    genotypeStack = []
    headers = []
    
    try:
        print("Loading genotypes")
        for variant in tqdm(vcf):
            vcfGenotypes = variant.genotypes
            summedGenotypes = [genotype[0] + genotype[1] for genotype in vcfGenotypes]
            p = np.sum(summedGenotypes) / (len(summedGenotypes) * 2)
            maf = min(p, 1-p)

            if maf >= mafHighPass:
                genotypeStack.append(summedGenotypes)
                headers.append([variant.CHROM, variant.ID, variant.POS])
        
        print("Done")
    except:
        pass

    genotypes = np.array(genotypeStack).T

    return genotypes, headers

def get_phenotype(vcf, phen: str):
    with open(phen, "r") as f:
        data = f.read()

    rows = data.split("\n")

    sampleMap = {}
    for row in rows:
        if row == "":
            continue

        cols = row.split("\t")
        sampleMap[cols[0]] = np.float32(cols[2])

    samples = vcf.samples
    phenotype = np.expand_dims(np.array([sampleMap.get(sample, 0) for sample in samples]), axis=1)

    return phenotype

def get_covars(vcf, eigenvecName: str):
    sampleMap = {}
    with open(eigenvecName, "r") as f:
        for line in f:
            vals = line.split(" ")

            sampleName = vals[0]
            covar = vals[2:]

            sampleMap[sampleName] = covar

    samples = vcf.samples
    covars = np.array([sampleMap.get(sample, 0) for sample in samples], dtype=np.float32)

    return covars

if __name__ == "__main__":
    vcf = read_vcf("data/gwas.vcf")
    phen = get_phenotype(vcf, "data/gwas.phen")
    print(f"phenotype shape: {phen.shape}")
    geno = get_genotypes(vcf)
    print(f"genotypes shape: {geno.shape}")