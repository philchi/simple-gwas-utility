from cyvcf2 import VCF
import numpy as np

def read_vcf(vcfName):
    return VCF(vcfName)

def get_genotypes(vcf):
    genotypeStack = []
    
    try:
        for _, variant in enumerate(vcf):
            vcfGenotypes = variant.genotypes
            summedGenotypes = [genotype[0] + genotype[1] for genotype in vcfGenotypes]
            genotypeStack.append(summedGenotypes)
    except:
        pass

    genotypes = np.array(genotypeStack).T

    return genotypes

def get_phenotype(vcf, phen):
    with open(phen, "r") as f:
        data = f.read()

    rows = data.split("\n")

    sampleMap = {}
    for row in rows:
        if row == "":
            continue

        cols = row.split("\t")
        sampleMap[cols[0]] = np.float64(cols[2])

    samples = vcf.samples
    phenotype = np.expand_dims(np.array([sampleMap.get(sample, 0) for sample in samples]), axis=1)

    return phenotype

if __name__ == "__main__":
    vcf = read_vcf("data/gwas.vcf")
    phen = get_phenotype(vcf, "data/gwas.phen")
    print(f"phenotype shape: {phen.shape}")
    geno = get_genotypes(vcf)
    print(f"genotypes shape: {geno.shape}")