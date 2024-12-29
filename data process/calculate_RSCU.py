###############################calculate CDS coden RSCU########################################


from collections import defaultdict
import numpy as np
from Bio import SeqIO

# definate all possible codons and their coreesponding amino acids
CODON_TABLE = {
    'F': ['TTT', 'TTC'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'I': ['ATT', 'ATC', 'ATA'],
    'M': ['ATG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'Y': ['TAT', 'TAC'],
    'H': ['CAT', 'CAC'],
    'Q': ['CAA', 'CAG'],
    'N': ['AAT', 'AAC'],
    'K': ['AAA', 'AAG'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'C': ['TGT', 'TGC'],
    'W': ['TGG'],  # TGA is stop codon，not part of W
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    '*': ['TAA', 'TAG', 'TGA']  #stop codons
}

# generate a list of all codons
ALL_CODONS = [codon for codons in CODON_TABLE.values() for codon in codons]

# count the frequency of codons
def count_codon_usage(sequence):
    codon_counts = defaultdict(int)
    total_codons = 0

    
    sequence = sequence.upper()

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in ALL_CODONS:
            codon_counts[codon] += 1
            total_codons += 1
    return codon_counts, total_codons

# calculate relative synonymous codon usage
def calculate_rscu(codon_counts):
    rscu_values = {}
    for aa, codons in CODON_TABLE.items():
        if aa == '*':  # ignore stop codons
            continue
        total_for_aa = sum(codon_counts.get(codon, 0) for codon in codons)
        num_codons = len(codons)
        for codon in codons:
            if total_for_aa > 0:
                rscu = (codon_counts.get(codon, 0) / total_for_aa) * num_codons
                rscu_values[codon] = rscu
            else:
                rscu_values[codon] = 0
    return rscu_values

# convert result to feature matrix
def generate_feature_vector(rscu_values):
    return [rscu_values.get(codon, 0) for codon in ALL_CODONS]

# main function: process each contig and aggregate codon counts to calculate RSCU
def process_fasta_file_with_aggregated_rscu(fasta_file, output_file):
    contig_data = defaultdict(lambda: defaultdict(int))  

    for record in SeqIO.parse(fasta_file, "fasta"):
        contig_id = record.id.split('_')[0]
        sequence = str(record.seq)
        codon_counts, total_codons = count_codon_usage(sequence)
        for codon, count in codon_counts.items():
            contig_data[contig_id][codon] += count
    aggregated_results = []
    
    for contig_id, codon_counts in contig_data.items():
        
        rscu_values = calculate_rscu(codon_counts)

        
        feature_vector = generate_feature_vector(rscu_values)

        
        aggregated_results.append([contig_id] + feature_vector)

    
    with open(output_file, "w") as out_file:
        header = [' '] + ALL_CODONS
        out_file.write(",".join(header) + "\n")
        for row in aggregated_results:
            out_file.write(",".join(map(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x), row)) + "\n")

    print(f"save as {output_file}")


if __name__ == "__main__":
    fasta_file = "C:/Users/ASUS/Desktop/meta/deep-learning/RSCU/test_contig.fasta"  
    output_file = "C:/Users/ASUS/Desktop/meta/deep-learning/RSCU/test_2.txt"  
    process_fasta_file_with_aggregated_rscu(fasta_file, output_file)
