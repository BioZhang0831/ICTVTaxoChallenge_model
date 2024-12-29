########################## Calculate TNF ####################################
import os
from os.path import abspath, dirname, join
import numpy as np
import itertools
from scipy.linalg import null_space
import csv
import argparse
import logging

def reverse_complement(nuc):
    table = str.maketrans("ACGT", "TGCA")
    return nuc[::-1].translate(table)

def all_kmers(k):
    return [''.join(p) for p in itertools.product("ACGT", repeat=k)]

def create_projection_kernel(indexof):
    linear_equations = list()

    # Constraint 1: Frequency sums to zero
    linear_equations.append([1] * 256)

    # Constraint 2: k-mer frequency equals its reverse complement frequency
    for kmer in all_kmers(4):
        revcomp = reverse_complement(kmer)

        # Only process canonical k-mer
        if kmer >= revcomp:
            continue

        line = [0] * 256
        line[indexof[kmer]] = 1
        line[indexof[revcomp]] = -1
        linear_equations.append(line)

    # Constraint 3: Prefix frequency equals suffix frequency
    for trimer in all_kmers(3):
        line = [0] * 256
        for suffix in "ACGT":
            line[indexof[trimer + suffix]] += 1
        for prefix in "ACGT":
            line[indexof[prefix + trimer]] += -1
        linear_equations.append(line)

    # Convert to NumPy array
    linear_equations = np.array(linear_equations)

    # Compute null space (kernel space)
    kernel = null_space(linear_equations).astype(np.float32)
    assert kernel.shape == (256, 103), f"Unexpected kernel shape: {kernel.shape}"
    return kernel

def create_rc_kernel(indexof):
    rc_matrix = np.zeros((256, 256), dtype=np.float32)
    for col, kmer in enumerate(all_kmers(4)):
        revcomp = reverse_complement(kmer)
        rc_matrix[indexof[kmer], col] += 0.5
        rc_matrix[indexof[revcomp], col] += 0.5
    return rc_matrix

def create_dual_kernel(indexof):
    return np.dot(create_rc_kernel(indexof), create_projection_kernel(indexof))

def save_kernel(dual_kernel, path):
    np.savez_compressed(path, dual_kernel)
    print(f'Kernel matrix saved to {path}')

def extract_kmer_frequencies(fasta_path, output_csv, skipped_log):
    indexof = {kmer: i for i, kmer in enumerate(all_kmers(4))}
    kmers = all_kmers(4)

    # Initialize frequency dictionary
    freq_dict = {}

    # Initialize skipped k-mer record
    skipped_kmers = {}

    with open(fasta_path, 'r') as f:
        contig_name = None
        sequence = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if contig_name and sequence:
                    freq, skipped = process_sequence(sequence, indexof)
                    freq_dict[contig_name] = freq
                    skipped_kmers[contig_name] = skipped
                contig_name = line[1:].split()[0]
                sequence = ''
            else:
                sequence += line.upper()
        # Process the last contig
        if contig_name and sequence:
            freq, skipped = process_sequence(sequence, indexof)
            freq_dict[contig_name] = freq
            skipped_kmers[contig_name] = skipped

    # Get all contig names
    contig_names = list(freq_dict.keys())

    # Write to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the first row: blank cell + k-mer names
        writer.writerow(['Contig'] + kmers)
        # Write the frequencies for each contig
        for contig in contig_names:
            writer.writerow([contig] + freq_dict[contig])

    print(f'k-mer frequency matrix saved to {output_csv}')

    # Write skipped k-mer log
    with open(skipped_log, 'w') as logf:
        for contig, skipped in skipped_kmers.items():
            if skipped:
                logf.write(f'Contig: {contig}\n')
                for kmer in skipped:
                    logf.write(f'  Skipped k-mer: {kmer}\n')
    print(f'Skipped k-mers logged to {skipped_log}')

def process_sequence(sequence, indexof):
    kmers = all_kmers(4)
    counts = np.zeros(256, dtype=np.int32)
    skipped = set()
    valid_bases = set('ACGT')

    for i in range(len(sequence) - 3):
        kmer = sequence[i:i+4]
        if set(kmer).issubset(valid_bases):
            counts[indexof[kmer]] += 1
        else:
            skipped.add(kmer)

    total = counts.sum()
    if total > 0:
        tnf = counts / total
    else:
        tnf = counts.astype(float)
    # Shift TNF so the sum equals zero
    tnf_shifted = tnf - (1.0 / 256)

    return tnf_shifted.tolist(), skipped

def main():
    parser = argparse.ArgumentParser(description='Calculate k-mer frequency matrix and generate projection kernel matrix.')
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file path')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--skipped_log', type=str, default='skipped_kmers.log', help='Log file path for skipped k-mers')
    parser.add_argument('--kernel_path', type=str, default='kernel.npz', help='Output kernel.npz file path')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(filename='kmer_analysis.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')

    # Extract k-mer frequencies and save to CSV
    extract_kmer_frequencies(args.fasta, args.output_csv, args.skipped_log)

    # Create indexof mapping
    indexof = {kmer: i for i, kmer in enumerate(all_kmers(4))}

    # Generate dual kernel
    dual_kernel = create_dual_kernel(indexof)

    # Save kernel.npz
    save_kernel(dual_kernel, args.kernel_path)

if __name__ == "__main__":
    main()
