##########################计算TNF####################################
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

    # 约束 1: 频率和为零
    linear_equations.append([1] * 256)

    # 约束 2: kmer 与其反向互补频率相等
    for kmer in all_kmers(4):
        revcomp = reverse_complement(kmer)

        # 仅处理 canonical kmer
        if kmer >= revcomp:
            continue

        line = [0] * 256
        line[indexof[kmer]] = 1
        line[indexof[revcomp]] = -1
        linear_equations.append(line)

    # 约束 3: 前缀频率等于后缀频率
    for trimer in all_kmers(3):
        line = [0] * 256
        for suffix in "ACGT":
            line[indexof[trimer + suffix]] += 1
        for prefix in "ACGT":
            line[indexof[prefix + trimer]] += -1
        linear_equations.append(line)

    # 转换为 NumPy 数组
    linear_equations = np.array(linear_equations)

    # 计算零空间（核空间）
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
    print(f'核矩阵已保存到 {path}')

def extract_kmer_frequencies(fasta_path, output_csv, skipped_log):
    indexof = {kmer: i for i, kmer in enumerate(all_kmers(4))}
    kmers = all_kmers(4)

    # 初始化频率字典
    freq_dict = {}

    # 初始化跳过的 kmer 记录
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
        # 处理最后一个 contig
        if contig_name and sequence:
            freq, skipped = process_sequence(sequence, indexof)
            freq_dict[contig_name] = freq
            skipped_kmers[contig_name] = skipped

    # 获取所有 contig名称
    contig_names = list(freq_dict.keys())

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入第一行：空白单元格 + kmer名称
        writer.writerow(['Contig'] + kmers)
        # 写入每个 contig 的频率
        for contig in contig_names:
            writer.writerow([contig] + freq_dict[contig])

    print(f'k-mer 频率矩阵已保存到 {output_csv}')

    # 写入跳过的 kmer 日志
    with open(skipped_log, 'w') as logf:
        for contig, skipped in skipped_kmers.items():
            if skipped:
                logf.write(f'Contig: {contig}\n')
                for kmer in skipped:
                    logf.write(f'  Skipped kmer: {kmer}\n')
    print(f'跳过的 k-mer 已记录到 {skipped_log}')

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
    # 将 TNF 移动，使其总和为零
    tnf_shifted = tnf - (1.0 / 256)

    return tnf_shifted.tolist(), skipped

def main():
    parser = argparse.ArgumentParser(description='计算 k-mer 频率矩阵和生成投影核矩阵。')
    parser.add_argument('--fasta', type=str, required=True, help='输入 FASTA 文件路径')
    parser.add_argument('--output_csv', type=str, required=True, help='输出 CSV 文件路径')
    parser.add_argument('--skipped_log', type=str, default='skipped_kmers.log', help='记录跳过 k-mer 的日志文件路径')
    parser.add_argument('--kernel_path', type=str, default='kernel.npz', help='输出 kernel.npz 文件路径')

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(filename='kmer_analysis.log', level=logging.INFO, 
                        format='%(asctime)s %(levelname)s:%(message)s')

    # 提取 k-mer 频率并保存到 CSV
    extract_kmer_frequencies(args.fasta, args.output_csv, args.skipped_log)

    # 创建 indexof 映射
    indexof = {kmer: i for i, kmer in enumerate(all_kmers(4))}

    # 生成 dual kernel
    dual_kernel = create_dual_kernel(indexof)

    # 保存 kernel.npz
    save_kernel(dual_kernel, args.kernel_path)

if __name__ == "__main__":
    main()