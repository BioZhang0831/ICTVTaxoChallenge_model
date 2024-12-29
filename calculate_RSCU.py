###############################统计CDS密码子RSCU########################################


from collections import defaultdict
import numpy as np
from Bio import SeqIO

# 定义所有可能的密码子及其对应的氨基酸 (标准遗传密码表)
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
    'W': ['TGG'],  # TGA 是终止密码子，不属于 W
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    '*': ['TAA', 'TAG', 'TGA']  # 终止密码子
}

# 生成所有密码子列表
ALL_CODONS = [codon for codons in CODON_TABLE.values() for codon in codons]

# 统计密码子频率
def count_codon_usage(sequence):
    codon_counts = defaultdict(int)
    total_codons = 0

    # 将序列转换为大写
    sequence = sequence.upper()

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in ALL_CODONS:
            codon_counts[codon] += 1
            total_codons += 1
    return codon_counts, total_codons

# 计算相对同义密码子使用度 (RSCU)
def calculate_rscu(codon_counts):
    rscu_values = {}
    for aa, codons in CODON_TABLE.items():
        if aa == '*':  # 忽略终止密码子
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

# 将结果转换为特征矩阵
def generate_feature_vector(rscu_values):
    return [rscu_values.get(codon, 0) for codon in ALL_CODONS]

# 主函数：处理每个 contig 并汇总密码子计数后计算 RSCU
def process_fasta_file_with_aggregated_rscu(fasta_file, output_file):
    contig_data = defaultdict(lambda: defaultdict(int))  # 存储每个 contig 的密码子计数

    for record in SeqIO.parse(fasta_file, "fasta"):
        contig_id = record.id.split('_')[0]

        sequence = str(record.seq)

        # 统计密码子频率
        codon_counts, total_codons = count_codon_usage(sequence)

        # 累加密码子计数到 contig
        for codon, count in codon_counts.items():
            contig_data[contig_id][codon] += count

    # 计算 RSCU 并生成特征向量
    aggregated_results = []
    for contig_id, codon_counts in contig_data.items():
        # 计算 RSCU
        rscu_values = calculate_rscu(codon_counts)

        # 转换为特征向量
        feature_vector = generate_feature_vector(rscu_values)

        # 添加到结果
        aggregated_results.append([contig_id] + feature_vector)

    # 保存到输出文件
    with open(output_file, "w") as out_file:
        header = [' '] + ALL_CODONS
        out_file.write(",".join(header) + "\n")
        for row in aggregated_results:
            out_file.write(",".join(map(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x), row)) + "\n")

    print(f"加权聚合特征已保存到 {output_file}")

# 使用方法
if __name__ == "__main__":
    fasta_file = "C:/Users/ASUS/Desktop/meta/deep-learning/RSCU/test_contig.fasta"  # 替换为您的输入文件路径
    output_file = "C:/Users/ASUS/Desktop/meta/deep-learning/RSCU/test_2.txt"  # 输出文件路径
    process_fasta_file_with_aggregated_rscu(fasta_file, output_file)