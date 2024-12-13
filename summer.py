def find_max_c_index(filename):
    max_c_index = -1
    max_line_number = -1
    current_line_number = 1

    with open(filename, 'r') as file:
        for line in file:
            if 'val c-index:' in line:
                c_index_str = line.split(': ')[1]
                c_index = float(c_index_str)
                if c_index > max_c_index:
                    max_c_index = c_index
                    max_line_number = current_line_number
            current_line_number += 1

    return max_c_index, max_line_number


# 使用脚本
# filename = r'C:\Users\Obliviate\PycharmProjects\pathomic-balance-fusion-survival\results\WiKG_coattn\mcat_coattn\tcga_luad_s1\4\log.txt'
# max_c_index, max_line_number = find_max_c_index(filename)
# print(f'最大的c-index是 {max_c_index}，在第 {max_line_number} 行')

# MCAT
# fold 0 最大的c-index是 0.6824，在第 33 行
# fold 1 最大的c-index是 0.6774，在第 6 行
# fold 2 最大的c-index是 0.6816，在第 3 行
# fold 3 最大的c-index是 0.6999，在第 7 行
# fold 4 最大的c-index是 0.6516，在第 24 行
# WiKG readout
# fold 0 最大的c-index是 0.6837，在第 75 行
# fold 1 最大的c-index是 0.6917，在第 120 行
# fold 2 最大的c-index是 0.665，在第 49 行
# fold 3 最大的c-index是 0.6978，在第 65 行
# fold 4 最大的c-index是 0.637，在第 185 行
# MOTCat
# fold 0 最大的c-index是 0.6932，在第 23 行
# fold 1 最大的c-index是 0.6678，在第 6 行
# fold 2 最大的c-index是 0.6854，在第 6 行
# fold 3 最大的c-index是 0.7041，在第 32 行
# fold 4 最大的c-index是 0.6548，在第 13 行
# MOTCat WiKG 01
# fold 0 最大的c-index是 0.6629，在第 6 行
# fold 1 最大的c-index是 0.6664，在第 7 行
# fold 2 最大的c-index是 0.6925，在第 4 行
# fold 3 最大的c-index是 0.7062，在第 40 行
# fold 4 最大的c-index是 0.6580，在第 40 行
# MOTCat WiKG 02
# fold 0 最大的c-index是 0.6521，在第 6 行
# fold 1 最大的c-index是 0.6733，在第 7 行
# fold 2 最大的c-index是 0.6944，在第 3 行
# fold 3 最大的c-index是 0.7062，在第 46 行
# fold 4 最大的c-index是 0.6548，在第 41 行

import numpy as np

# MCAT数据
mcat_c_indexes = [0.6824, 0.6774, 0.6816, 0.6999, 0.6516]

# WiKG readout数据
mcat_wikg_c_indexes = [0.6837, 0.6917, 0.665, 0.6978, 0.637]

# MOTCat数据
motcat_c_indexes = [0.6932, 0.6678, 0.6854, 0.7041, 0.6548]

# MOTCat WiKG 01数据
motcat_wikg01_c_indexes = [0.6629, 0.6664, 0.6925, 0.7062, 0.6580]

# MOTCat WiKG TOP6数据
motcat_wikg_top6_c_indexes = [0.6521, 0.6733, 0.6944, 0.7062, 0.6548]

# MOTCat WiKG TOP10数据
motcat_wikg_top10_c_indexes = [0.6723, 0.6712, 0.6944, 0.6825, 0.6580]

# MOTCat WiKG TOP12数据
motcat_wikg_top12_c_indexes = [0.6712, 0.6712, 0.6963, 0.7062, 0.6446]

# MOTCat WiKG TOP24数据
motcat_wikg_top24_c_indexes = [0.6698, 0.6821, 0.6861, 0.7069, 0.5518]

# 计算平均值
mean_mcat = np.mean(mcat_c_indexes)
mean_wikg = np.mean(mcat_wikg_c_indexes)
mean_motcat = np.mean(motcat_c_indexes)
motcat_wikg01 = np.mean(motcat_wikg01_c_indexes)
motcat_wikg_top6 = np.mean(motcat_wikg_top6_c_indexes)
motcat_wikg_top10 = np.mean(motcat_wikg_top10_c_indexes)
motcat_wikg_top12 = np.mean(motcat_wikg_top12_c_indexes)
motcat_wikg_top24 = np.mean(motcat_wikg_top24_c_indexes)

# 计算方差（使用NumPy的var函数，它默认计算的是样本方差，即除以n-1）
std_mcat = np.std(mcat_c_indexes, ddof=1)
std_mcat_wikg = np.std(mcat_wikg_c_indexes, ddof=1)
std_motcat = np.std(motcat_c_indexes, ddof=1)
std_motcat_wikg01 = np.std(motcat_wikg01_c_indexes, ddof=1)
std_motcat_wikg_top6 = np.std(motcat_wikg_top6_c_indexes, ddof=1)
std_motcat_wikg_top10 = np.std(motcat_wikg_top10_c_indexes, ddof=1)
std_motcat_wikg_top12 = np.std(motcat_wikg_top12_c_indexes, ddof=1)
std_motcat_wikg_top24 = np.std(motcat_wikg_top24_c_indexes, ddof=1)

# 打印结果
print(f"MCAT的平均值: {mean_mcat}")  # 0.67858
print(f"MCAT的方差: {std_mcat}")
print(f"MOTCat的平均值: {mean_motcat}")  # 0.681
print(f"MOTCat的方差: {std_motcat}")
print(f"MOTCat WiKG top24的平均值: {motcat_wikg_top24}")  # 0.6772
print(f"MOTCat WiKG 24的方差: {std_motcat_wikg_top24}")


def draw(filename):
    import matplotlib.pyplot as plt

    # 假设log.txt文件位于当前工作目录中
    # filename = 'log.txt'

    # 用于存储c-index值的列表
    c_index_values = []

    # 打开文件并读取内容
    with open(filename, 'r') as file:
        for line in file:
            # 查找包含'val c-index:'的行
            if 'val c-index:' in line:
                # 提取数值部分（假设数值是行中'val c-index:'之后的唯一内容）
                # 使用split()方法分割字符串，并取最后一个元素（即数值）
                # 然后使用float()将其转换为浮点数
                c_index_value = float(line.split(':')[-1].strip())
                c_index_values.append(c_index_value)

    # x轴的值（数据点的索引）
    x = list(range(len(c_index_values)))

    # 创建折线图
    plt.plot(x, c_index_values, marker='o', linestyle='-', color='b', label='c-index')

    # 添加标题和标签
    plt.title('c-index Over Time')
    plt.xlabel('Index')
    plt.ylabel('c-index')

    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形


# file = r'C:\Users\Obliviate\Desktop\motcat_coattn\tcga_luad_s1\4\log.txt'
# draw(filename=file)
