import pandas as pd

# 定义读取csv文件的函数
def read_csv_file(file_path):
    """
    读取csv文件，并将第一列作为索引
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

# 定义提取相同索引的函数
def extract_common_indexes(df1, df2):
    """
    提取df1和df2中共有的索引
    """
    common_indexes = df1.index.intersection(df2.index)
    df1_common = df1.loc[common_indexes]
    df2_common = df2.loc[common_indexes]
    return df1_common, df2_common

# 定义写入csv文件的函数
def write_csv_file(df, file_path):
    """
    将数据框写入csv文件
    """
    df.to_csv(file_path)

# 定义主函数
def main():
    # 定义文件路径
    file1_path = 'AMPds数据集\Residential_2.csv'
    file2_path = 'AMPds数据集\Residential_4.csv'
    file3_path = 'AMPds数据集\Residential_5.csv'
    file4_path = 'AMPds数据集\Residential_7.csv'
    file5_path = 'AMPds数据集\Residential_8.csv'
    file6_path = 'AMPds数据集\Residential_9.csv'
    file7_path = 'AMPds数据集\Residential_10.csv'
    file8_path = 'AMPds数据集\Residential_11.csv'
    file9_path = 'AMPds数据集\Residential_12.csv'
    file10_path = 'AMPds数据集\Residential_13.csv'
    file11_path = 'AMPds数据集\Residential_14.csv'
    file12_path = 'AMPds数据集\Residential_19.csv'
    file13_path = 'AMPds数据集\Residential_20.csv'
    # 读取csv文件
    df1 = read_csv_file(file1_path)
    df2 = read_csv_file(file2_path)
    df3 = read_csv_file(file3_path)
    df4 = read_csv_file(file4_path)
    df5 = read_csv_file(file5_path)
    df6 = read_csv_file(file6_path)
    df7 = read_csv_file(file7_path)
    df8 = read_csv_file(file8_path)
    df9 = read_csv_file(file9_path)
    df10 = read_csv_file(file10_path)
    df11 = read_csv_file(file11_path)
    df12 = read_csv_file(file12_path)
    df13 = read_csv_file(file13_path)
    # 提取相同索引的部分
    df1_common, df2_common = extract_common_indexes(df1, df2)
    df1_common, df3_common = extract_common_indexes(df1, df3)
    df1_common, df4_common = extract_common_indexes(df1, df4)
    df1_common, df5_common = extract_common_indexes(df1, df5)
    df1_common, df6_common = extract_common_indexes(df1, df6)
    df1_common, df7_common = extract_common_indexes(df1, df7)
    df1_common, df8_common = extract_common_indexes(df1, df8)
    df1_common, df9_common = extract_common_indexes(df1, df9)
    df1_common, df10_common = extract_common_indexes(df1, df10)
    df1_common, df11_common = extract_common_indexes(df1, df11)
    df1_common, df12_common = extract_common_indexes(df1, df12)
    df1_common, df13_common = extract_common_indexes(df1, df13)

    # 将相同索引的部分写入新的csv文件中
    write_csv_file(df2_common, 'data\data4.csv')
    write_csv_file(df3_common, 'data\data5.csv')
    write_csv_file(df4_common, 'data\data7.csv')
    write_csv_file(df5_common, 'data\data8.csv')
    write_csv_file(df6_common, 'data\data9.csv')
    write_csv_file(df7_common, 'data\data10.csv')
    write_csv_file(df8_common, 'data\data11.csv')
    write_csv_file(df9_common, 'data\data12.csv')
    write_csv_file(df10_common, 'data\data13.csv')
    write_csv_file(df11_common, 'data\data14.csv')
    write_csv_file(df12_common, 'data\data19.csv')
    write_csv_file(df13_common, 'data\data20.csv')

# 调用主函数
if __name__ == '__main__':
    main()
