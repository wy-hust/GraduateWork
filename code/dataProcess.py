import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

# 定义读取csv文件的函数
def read_csv_file(file_path):
    """
    读取csv文件，并将第三列作为数据
    """
    df = pd.read_csv(file_path, usecols=[2],skip_blank_lines=False,nrows=14401)
    df.fillna(method='ffill', inplace=True)
    return df

# 定义计算皮尔逊系数相关性的函数
def calculate_pearson_correlation(df1, df2):
    """
    计算df1和df2的皮尔逊系数相关性
    """
    correlation, _ = pearsonr(df1.iloc[:, 0], df2.iloc[:, 0])
    return correlation

# 定义计算斯皮尔曼系数相关性的函数
def calculate_spearman_correlation(df1, df2):
    """
    计算df1和df2的斯皮尔曼系数相关性
    """
    correlation, _ = spearmanr(df1.iloc[:, 0], df2.iloc[:, 0])
    return correlation

# 定义计算最大互信息系数的函数
def calculate_max_mutual_information(df1, df2):
    """
    计算df1和df2的最大互信息系数
    """
    x = np.array(df1.iloc[:, 0]).reshape(-1, 1)
    y = np.array(df2.iloc[:, 0]).reshape(-1, 1)
    mi = mutual_info_regression(x, y)
    max_mi = np.max(mi)
    return max_mi

# 定义主函数
def main():
    # 定义文件路径
    file_paths = ['data\data2.csv', 'data\data3.csv', 'data\data4.csv', 'data\data5.csv', 'data\data7.csv',
                   'data\data8.csv', 'data\data9.csv', 'data\data10.csv', 'data\data11.csv', 'data\data12.csv', 
                  'data\data13.csv', 'data\data14.csv', 'data\data19.csv','data\data20.csv']

    # 读取csv文件，并将第三列作为数据
    dfs = [read_csv_file(file_path) for file_path in file_paths]

    for i in range(len(dfs)):
        print(len(dfs[i].iloc[:,0]))
    # 计算皮尔逊系数相关性、斯皮尔曼系数相关性和最大互信息系数
    pearson_correlations = np.zeros((len(dfs), len(dfs)))
    spearman_correlations = np.zeros((len(dfs), len(dfs)))
    max_mutual_information = np.zeros((len(dfs), len(dfs)))
    for i in range(len(dfs)):
        for j in range(len(dfs)):
            pearson_correlations[i, j] = calculate_pearson_correlation(dfs[i], dfs[j])
            spearman_correlations[i, j] = calculate_spearman_correlation(dfs[i], dfs[j])
            max_mutual_information[i, j] = calculate_max_mutual_information(dfs[i], dfs[j])

    # 展示结果
    print("Pearson Correlation:")
    for i in range(len(dfs)):
        print("",end=" ")
        print(i,end=" ")
    print()
    for i in range(len(dfs)):
        print(i,end=" ")
        for j in range(len(dfs)):
            print(f"{pearson_correlations[i, j]:.2f}",end=" ")
        print()    

    print("Spearman Correlation: ")
    for i in range(len(dfs)):
        print("",end=" ")
        print(i,end=" ")
    print()
    for i in range(len(dfs)):
        print(i,end=" ")
        for j in range(len(dfs)):
            print(f"{spearman_correlations[i, j]:.2f}",end=" ")
        print()   

    print("max_mutual_information: ")
    for i in range(len(dfs)):
        print("",end=" ")
        print(i,end=" ")
    print()
    for i in range(len(dfs)):
        print(i,end=" ")
        for j in range(len(dfs)):
            print(f"{max_mutual_information[i, j]:.2f}",end=" ")
        print()    
    

if __name__ == '__main__':
    main()
