import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import load_logger
class data_info:
    def __init__(self, log_file):
        super(data_info, self).__init__()
        self.logger = load_logger(log_file)
    def describe(self, df:pd.DataFrame()):
        self.logger.info("查看数据基本状态")

        print('{:*^60}'.format('数据前两行：'))
        print(df.head(2))
        '''
        ***************************数据前两行：***************************
           color   director_name  ...  aspect_ratio  movie_facebook_likes
        0  Color   James Cameron  ...          1.78                 33000
        1  Color  Gore Verbinski  ...          2.35                     0
        '''
        print('{:*^60}'.format('数据类型：'))
        print(pd.DataFrame(df.dtypes))

        print('{:*^60}'.format('数据统计描述：'))
        print(df.describe().round(2))
        '''
        **************************数据统计描述：***************************
        num_critic_for_reviews  duration  ...  aspect_ratio  movie_facebook_likes
        count                 5043.00   5043.00  ...       5043.00               5043.00
        mean                   140.19    107.20  ...          2.22               7525.96
        std                    121.00     25.16  ...          1.34              19320.45
        min                      1.00      7.00  ...          1.18                  0.00
        25%                     50.00     93.00  ...          1.85                  0.00
        50%                    111.00    103.00  ...          2.22                166.00
        75%                    194.00    118.00  ...          2.35               3000.00
        max                    813.00    511.00  ...         16.00             349000.00
        '''
        print('{:*^60}'.format('相关性分析：'))
        print(df.corr().round(2).T)
        plt.figure(dpi=800)
        corr = df.corr()
        sns.heatmap(df.corr().round(1), cmap='Reds', annot=True)
        plt.show()
        plt.savefig("corr.png")
        max_abs = corr.apply(np.abs)[corr != 1].max().max()  # 第一个max():所有行中，每行最大取一个；第二个max()从行最大里面取一个最大，就是最强相关
        table_a = corr.apply(np.abs)[corr != 1]

        lst = []  # 自定义列表用来存放相关系数绝对值最大时候对应的行索引和列索引
        for index in table_a.index:  # 遍历 行的索引
            for col in table_a.columns:  # 遍历  列的索引
                if table_a.loc[index, col] == max_abs:  # 条件：相关性系数最大时，对应的行索引和列索引
                    lst.append([index, col])  # 添加到定义的空列表中

        print("相关性最强的是：")
        print(lst)


