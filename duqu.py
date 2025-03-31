import numpy as np
from pymatgen.core.structure import Structure
import os

#导入npy文件路径位置
test = np.load('a.npy')
print(np.max(test,axis=1))
print(np.sum(test,axis=1))

# quanzhong=test[6].tolist()
# new_quanzhong=[i[0] for i in quanzhong]

# print(new_quanzhong)
# daoshu_l=[]
# crystal = Structure.from_file('contcar240-pouxi/CONTCAR15128')
# for i in range(len(crystal)):
#     print(crystal[i].specie)
# all_nbrs = crystal.get_all_neighbors(8, include_index=True)
# all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
# for i in range(12):
#     print(crystal[all_nbrs[6][i][2]].specie,all_nbrs[6][i][0],all_nbrs[6][i][2])
#     daoshu_l.append(1/all_nbrs[6][i][1])
#
# print(daoshu_l)
# k=np.array(daoshu_l)
# print(k/sum(k))


# #双折线
#
#
# import matplotlib.pyplot as plt
# import random
# import pandas as pd
# import matplotlib as mpl
#
# mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# # 数据准备
# date = ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月"]
# sales = [random.randint(10000, 25000) for i in range(8)]
# cost = [int(i / 100) - random.randint(1, 20) for i in sales]
# df = pd.DataFrame(data={"销量": sales, "成本": cost}, index=date)
#
# # 绘制第一个Y轴
# fig = plt.figure(figsize=(20, 8), dpi=80)
# ax = fig.add_subplot(111)
# lin1 = ax.plot(df.index, df["销量"], marker="o", label="sales")
# ax.set_title("双Y轴图", size=20)
# ax.set_xlabel("时间", size=18)
# ax.set_ylabel("销量(件)", size=18)
# for i, j in df["销量"].items():
#     ax.text(i, j + 20, str(j), va="bottom", ha="center", size=15)
#
# # 绘制另一Y轴
# ax1 = ax.twinx()
# lin2 = ax1.plot(df.index, df["成本"], marker="o", color="red", label="cost")
# ax1.set_ylabel("成本(元)", size=18)
#
# # 合并图例
# lins = lin1 + lin2
# labs = [l.get_label() for l in lins]
# ax.legend(lins, labs, loc="upper left", fontsize=15)
#
# plt.show()