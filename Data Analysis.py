# -*- coding: utf-8 -*-
# @Time    : 2022/1/30 19:06
# @Author  : Justus
# @FileName: Data Analysis.py
# @Software: PyCharm

import json
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

path1 = r'ch02/usagov_bitly_data2012-03-16-1331923249.txt'
recodes = [json.loads(line) for line in open(path1)]
print(recodes[0]["tz"])
print(type(recodes[0]["tz"]))
print("最常出现的时区(tz字段)")
time_zones = [rec["tz"] for rec in recodes if "tz" in rec]
print(time_zones)
# 利用DataFrame进行计数
frame = DataFrame(recodes)
print(type(frame["tz"][:10]))
tz_counts = frame["tz"].value_counts()
print(tz_counts[:10])
clean_tz = frame['tz'].fillna("Missing")
clean_tz[clean_tz == ''] = "Unknown"
tz_counts = clean_tz.value_counts()
print(type(tz_counts))
tz_counts[:10].plot(kind='barh', rot=0)
# plt.show()
results = Series([x.split()[0] for x in frame.a.dropna()])
print(results)
print(results.value_counts()[:8])
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
print(operating_system)
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])
indexer = agg_counts.sum(1).argsort()
print(indexer[:10])
count_subset = agg_counts.take(indexer)[-10:]
print(count_subset)
a = np.array([1.2, 3.4, 5.6])
print(np.modf(a))

