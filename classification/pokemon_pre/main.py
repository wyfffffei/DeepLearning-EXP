import numpy as np
import pandas as pd

pokemon_df = pd.read_csv("./pokemon.csv").set_index('#')
combats_df = pd.read_csv("./combats.csv")

# 检查数据缺失情况
# print(pokemon_df.info())
# print(pokemon_df["Type 2"].value_counts(dropna=False))

# 填充缺失数据
pokemon_df["Type 2"].fillna("empty", inplace=True)

# 检查数据类型
print(pokemon_df.dtypes)
print('-' * 30)
print(combats_df.dtypes)
