# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 18:10:28 2023

@author: coope
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\yelp_dataset\\total_data.csv'
out_path_binary = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\Raw Data\\Yelp Data\\binary.csv'
out_path_nominal = 'C:\\Users\\coope\\OneDrive\\Desktop\\Side_Projects\\Raw Data\\Yelp Data\\nominal.csv'

df = pd.read_csv(path, index_col = 0)


plt.hist(df["stars"])
plt.show() 

pd.crosstab(index = df["stars"], columns = "prop") / pd.crosstab(index = df["stars"], columns = "prop").sum()
pd.crosstab(index = df["stars"], columns = "prop")


# Data set under assumption that 1, 2, and 3 are bad, 4 and 5 are good
df["binary"] = ["Positive" if x >= 4 else "Negative" for x in df["stars"]]
#binary

pd.crosstab(index = df["binary"], columns = "prop") / pd.crosstab(index = df["binary"], columns = "prop").sum()
df.to_csv(out_path_binary, columns = ["text", "binary"], index = False)


# Data set under assumption that 1 and 2 are bad, 4 and 5 are good, and 3 is Neutral
df["nominal"] = ["Positive" if x >= 4 else "Negative" if x <= 2 else "Neutral" for x in df["stars"]]
pd.crosstab(index = df["nominal"], columns = "prop") / pd.crosstab(index = df["nominal"], columns = "prop").sum()

df.to_csv(out_path_nominal, columns = ["text", "nominal"], index = False)




## Most fair - sample down 5 levels so they are all equal to each other

df["stars"]