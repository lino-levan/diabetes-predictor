'''
Simple script to take all.csv and split it into the different datasets
'''

import pandas as pd
from sklearn.model_selection import train_test_split

all_data = pd.read_csv("data/all.csv")

train, temp = train_test_split(all_data, test_size=0.2)
test, validation = train_test_split(temp, test_size=0.5)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
validation.to_csv("data/validation.csv", index=False)
