#split the data set into two parts
#ex. training 80% of data, testing 20% of data
#we do this because we need to test data that the model hasn't seen before

import pandas as pd
import numpy as np
import os

os.chdir("E:\Gun's stuff\Machine Learning\training and testing")

df = pd.read_csv('carprices.csv')

print(df.head)