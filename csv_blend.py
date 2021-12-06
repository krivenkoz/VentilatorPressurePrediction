import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

# CSV names
path_to_csv = 'e:\\Krivenko\\Kaggle\\2021\\New20211005\\csv\\'
csv_names = glob.glob(path_to_csv + '*.csv')

# Weights
weights = np.ones((len(csv_names))) / len(csv_names)
#weights = np.array([0.03, 0.03, 0.03, 0.91])

# Read all CSV
all_csv_df = []
for csv_name in csv_names:
    csv = pd.read_csv(csv_name)
    all_csv_df.append(csv)

# Unchangeble column's name
to_drop = ['id']
columns_to_change = [col for col in all_csv_df[0].columns if col not in to_drop ]

#
target_df = all_csv_df[0]
target_df[columns_to_change] = weights[0] * target_df[columns_to_change]
for i, csv_df in enumerate(all_csv_df[1:]):
    aa = weights[i+1]
    target_df[columns_to_change] += weights[i+1] * csv_df[columns_to_change]

target_df.to_csv('submission.csv', index=False)