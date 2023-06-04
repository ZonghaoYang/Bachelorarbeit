from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Define the paths to the datasets containing positive and negative examples
file_path1 = '/content/drive/MyDrive/Data/edited_positive_set.tsv'
file_path2 = '/content/drive/MyDrive/Data/edited_negative_set.tsv'

# Read the files into dataframes
df1 = pd.read_csv(file_path1, sep='\t')
df2 = pd.read_csv(file_path2, sep='\t')

# Combine the dataframes
combined_df = pd.concat([df1, df2])

# Save the combined dataframe to a new file
combined_df.to_csv('/content/drive/MyDrive/TestSet_Combined_unshuffled.tsv', sep='\t', index=False)