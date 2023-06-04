import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# Data containing generated texts from PEDL
input_file = '/content/drive/MyDrive/TestSet_Negative.tsv'

# Filter out irrelevant columns
df = pd.read_csv(input_file, sep='\t', header=None, usecols=[1, 3, 5])
# Rename the needed columns
df.columns = ['gene1', 'gene2', 'text']

# Convert placeholders
df['text'] = df['text'].replace('<e[12]>[^<]*</e[12]>', '@GeneOrGeneProduct$', regex=True)

# Apply label 'Association' to all the positive examples
df['label'] = 'None'

output_file = '/content/drive/MyDrive/Data/edited_negative_set.tsv'
df.to_csv(output_file, sep='\t', index=False)