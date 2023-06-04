from google.colab import drive
import pandas as pd

drive.mount('/content/drive')

# Define the file paths in your Google Drive
input_file_path = "/content/drive/MyDrive/Data/canonical_ncbi_id.tsv"
output_file_path = "/content/drive/MyDrive/Data/unique_canonical_ncbi_id.tsv"

# Read the text file into a pandas DataFrame
data = pd.read_csv(input_file_path, sep='\t', names=['From', 'To'])

# Drop the duplicates in the 'From' column and keep the first occurrence
unique_data = data.drop_duplicates(subset='From', keep='first')

# Save the cleaned data to a new text file
unique_data.to_csv(output_file_path, sep='\t', index=False, header=False)

# Read the cleaned data file and the gene pairs file into pandas DataFrames
cleaned_data = pd.read_csv(output_file_path, sep='\t', names=['UniProtKB', 'NCBI'])
gene_pairs = pd.read_csv("/content/drive/MyDrive/Data/exclude_iso_uniprotkb.txt", sep='\t', names=['Base1', 'Base2'])

# Merge the DataFrames on UniProtKB ids
merged_data1 = gene_pairs.merge(cleaned_data, left_on='Base1', right_on='UniProtKB', how='left')
merged_data2 = gene_pairs.merge(cleaned_data, left_on='Base2', right_on='UniProtKB', how='left')

# Combine the two merged DataFrames
combined_data = merged_data1.copy()
combined_data['NCBI_2'] = merged_data2['NCBI']

# Drop the rows with missing NCBI ids
filtered_data = combined_data.dropna(subset=['NCBI', 'NCBI_2'])

# Drop the UniProtKB id columns and rename the columns to 'gene1' and 'gene2'
mapped_data = pd.DataFrame()
mapped_data['gene1'] = filtered_data['NCBI']
mapped_data['gene2'] = filtered_data['NCBI_2']

# Save the resulting DataFrame with mapped NCBI ids to a new file
mapped_data.to_csv("/content/drive/MyDrive/Data/filtered_mapped_canonical_ncbi.txt", sep='\t', index=False, header=True)