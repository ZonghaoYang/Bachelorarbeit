from google.colab import drive
drive.mount('/content/drive')

# Remove the suffix from protein isoform's id
def remove_isoform_suffix(uniprot_id: str) -> str:
    return uniprot_id.split("-")[0]

input_file = '/content/drive/MyDrive/Data/negative_examples_uniprotkb.txt'
output_file = '/content/drive/MyDrive/Data/exclude_iso_uniprotkb.txt'

# Read UniProtKB gene pairs from the input file
with open(input_file, "r") as f:
    gene_pairs = [tuple(line.strip().split("\t")) for line in f.readlines()[1:]]

# Remove the isoform suffixes
base_pairs = [(remove_isoform_suffix(gene1), remove_isoform_suffix(gene2)) for gene1, gene2 in gene_pairs]

# Write the base UniProtKB pairs to an output file
with open(output_file, "w") as f:
    f.write("Base1\tBase2\n")
    for gene1, gene2 in base_pairs:
        f.write(f"{gene1}\t{gene2}\n")