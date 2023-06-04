import pandas as pd

data = pd.read_csv('mapped_protein_links.txt', sep='\t')

# Exclude columns containing scores that are not knowledge-based
data = data.drop(columns=['textmining', 'database', 'combined_score'])

data.to_csv('data_without_textmining_database.txt', sep='\t', index=False)

from google.colab import files
files.download("data_without_textmining_database.txt")

data = pd.read_csv('data_without_textmining_database.txt', sep='\t')

# Method provided by STRING:
prior = 0.041

def compute_prior_away(score, prior):
    if score < prior: score = prior
    score_no_prior = (score - prior) / (1 - prior)
    return score_no_prior


def compute_combined_score(row):
    # divide by 1000
    neighborhood = row['neighborhood'] / 1000
    fusion = row['fusion'] / 1000
    cooccurence = row['cooccurence'] / 1000
    coexpression = row['coexpression'] / 1000
    experimental = row['experimental'] / 1000

    # compute prior away
    neighborhood_prior_corrected = compute_prior_away(neighborhood, prior)
    fusion_prior_corrected = compute_prior_away(fusion, prior)
    cooccurence_prior_corrected = compute_prior_away(cooccurence, prior)
    coexpression_prior_corrected = compute_prior_away(coexpression, prior)
    experimental_prior_corrected = compute_prior_away(experimental, prior)

    ## next, do the 1 - multiplication:
    combined_score_one_minus = (
        (1.0 - neighborhood_prior_corrected) *
        (1.0 - fusion_prior_corrected) *
        (1.0 - cooccurence_prior_corrected) *
        (1.0 - coexpression_prior_corrected) *
        (1.0 - experimental_prior_corrected))

    ## and lastly, do the 1 - conversion again, and put back the prior *exactly once*
    combined_score = (1.0 - combined_score_one_minus)
    combined_score *= (1.0 - prior)
    combined_score += prior

    
    return int(combined_score * 1000)


data['combined_score'] = data.apply(compute_combined_score, axis=1)

data.to_csv('data_with_new_combined_score.txt', sep='\t', index=False)

from google.colab import files
files.download("data_with_new_combined_score.txt")