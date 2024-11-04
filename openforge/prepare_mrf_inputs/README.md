# Notes

## Steps to generate MRF inputs for a data matching dataset with a LLM prior and k-nearest neighbors given by an embedding model
1. Generate predictions and confidence scores for pairs in the original datasets (prior)
    ```
    python em-wa_prior_llm.py --config_path=<>
    ```
2. Create MRFs based on the prior predictions and k-nearest neighbors given by an embedding model
    ```
    python em-wa-create-mrfs_nv-embed-v2.py --config_path=<>
    ```
3. Populate predictions and confidence scores for extrapolated pairs in created MRFs
    ```
    python em-wa-populate-mrfs_llm.py --config_path=<>
    ```
