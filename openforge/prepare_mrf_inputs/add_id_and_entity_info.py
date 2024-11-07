import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--entity_info_filepath", type=str, required=True)
    parser.add_argument("--prior_data_filepath", type=str, required=True)

    args = parser.parse_args()

    entity_df = pd.read_json(args.entity_info_filepath)
    prior_df = pd.read_json(args.prior_data_filepath)

    l_ids = entity_df["l_id"].values.tolist()
    r_ids = entity_df["r_id"].values.tolist()
    l_entities = entity_df["l_entity"].values.tolist()
    r_entities = entity_df["r_entity"].values.tolist()

    prior_df["l_id"] = l_ids
    prior_df["r_id"] = r_ids
    prior_df["l_entity"] = l_entities
    prior_df["r_entity"] = r_entities

    prior_df.to_json(args.prior_data_filepath, orient="records", indent=4)
