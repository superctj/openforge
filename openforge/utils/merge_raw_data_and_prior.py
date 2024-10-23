import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_data_filepath", type=str, required=True)
    parser.add_argument("--prior_data_filepath", type=str, required=True)

    args = parser.parse_args()

    raw_df = pd.read_csv(args.raw_data_filepath)
    prior_df = pd.read_csv(args.prior_data_filepath)

    lid = []
    rid = []

    for i, row in raw_df.iterrows():
        lid.append(row["ltable_id"])
        rid.append(row["rtable_id"])

    prior_df["l_id"] = lid
    prior_df["r_id"] = rid

    prior_df.rename(
        columns={"entity_1": "l_entity", "entity_2": "r_entity"}, inplace=True
    )
    prior_df.to_csv(args.prior_data_filepath, index=False)
