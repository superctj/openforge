import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_filepath",
        type=str,
        required=True,
        help="Path to the MRF input file.",
    )

    args = parser.parse_args()

    df = pd.read_json(args.input_filepath)

    for _, row in df.iterrows():
        if row["prediction"] != row["label"] or row["confidence_score"] < 1:
            print(f"Variable name: {row['random_variable_name']}")
            print(f"Prediction: {row['prediction']} and label: {row['label']}")
            print(f"Confidence score: {row['confidence_score']}")
            print("-" * 80)
