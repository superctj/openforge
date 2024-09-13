import os

import pandas as pd


def count_num_specs(specs_path: str) -> int:
    num_specs = 0

    for source in os.listdir(specs_path):
        source_path = os.path.join(specs_path, source)

        for f in os.listdir(source_path):
            if f.endswith(".json"):
                num_specs += 1

    return num_specs


def count_domain_num_specs(domain_path: str) -> int:
    num_specs = 0

    for f in os.listdir(domain_path):
        if f.endswith(".json"):
            num_specs += 1

    return num_specs


def count_num_matches(ground_truth_path: str) -> int:
    num_matches = 0
    df = pd.read_csv(ground_truth_path, sep=",")

    # Deduplicate the dataframe
    print(df.head())
    df = df.drop_duplicates()
    print(df.head())
    # Group by 'entity_id' and count the number of rows in each group
    group_counts = df.groupby(["entity_id"]).size()
    print(group_counts.head())

    for count in group_counts:
        num_matches += (count - 1) * count // 2

    return num_matches


def count_domain_num_matches(ground_truth_path: str, domain: str) -> int:
    num_matches = 0
    df = pd.read_csv(ground_truth_path, sep=",")

    # Deduplicate the dataframe
    df = df.drop_duplicates()
    # Filter by 'domain' in the column 'spec_id'
    df = df[df["spec_id"].str.startswith(domain)]
    print(df.head())

    # Group by 'entity_id' and count the number of rows in each group
    group_counts = df.groupby(["entity_id"]).size()
    print(group_counts.head())

    for count in group_counts:
        num_matches += (count - 1) * count // 2

    return num_matches


if __name__ == "__main__":
    specs_path = "/ssd2/congtj/openforge/DI2KG/2013_monitor_specs"
    ground_truth_path = (
        "/ssd2/congtj/openforge/DI2KG/monitor_entity_resolution_gt(in).csv"
    )
    domain = "www.ebay.com"

    num_specs = count_num_specs(specs_path)
    print(f"Number of specifications: {num_specs}")

    num_matches = count_num_matches(ground_truth_path)
    print(f"Number of matches: {num_matches}")

    domain_path = os.path.join(specs_path, domain)
    domain_num_specs = count_domain_num_specs(domain_path)
    print(
        f"Number of specifications in the domain '{domain}': {domain_num_specs}"
    )

    domain_num_matches = count_domain_num_matches(ground_truth_path, domain)
    print(f"Number of matches in the domain '{domain}': {domain_num_matches}")
