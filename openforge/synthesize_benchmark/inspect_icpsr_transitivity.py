import argparse
import csv
import os

import pandas as pd

# from openforge.utils.custom_logging import create_custom_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data_dir",
        type=str,
        default="/ssd/congtj/openforge/icpsr",
        help="Path to the source data directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ssd/congtj/openforge/icpsr/artifact",
        help="Directory to save outputs.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/icpsr",
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # logger = create_custom_logger(args.log_dir)
    # logger.info(f"Running program: {__file__}\n")
    # logger.info(f"{args}\n")

    subject_terms_filepath = os.path.join(
        args.source_data_dir, "subject_terms.xlsx"
    )
    relation_filepath = os.path.join(
        args.source_data_dir, "term_relations.xlsx"
    )

    subject_terms = pd.read_excel(subject_terms_filepath)
    term_relations = pd.read_excel(relation_filepath)

    # logger.info(f"\nSubject terms: {subject_terms.head()}")
    # logger.info(f"\nTerm relations: {term_relations.head()}\n")

    concept_ids = subject_terms["TERM_ID"].to_list()
    id_term_map = {}

    for row in subject_terms.itertuples():
        id_term_map[row.TERM_ID] = row.TERM

    left_table = term_relations[["SUBJECT_ID", "RELATIONSHIP", "OBJECT_ID"]]
    right_table = term_relations[["SUBJECT_ID", "RELATIONSHIP", "OBJECT_ID"]]

    join_table = left_table.join(
        right_table.set_index("SUBJECT_ID"),
        on="OBJECT_ID",
        lsuffix="_left",
        rsuffix="_right",
    )

    # logger.info(f"\nJoin table: {join_table.head()}")
    # logger.info(f"\nJoin table columns: {join_table.columns}")
    output_filepath = os.path.join(
        args.output_dir, "hypernymy_transitivity.csv"
    )
    hyper_concepts = set()

    with open(output_filepath, "w") as f:
        field_names = [
            "concept_1_id",
            "concept_2_id",
            "concept_3_id",
            "concept_1",
            "concept_2",
            "concept_3",
        ]
        csv_writer = csv.DictWriter(f, fieldnames=field_names)
        csv_writer.writeheader()

        for row in join_table.itertuples():
            if row.RELATIONSHIP_left == 1 and row.RELATIONSHIP_right == 1:
                concept_1 = id_term_map.get(row.SUBJECT_ID)
                concept_2 = id_term_map.get(row.OBJECT_ID_left)
                concept_3 = id_term_map.get(row.OBJECT_ID_right)

                csv_writer.writerow(
                    {
                        "concept_1_id": row.SUBJECT_ID,
                        "concept_2_id": row.OBJECT_ID_left,
                        "concept_3_id": row.OBJECT_ID_right,
                        "concept_1": concept_1,
                        "concept_2": concept_2,
                        "concept_3": concept_3,
                    }
                )

                hyper_concepts.add(concept_1)
                hyper_concepts.add(concept_2)
                hyper_concepts.add(concept_3)

    print(f"Number of hypernymy concepts: {len(hyper_concepts)}")

    # join_table = left_table.join(
    #     right_table.set_index("SUBJECT_ID"),
    #     on="SUBJECT_ID",
    #     lsuffix="_left",
    #     rsuffix="_right",
    # )

    # logger.info(f"\nJoin table: {join_table.head()}")
    # logger.info(f"\nJoin table columns: {join_table.columns}")

    # for row in join_table.itertuples():
    #     if row.RELATIONSHIP_left == 4 and row.RELATIONSHIP_right == 4:
    #         concept_1 = id_term_map.get(row.SUBJECT_ID)
    #         concept_2 = id_term_map.get(row.OBJECT_ID_left)
    #         concept_3 = id_term_map.get(row.OBJECT_ID_right)

    #         logger.info(f"{concept_1} = {concept_2} = {concept_3}")
