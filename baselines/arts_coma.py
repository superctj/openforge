import argparse
import os

import pandas as pd

from valentine import valentine_match
from valentine.algorithms import Coma

from openforge.ARTS.helpers.mongodb_helper import readCSVFileWithTableID
from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.prior_model_common import log_exp_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data",
        type=str,
        default="/ssd/congtj/openforge/arts/artifact/top_40_nodes/training_prop_0.5/openforge_arts_test.csv",  # noqa: 501
        help="Path to the source data.",
    )

    parser.add_argument(
        "--use_instance",
        type=int,
        default=0,
        help="Whether to use data instances in COMA matcher.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold to accept a match.",
    )

    parser.add_argument(
        "--num_rows",
        type=int,
        default=10000,
        help="Maximum number of rows to read from source tables.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/arts/coma",
        help="Directory to save logs.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = create_custom_logger(args.log_dir)
    logger.info(f"Running program: {__file__}")
    logger.info(f"{args}\n")

    df = pd.read_csv(args.source_data, delimiter=",", header=0)
    matcher = Coma(use_instances=bool(args.use_instance), java_xmx="16g")

    y_true, y_pred = [], []

    for row in df.itertuples():
        table1_id = row.concept_1_table_id
        table2_id = row.concept_2_table_id

        table1 = readCSVFileWithTableID(table1_id, nrows=args.num_rows)
        table1 = table1.astype(str)
        table2 = readCSVFileWithTableID(table2_id, nrows=args.num_rows)
        table2 = table2.astype(str)

        column1 = table1[row.concept_1_col_name].to_frame()
        column2 = table2[row.concept_2_col_name].to_frame()

        column1.columns = [row.concept_1]
        column2.columns = [row.concept_2]

        matches = valentine_match(column1, column2, matcher)

        if matches:
            assert len(matches) == 1

            for key in matches:
                if matches[key] > args.threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
        else:
            y_pred.append(0)

        y_true.append(row.relation_variable_label)

    log_exp_metrics("test", y_true, y_pred, logger)
