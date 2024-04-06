import argparse
import os

import pandas as pd

from valentine import valentine_match
from valentine.algorithms import Coma

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.prior_model_common import log_exp_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data",
        type=str,
        default="/ssd/congtj/openforge/sotab_v2/artifact/sotab_v2_test_openforge_large/test.csv",  # noqa: 501
        help="Path to the source data.",
    )

    parser.add_argument(
        "--use_instance",
        type=int,
        default=1,
        help="Whether to use data instances in COMA matcher.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold to accept a match.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/home/congtj/openforge/logs/sotab_v2/coma",
        help="Directory to store logs.",
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
        table1_path = row.label_1_table_path
        table2_path = row.label_2_table_path

        table1 = pd.read_json(table1_path, compression="gzip", lines=True)
        table1 = table1.astype(str)
        table2 = pd.read_json(table2_path, compression="gzip", lines=True)
        table2 = table2.astype(str)

        column1 = table1.iloc[:, row.label_1_col_idx].to_frame()
        column2 = table2.iloc[:, row.label_2_col_idx].to_frame()

        column1.columns = [row.label_1_processed]
        column2.columns = [row.label_2_processed]

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
