import argparse
import os

import pandas as pd

from openforge.feature_extraction.fb_fasttext import FasttextTransformer
from openforge.feature_extraction.qgram import QGramTransformer
from openforge.feature_extraction.similarity_metrics import (
    cosine_similarity,
    edit_distance,
    jaccard_index,
)
from openforge.synthesize_benchmark.prepare_icpsr_hyper_hypo_relations import (
    DataEntry,
    get_concept_signatures,
)


def augment_data(
    df: pd.DataFrame,
    qgram_transformer: object,
    fasttext_transformer: object,
    output_dir: str,
    split: str,
) -> None:
    data_entries = []

    for i, row in df.iterrows():
        concept_1 = row["concept_2"]
        concept_2 = row["concept_1"]
        orig_var_name = row["relation_variable_name"]
        orig_label = row["relation_variable_label"]

        concept_1_name_qgram_signature, concept_1_name_fasttext_signature = (
            get_concept_signatures(
                concept_1, qgram_transformer, fasttext_transformer
            )
        )
        concept_2_name_qgram_signature, concept_2_name_fasttext_signature = (
            get_concept_signatures(
                concept_2, qgram_transformer, fasttext_transformer
            )
        )

        name_qgram_sim = jaccard_index(
            concept_1_name_qgram_signature, concept_2_name_qgram_signature
        )
        name_jaccard_sim = jaccard_index(
            set(concept_1.split()), set(concept_2.split())
        )
        name_edit_dist = edit_distance(concept_1, concept_2)
        name_fasttext_sim = cosine_similarity(
            concept_1_name_fasttext_signature,
            concept_2_name_fasttext_signature,
        )
        name_word_count_ratio = len(concept_1.split()) / len(concept_2.split())
        name_char_count_ratio = len(concept_1) / len(concept_2)

        i, j = orig_var_name.split("_")[1].split("-")
        var_name = f"R_{j}-{i}"
        if orig_label == 0:
            var_label = 0
        elif orig_label == 1:
            var_label = 2
        elif orig_label == 2:
            var_label = 1
        else:
            raise ValueError(f"Invalid relation variable label: {orig_label}")

        data_entries.append(
            DataEntry(
                concept_1=concept_1,
                concept_2=concept_2,
                name_qgram_similarity=name_qgram_sim,
                name_jaccard_similarity=name_jaccard_sim,
                name_edit_distance=name_edit_dist,
                name_fasttext_similarity=name_fasttext_sim,
                name_word_count_ratio=name_word_count_ratio,
                name_char_count_ratio=name_char_count_ratio,
                relation_variable_label=var_label,
                relation_variable_name=var_name,
            )
        )

        aug_df = pd.DataFrame(data_entries)
        aug_df = pd.concat([df, aug_df], ignore_index=True)

        output_filepath = os.path.join(
            output_dir, f"openforge_icpsr_hyper_hypo_{split}_augmented.csv"
        )
        aug_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment ICPSR hyper hypo data"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/ssd/congtj/openforge/icpsr/artifact",
        help="Directory containing the ICPSR hyper hypo data",
    )

    parser.add_argument(
        "--fasttext_model_dir",
        type=str,
        default="/ssd/congtj",
        help="Directory containing fasttext model weights.",
    )

    args = parser.parse_args()

    qgram_transformer = QGramTransformer(qgram_size=3)
    fasttext_transformer = FasttextTransformer(
        cache_dir=args.fasttext_model_dir
    )

    # Load the data
    train_df = pd.read_csv(
        os.path.join(args.data_dir, "openforge_icpsr_hyper_hypo_training.csv")
    )
    valid_df = pd.read_csv(
        os.path.join(args.data_dir, "openforge_icpsr_hyper_hypo_validation.csv")
    )
    test_df = pd.read_csv(
        os.path.join(args.data_dir, "openforge_icpsr_hyper_hypo_test.csv")
    )

    augment_data(
        train_df,
        qgram_transformer,
        fasttext_transformer,
        args.data_dir,
        "training",
    )

    augment_data(
        valid_df,
        qgram_transformer,
        fasttext_transformer,
        args.data_dir,
        "validation",
    )

    augment_data(
        test_df,
        qgram_transformer,
        fasttext_transformer,
        args.data_dir,
        "test",
    )
