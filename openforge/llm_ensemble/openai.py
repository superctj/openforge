import argparse
import os

import openai
import pandas as pd

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import (
    craft_sotab_user_prompt,
    load_openforge_sotab_benchmark,
    sample_few_shot_examples,
)
from openforge.utils.util import parse_config


def get_gpt35_prediction(user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Your task is to decide whether two data objects (e.g., string, tuple, column, entity, etc.) match each other or semantically equivalent. Return the result in the JSON format: '{'match': true}' or '{'match': false}'.",  # noqa: E501
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=0.7,
    )
    decision = response["choices"][0]["message"]["content"]

    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the experiment configuration file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="evaluation",
        help="Mode of operation. 'inference' will call OpenAI API to obtain predictions while 'evaluation' will evaluate existing predictions.",  # noqa: E501
    )

    args = parser.parse_args()

    # Parse experiment configuration
    config = parse_config(args.config_path)

    # Create logger
    output_dir = config.get("results", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    train_df, valid_df, test_df = load_openforge_sotab_benchmark(
        config.get("benchmark", "data_dir"), logger
    )

    openai.api_key = os.getenv("OPENAI_API_KEY")
    random_seed = config.getint("benchmark", "random_seed")
    model_id = config.get("llm", "model_id")
    num_shots = config.getint("llm", "num_shots")

    all_prompts = []
    all_predictions = []
    all_labels = []

    for i, row in test_df.iterrows():
        few_shot_df = sample_few_shot_examples(
            train_df, n=num_shots, balanced=True, random_seed=random_seed
        )
        prompt = craft_sotab_user_prompt(row, few_shot_df)

        all_prompts.append(prompt)
        all_labels.append(row["relation_variable_label"])

        if i == 0:
            logger.info(f"1st prompt:\n{prompt}")

    df = pd.DataFrame({"prompt": all_prompts, "label": all_labels})
    df.to_json(
        os.path.join(
            output_dir, f"sotab_test-w_sample_values-{num_shots}_shots.json"
        ),
        orient="records",
        lines=True,
    )
