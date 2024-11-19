import argparse
import os

# import openai
import pandas as pd

from openai import OpenAI

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import parse_llm_response
from openforge.utils.prior_model_common import log_exp_metrics
from openforge.utils.util import parse_config


def get_gpt_prediction(
    client,
    model_id: str,
    user_prompt: str,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
    logger=None,
) -> int:
    if not system_prompt:
        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": system_prompt,  # "Your task is to decide whether two data objects (e.g., string, tuple, column, entity, etc.) match each other or semantically equivalent. Return the result in the JSON format: '{'match': true}' or '{'match': false}'.",  # noqa: E501
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_new_tokens,
    )

    if user_prompt.startswith("Semantic column types"):
        pred, confdc_score = parse_llm_response(response, keyword="equivalent")
    else:
        pred, confdc_score = parse_llm_response(
            response.choices[0].message.content, keyword="match"
        )

    if logger:
        logger.info(f"Response: {response}")

    return pred, confdc_score


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
        help="Mode of operation. 'inference' will incur model API to obtain predictions; 'test' will incur model API up to 3 times for testing purpose; 'evaluation' will evaluate existing predictions.",  # noqa: E501
    )

    args = parser.parse_args()

    # Parse experiment configuration
    config = parse_config(args.config_path)

    # Create logger
    output_dir = config.get("io", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    data_filepath = config.get("io", "input_filepath")
    test_df = pd.read_json(data_filepath)
    logger.info(f"\n{test_df.head()}\n")

    model_id = config.get("llm", "model_id")
    max_new_tokens = config.getint("llm", "max_new_tokens")

    if args.mode == "inference" or args.mode == "test":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        all_predictions = []
        all_labels = []

        for i, row in test_df.iterrows():
            logger.info(f"{i+1}/{test_df.shape[0]}:")

            prompt = row["prompt"]
            confdc_score = -1

            count = 0
            while confdc_score <= 0.5 or confdc_score >= 1:
                pred, confdc_score = get_gpt_prediction(
                    client,
                    model_id,
                    user_prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    logger=logger,
                )

                count += 1
                if count >= 3:
                    confdc_score = 0.6
                    break

            all_predictions.append(pred)
            all_labels.append(int(row["label"]))

            logger.info(f"prediction={pred}, label={row['label']}")
            logger.info("-" * 80)

            if args.mode == "test" and i >= 2:
                exit(0)

        test_df["prediction"] = all_predictions
        output_filename = data_filepath.split("/")[-1].split(".")[0]
        output_filepath = os.path.join(output_dir, f"{output_filename}.json")
        test_df.to_json(output_filepath, orient="records", indent=4)
    else:
        assert args.mode == "evaluation"

        all_predictions = test_df["prediction"].tolist()
        all_labels = test_df["label"].tolist()

    log_exp_metrics(
        output_filename, all_labels, all_predictions, logger, multi_class=False
    )
