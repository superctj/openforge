import argparse
import os

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
                "content": system_prompt,
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
    message = response.choices[0].message.content

    if logger:
        # logger.info(f"Response: {response}")
        logger.info(f"Message: {message}")

    pred, confdc_score = parse_llm_response(message, keyword="type")

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
    # if "relation_variable_label" in test_df.columns:
    #     test_df.rename(
    #         columns={"relation_variable_label": "label"}, inplace=True
    #     )

    logger.info(f"\n{test_df.head()}\n")

    model_id = config.get("llm", "model_id")
    max_new_tokens = config.getint("llm", "max_new_tokens")

    all_predictions = []
    class_0_pred_probs = []
    class_1_pred_probs = []
    class_2_pred_probs = []
    all_labels = []

    if args.mode == "inference" or args.mode == "test":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        for i, row in test_df.iterrows():
            logger.info(f"{i+1}/{test_df.shape[0]}:")

            prompt = row["prompt"]
            confdc_score = -1

            count = 0
            while confdc_score <= 0.33 or confdc_score >= 1:
                pred, confdc_score = get_gpt_prediction(
                    client,
                    model_id,
                    user_prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    logger=logger,
                )

                count += 1
                if count >= 3:
                    confdc_score = 0.5
                    break

            all_predictions.append(pred)
            rest_confdc_score = (1 - confdc_score) / 2
            
            if pred == 0:
                class_0_pred_probs.append(confdc_score)
                class_1_pred_probs.append(rest_confdc_score)
                class_2_pred_probs.append(rest_confdc_score)
            elif pred == 1:
                class_0_pred_probs.append(rest_confdc_score)
                class_1_pred_probs.append(confdc_score)
                class_2_pred_probs.append(rest_confdc_score)
            else:
                assert pred == 2

                class_0_pred_probs.append(rest_confdc_score)
                class_1_pred_probs.append(rest_confdc_score)
                class_2_pred_probs.append(confdc_score)

            all_labels.append(int(row["relation_variable_label"]))

            logger.info(f"label: {row['relation_variable_label']}")
            logger.info(f"prediction: {pred}")
            logger.info(f"confidence score: {confdc_score}")
            logger.info("-" * 80)

            if args.mode == "test" and i >= 1:
                exit(0)

        test_df["prediction"] = all_predictions
        test_df["class_0_prediction_probability"] = class_0_pred_probs
        test_df["class_1_prediction_probability"] = class_1_pred_probs
        test_df["class_2_prediction_probability"] = class_2_pred_probs

        output_filename = data_filepath.split("/")[-1].split(".")[0]
        output_filepath = os.path.join(output_dir, f"{output_filename}.json")
        test_df.to_json(output_filepath, orient="records", indent=4)
    else:
        assert args.mode == "evaluation"

        all_predictions = test_df["prediction"].tolist()
        all_labels = test_df["relation_variable_label"].tolist()

    log_exp_metrics(
        output_filename, all_labels, all_predictions, logger, multi_class=True
    )
