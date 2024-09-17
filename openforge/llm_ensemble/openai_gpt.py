import argparse
import os

# import openai
import pandas as pd

from openai import OpenAI

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import parse_llm_response
from openforge.utils.prior_model_common import log_exp_metrics
from openforge.utils.util import parse_config


def get_gpt35_prediction(
    client,
    model_id: str,
    temperature: float,
    user_prompt: str,
) -> int:
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

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=50,
        logprobs=True,
    )

    pred = parse_llm_response(response.choices[0].message.content)

    return pred


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
    output_dir = config.get("exp", "output_dir")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = create_custom_logger(output_dir)
    logger.info(f"Running program: {__file__}\n")
    logger.info(f"{args}\n")

    printable_config = {section: dict(config[section]) for section in config}
    logger.info(f"Experiment configuration:\n{printable_config}\n")

    data_filepath = config.get("exp", "data_filepath")
    test_df = pd.read_json(data_filepath)
    logger.info(f"\n{test_df.head()}\n")

    model_id = config.get("llm", "model_id")
    temperature = config.getfloat("llm", "temperature")

    if args.mode == "inference":
        all_predictions = []
        all_labels = []
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        for i, row in test_df.iterrows():
            prompt = row["prompt"]

            if model_id == "gpt-3.5-turbo-1106":
                pred = get_gpt35_prediction(
                    client,
                    model_id,
                    temperature,
                    prompt,
                )

            all_predictions.append(pred)
            all_labels.append(int(row["label"]))

            logger.info(f"i={i}: prediction={pred}, label={row['label']}")
            logger.info("-" * 50)

        test_df["prediction"] = all_predictions
        output_filepath = os.path.join(
            output_dir, f"{data_filepath.split('/')[-1]}.json"
        )
        test_df.to_json(
            config.get("exp", "output_filepath"), orient="records", indent=4
        )
    else:
        assert args.mode == "evaluation"

        all_predictions = test_df["prediction"].tolist()
        all_labels = test_df["label"].tolist()

    log_exp_metrics(
        "test", all_labels, all_predictions, logger, multi_class=False
    )
