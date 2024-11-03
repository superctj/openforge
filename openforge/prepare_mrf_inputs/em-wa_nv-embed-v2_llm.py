"""
Each local MRF is created based on the nv-embed-v2 embedding model and a LLM is
used as a prior model.

The program assumes MRFs have already been created and only invokes a LLM to
make predictions.
"""

import argparse
import os

import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import parse_llm_response
from openforge.utils.util import parse_config


ENTITY_MATCHING_INSTRUCTION = """Entity matching is the task of determining whether two instances refer to the same real-world entity. For the following pair of entities, please determine if they refer to the same real-world entity. Return your prediction and confidence score in the following JSON format: '{"match": <true or false>, "confidence score": <Confidence score needs to be greater than 0.5 and smaller than 1.>}'. Give only the prediction and the confidence score. No explanation is needed.\n
"""  # noqa: E501


def get_llm_prediction(
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int = 20,
    device: str = "cuda",
    logger=None,
) -> int:
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    formatted_messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_messages, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )

    # Only decode the part of the output that is generated by the model
    response = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]

    if logger:
        logger.info(f"Prompt: {user_prompt}")
        logger.info(f"Response: {response}")

    pred, confdc_score = parse_llm_response(response, keyword="match")

    return pred, confdc_score


def run_prior_inference(
    model,
    tokenizer,
    max_new_tokens: int,
    num_retries: int,
    input_dir: str,
    output_dir: str,
    device,
):
    for f in os.listdir(input_dir):
        if f.endswith(".json"):
            input_df = pd.read_json(os.path.join(input_dir, f))
            input_df = input_df.drop(
                columns=["prior_prediction", "prior_confidence_score"]
            )

            all_predictions = []
            all_confdc_scores = []

            for i, row in input_df.iterrows():
                prompt = (
                    ENTITY_MATCHING_INSTRUCTION
                    + f"Input:\nInstance 1: {row['l_entity']}\nInstance 2: {row['r_entity']}\n\nOutput:"  # noqa: E501
                )

                confdc_score = -1
                num_attempts = 0

                while confdc_score < 0.5 or confdc_score >= 1:
                    pred, confdc_score = get_llm_prediction(
                        model, tokenizer, prompt, max_new_tokens, device, logger
                    )

                    num_attempts += 1
                    if num_attempts >= num_retries:
                        confdc_score = 0.6
                        break

                all_predictions.append(pred)
                all_confdc_scores.append(confdc_score)

                logger.info(f"prediction: {pred}")
                logger.info(f"confidence score: {confdc_score}")
                logger.info("-" * 80)
                # if i >= 2:  # for testing
                #     exit(0)

            input_df["prior_prediction"] = all_predictions
            input_df["prior_confidence_score"] = all_confdc_scores

            output_filepath = os.path.join(output_dir, f)
            input_df.to_json(output_filepath, orient="records", indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the experiment configuration file",
    )

    args = parser.parse_args()

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

    input_dir = config.get("io", "input_dir")
    valid_input_dir = os.path.join(input_dir, "validation")
    test_input_dir = os.path.join(input_dir, "test")
    valid_output_dir = os.path.join(output_dir, "validation")
    if not os.path.exists(valid_output_dir):
        os.makedirs(valid_output_dir)
    test_output_dir = os.path.join(output_dir, "test")
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    model_id = config.get("prior", "model_id")
    max_new_tokens = config.getint("prior", "max_new_tokens")
    num_retries = config.getint("prior", "num_retries")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", trust_remote_code=True
    )
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    run_prior_inference(
        model,
        tokenizer,
        max_new_tokens,
        num_retries,
        valid_input_dir,
        valid_output_dir,
        device,
    )
    run_prior_inference(
        model,
        tokenizer,
        max_new_tokens,
        num_retries,
        test_input_dir,
        test_output_dir,
        device,
    )
