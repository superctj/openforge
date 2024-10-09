import argparse
import os

import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import parse_llm_response
from openforge.utils.prior_model_common import log_exp_metrics
from openforge.utils.util import parse_config


def get_llm_prediction(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 50,
    device: str = "cuda",
    logger=None,
) -> int:
    if not system_prompt:
        messages = [
            {
                "role": "user",
                "content": user_prompt,
            },
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

    formatted_messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_messages, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            # temperature=0.1,
        )

    # Only decode the part of the output that is generated by the model
    response = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :]
    )[0]
    pred = parse_llm_response(response)

    if logger:
        logger.info(f"Response: {response}")

    return pred


def get_prediction_from_pipeline(
    model_id: str,
    tokenizer,
    user_prompt: str,
    device: str,
    logger=None,
) -> int:
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    pipe = pipeline(
        "text-generation", model=model_id, tokenizer=tokenizer, device=device
    )
    outputs = pipe(messages)

    # Only decode the part of the output that is generated by the model
    # response = tokenizer.batch_decode(outputs[:, inputs.shape[1] :])[0]
    # pred = parse_llm_response(response)

    # logger.info(f"Response: {response}")
    logger.info(f"Outputs: {outputs}")
    pred = 0

    return pred


def get_llm_prediction_from_single_token(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str = None,
    device: str = "cuda",
    logger=None,
) -> int:
    if not system_prompt:
        messages = [
            {
                "role": "user",
                "content": user_prompt,
            },
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

    formatted_messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_messages, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)

    candidate_tokens = ["n", "e"]
    candidate_token_ids = [
        tokenizer.encode(token, add_special_tokens=False)[0]
        for token in candidate_tokens
    ]
    candidate_probs = np.array(
        [next_token_probs[token_id].item() for token_id in candidate_token_ids]
    )
    normalized_probs = candidate_probs / np.sum(candidate_probs)

    pred = np.argmax(normalized_probs)
    confdc_score = normalized_probs[pred]

    if logger:
        logger.info("Response:")
        logger.info(f"Candidate tokens: {candidate_tokens}")
        logger.info(f"Candidate token ids: {candidate_token_ids}")
        logger.info(f"Raw probabilities: {candidate_probs}")
        logger.info(f"Normalized probabilities: {normalized_probs}")

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
        default="inference",
        help="Mode of operation. 'inference' will incur model API to obtain predictions; 'test' will incur model API up to 3 times for testing purpose; 'evaluation' will evaluate existing predictions.",  # noqa: E501
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
    # temperature = config.getfloat("llm", "temperature")

    if args.mode == "inference" or args.mode == "test":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if (
            model_id == "microsoft/Phi-3-small-8k-instruct"
            or model_id == "microsoft/Phi-3.5-mini-instruct"
        ):  # noqa: E501
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        elif (
            model_id.startswith("meta")
            or model_id.startswith("google")
            or model_id.startswith("tiiuae")
        ):  # noqa: E501
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype="auto", trust_remote_code=True
            )

        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        all_predictions = []
        all_labels = []

        for i, row in test_df.iterrows():
            logger.info(f"{i+1}/{test_df.shape[0]}:")

            # llama-3.1-8b-instruct does not follow the prompt to simply return the result in JSON format # noqa: E501
            if model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
                system_prompt = (
                    "Your task is to decide whether two data objects (e.g., string, tuple, column, entity, etc.) match each other or semantically equivalent. Return the result in the JSON format: '{'match': true}' or '{'match': false}' without any explaination or code.",  # noqa: E501
                )
            else:
                system_prompt = None

            # llama-2 tends to generate longer responses and includes the answer near the end of the response # noqa: E501
            if model_id == "meta-llama/Llama-2-7b-chat-hf":
                max_new_tokens = config.getint("llm", "max_new_tokens")
            else:
                max_new_tokens = 50

            prompt = row["prompt"]
            # pred = get_llm_prediction(
            #     model,
            #     tokenizer,
            #     user_prompt=prompt,
            #     system_prompt=system_prompt,
            #     max_new_tokens=max_new_tokens,
            #     device=device,
            #     logger=logger,
            # )
            pred, confdc_score = get_llm_prediction_from_single_token(
                model,
                tokenizer,
                user_prompt=prompt,
                system_prompt=system_prompt,
                device=device,
                logger=logger,
            )

            all_predictions.append(pred)
            all_labels.append(int(row["label"]))

            logger.info(f"prediction={pred}, label={row['label']}")
            logger.info("-" * 80)

            if args.mode == "test" and i >= 2:
                break

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
