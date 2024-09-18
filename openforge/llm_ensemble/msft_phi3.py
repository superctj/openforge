import argparse
import os

import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from openforge.utils.custom_logging import create_custom_logger
from openforge.utils.llm_common import parse_llm_response
from openforge.utils.prior_model_common import log_exp_metrics
from openforge.utils.util import parse_config


def get_llm_prediction(
    model,
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

    formatted_messages = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_messages, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
        )

    # Only decode the part of the output that is generated by the model
    response = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :]
    )[0]
    pred = parse_llm_response(response)

    if logger:
        logger.info(f"Response: {response}")

    return pred


# def get_mistral_prediction(
#     model,
#     tokenizer,
#     user_prompt: str,
#     device: str,
#     logger=None,
# ) -> int:
#     messages = [
#         {
#             "role": "user",
#             "content": user_prompt,
#         },
#     ]

#     inputs = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt"
#     ).to(device)

#     with torch.no_grad():
#         outputs = model.generate(inputs, max_new_tokens=50, do_sample=True)

#     # Only decode the part of the output that is generated by the model
#     response = tokenizer.batch_decode(outputs[:, inputs.shape[1] :])[0]
#     pred = parse_llm_response(response)

#     logger.info(f"Response: {response}")

#     return pred


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
    # temperature = config.getfloat("llm", "temperature")

    if args.mode == "inference":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_id == "microsoft/Phi-3-small-8k-instruct":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

        all_predictions = []
        all_labels = []

        for i, row in test_df.iterrows():
            prompt = row["prompt"]
            pred = get_llm_prediction(model, tokenizer, prompt, device, logger)

            all_predictions.append(pred)
            all_labels.append(int(row["label"]))

            # logger.info(f"prediction={pred}, label={row['label']}")
            # logger.info("-" * 80)

            # if i >= 2:
            #     break

            logger.info(
                f"{i+1}/{test_df.shape[0]}: prediction={pred}, label={row['label']}"  # noqa: E501
            )
            logger.info("-" * 80)

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
