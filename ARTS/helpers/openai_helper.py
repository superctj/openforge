import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
OPENAI_API_KEY = "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" # Jag's


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    assert OPENAI_API_KEY is not None
    openai.api_key = OPENAI_API_KEY
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

# def buildPrompt(table_id: str, contexts: list | None = None):
#     if contexts is None:
#         contexts = ["name", "notes", "column names"]
#     dataset_metadata = getDatasetMetadataWithTableID(table_id)
#     table_df = readCSVFileWithTableID(table_id)

#     prompt = "Given a dataset with the following information: \n"
#     if "name" in contexts:
#         prompt += "Dataset Name: {} \n".format(dataset_metadata['title'])
#     if "notes" in contexts:
#         prompt += "Dataset Notes: {} \n".format(dataset_metadata['notes'].replace('\n', ' '))
#     if "column names" in contexts:
#         prompt += "A data table for this dataset has the following columns: {}.\n".format(
#             ",".join([x.strip() for x in table_df.columns]))
#     # prompt += "Please answer: what's the meaning for each of the columns? Use simplest words and do not explain."
#     prompt += "Please answer: what's the meaning for each of the columns? No need to explain. Use simplest words, do not show sample value.  For each columns, answer in the format: <column name>  : <column meaning>"

#     return prompt


def getGPTResponse(prompt, model="gpt-3.5-turbo", max_tokens=200):
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # OPENAI_API_KEY = "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" # Junjie's
    OPENAI_API_KEY = "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" # Jag's
    # OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
    # OPENAI_API_KEY = "OPENAI_API_KEY"
    assert OPENAI_API_KEY is not None
    openai.api_key = OPENAI_API_KEY

    if model == "gpt-3.5-turbo":
        conversation = [
            {'role': 'user', 'content': prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )

        # reply = response['choices'][0]['message']['content']

        return response
    else:
        raise NotImplementedError
