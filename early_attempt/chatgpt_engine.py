import os

import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


def get_subject(col_descr):
    messages = [ {"role": "user", 
                  "content": f"What is the subject in {col_descr}? Must answer with one or two words."
                 }
                ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100, #  max number of tokens in completion, which is about 75 words
        temperature=0.7
    )
    subject = response["choices"][0]["message"]["content"]
    return subject


def post_process_title(title):
    if "title" in title.lower():
        processed_title = ":".join(title.split(":")[1:]).strip(" \"")
        return processed_title
    else:
        return title 


if __name__ == "__main__":
    col_descr = "A candidate's last name as it appears on their application."
    subject = get_subject(col_descr)
    print(subject)
