"""
Updated code from the "Generative AI: PaLM-2 model deployment with Cloud Run" article:
https://medium.com/google-cloud/generative-ai-palm-2-model-deployment-with-cloud-run-54e8a398b24b

https://cloud.google.com/vertex-ai/generative-ai/docs/text/test-text-prompts#generative-ai-test-text-prompt-python_vertex_ai_sdk
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/getting-started/intro_palm_api.ipynb
https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models
"""
import os
import logging
from dotenv import load_dotenv

import vertexai
from vertexai.language_models import TextGenerationModel

import gradio as gr

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')

load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL)

vertexai.init(project=PROJECT_ID, location=LOCATION)

model = TextGenerationModel.from_pretrained("text-bison@002")


def predict(prompt, max_output_tokens, temperature, top_p, top_k):
    """Predict text with text-bison@002

    Args:
        prompt (_type_): _description_
        max_output_tokens (_type_): _description_
        temperature (_type_): _description_
        top_p (_type_): _description_
        top_k (_type_): _description_

    Returns:
        _type_: _description_
    """
    logging.info(prompt)
    answer = model.predict(
        prompt,
        max_output_tokens=max_output_tokens,  # default 128
        temperature=temperature,  # default 0
        top_p=top_p,  # default 1
        top_k=top_k)  # default 40
    logging.info(answer.text)
    return answer.text


# pylint: disable=line-too-long
examples = [
    ["Give me ten interview questions for the role of program manager."],
    ["How do you recommend I prepare and grill tasty medium to well-done steak?"],
    ["Brainstorm some ideas for applying Generative AI to solve real business problems:"],
    ["You are an equities analyst researching information for your report with relevant facts and figures. Tell me about the mortgage market in US."],
    ["Tell me a good Fish and Chips recipe, please."],
]
FIRST_STRING = examples[0][0]

demo = gr.Interface(
    predict,
    [
        gr.Textbox(label="Enter prompt:", value=FIRST_STRING),
        gr.Slider(32, 1024, value=512, step=32, label="max_output_tokens"),
        gr.Slider(0, 1, value=0.2, step=0.1, label="temperature"),
        gr.Slider(0, 1, value=0.8, step=0.1, label="top_p"),
        gr.Slider(1, 40, value=38, step=1, label="top_k"),
    ],
    "text",
    examples=examples,
)

# Leaving at 8080 cause it is Cloud Run default port, also removed PORT expose from Dockerfile

# This works because Gradio PORT 7860 is EXPOSED in Dockerfile
# PORT = int(os.getenv("PORT", '7860'))
demo.launch(server_name="0.0.0.0", server_port=8080)
