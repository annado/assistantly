import os
import json

import chainlit as cl
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import openai

from prompts import SYSTEM_PROMPT

# Configuration setting to enable or disable the system prompt
ENABLE_SYSTEM_PROMPT = True


configurations = {
    "mistral_7B_instruct": {
        "endpoint_url": os.getenv("MISTRAL_7B_INSTRUCT_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "mistral_7B": {
        "endpoint_url": os.getenv("MISTRAL_7B_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "model": "mistralai/Mistral-7B-v0.1"
    },
    "openai_gpt-4": {
        "endpoint_url": os.getenv("OPENAI_ENDPOINT"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini"
    }
}

# Choose configuration
config_key = os.getenv("MODEL_CONFIG")

# Get selected configuration
config = configurations[config_key]

# Initialize the OpenAI async client
client = wrap_openai(openai.AsyncClient(api_key=config["api_key"], base_url=config["endpoint_url"]))

gen_kwargs = {
    "model": config["model"],
    "temperature": 0.2,
    "max_tokens": 2000
}

@traceable
def get_latest_user_message(message_history):
    # Iterate through the message history in reverse to find the last user message
    for message in reversed(message_history):
        if message['role'] == 'user':
            return message['content']
    return None

def chat_start(documents):
    emails = json.dumps(documents)
    email_content = (
        f"Emails:\n\n{emails}"
    )
    context = f"{SYSTEM_PROMPT}\n\n{email_content}"
    message_history = [{"role": "system", "content": context}]
    cl.user_session.set("message_history", message_history)

def insert_emails_to_history(message_history, documents):
    emails = json.dumps(documents)
    email_content = (
        f"Emails:\n\n{emails}"
    )
    message_history.append({"role": "system", "content": email_content})


@traceable
def append_message_to_history(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    return message_history

@traceable
async def start_response():
    response_message = cl.Message(content="")
    await response_message.send()
    return response_message

async def stream_response(message, message_history, response_message):
    try:
        if config_key == "mistral_7B":
            stream = await client.completions.create(prompt=message.content, stream=True, **gen_kwargs)
            async for part in stream:
                if token := part.choices[0].text or "":
                    await response_message.stream_token(token)
        else:
            stream = await client.chat.completions.create(messages=message_history, stream=True,
                **gen_kwargs)
        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await response_message.stream_token(token)
    except openai.APIError as e:
        # Handle API error here, e.g. retry or log
        print("API error", e)
        response_message.content = "OpenAI API returned an API Error"
        pass
    except openai.APIConnectionError as e:
        print("API error", e)
        # Handle connection error here
        response_message.content = "Failed to connect to OpenAI API"
    except openai.RateLimitError as e:
        print("API error", e)
        # Handle rate limit error (we recommend using exponential backoff)
        response_message.content = "OpenAI API request exceeded rate limit"

    return response_message

async def record_ai_response(cl, message_history, response_message):
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

@traceable
@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history = append_message_to_history(message, message_history)
    response_message = await start_response()
    await stream_response(message, message_history, response_message)
    await record_ai_response(cl, message_history, response_message)
