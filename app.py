import os
from dotenv import load_dotenv
import chainlit as cl
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from prompts import ASSESSMENT_PROMPT, SYSTEM_PROMPT, CLASS_CONTEXT

# Load environment variables
load_dotenv()
   
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
config_key = "openai_gpt-4"
# config_key = "mistral_7B_instruct"
# config_key = "mistral_7B"

# Get selected configuration
config = configurations[config_key]

# Initialize the OpenAI async client
client = wrap_openai(openai.AsyncClient(api_key=config["api_key"], base_url=config["endpoint_url"]))

gen_kwargs = {
    "model": config["model"],
    "temperature": 0.3,
    "max_tokens": 500
}

# Configuration setting to enable or disable the system prompt
ENABLE_SYSTEM_PROMPT = True
ENABLE_CLASS_CONTEXT = True

@traceable
def get_latest_user_message(message_history):
    # Iterate through the message history in reverse to find the last user message
    for message in reversed(message_history):
        if message['role'] == 'user':
            return message['content']
    return None

def is_beginning_of_history(message_history):
    return not message_history or message_history[0].get("role") != "system"

def parse_email_to_markdown(file_path):
    # Open and read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    email = md(html_content)
    return email

def parse_email_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")

    for script in soup(["script", "style"]):
      script.extract() 

    text = [p.text for p in soup.find_all("p")]
    full_text = "\n".join(text)
    return full_text

def write_to_file(file_path):
  with open(file_path, 'w', encoding='utf-8') as file:
      file.write(text)

def insert_system_prompt_to_history(message_history):
    system_prompt_content = (
        f"{SYSTEM_PROMPT}\n{CLASS_CONTEXT}\n{ASSESSMENT_PROMPT}\n"
    )
    message_history.insert(0, {"role": "system", "content": system_prompt_content})


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

@traceable
async def stream_response(message, message_history, response_message):
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

async def record_ai_response(message_history, response_message):
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

@cl.on_message
async def on_message(message: cl.Message):
    message_history = append_message_to_history(message)
    if is_beginning_of_history(message_history):
        print("Appending system prompt")
        insert_system_prompt_to_history(message_history)

    response_message = await start_response()
    await stream_response(message, message_history, response_message)
    await record_ai_response(message_history, response_message)
    await response_message.update()


if __name__ == "__main__":
    cl.main()
