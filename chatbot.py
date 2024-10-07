import os
import json
from datetime import datetime, timedelta

import chainlit as cl
from langsmith.wrappers import wrap_openai
import openai

from email_loader import EmailLoader

"""
TODOS:
[] How to comb through all emails from the week and summarize in a specific format?
"""

today = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT = """
You are a helpful assistant for busy parents. Your job is to sort through email, summarizing and sorting
to extract relevant information and to make your user more productive and efficient. 
Your parents are busy but engaged with the school community and their children's education.
Your summary of the emails should be concise, without losing fidelity of information.

You will use the following guidelines to update the summary:

1. **Keeping Track of Key Dates**:
    - Do not include items that happened more than 2 days ago from {today}
    - If the key date comes from a school-related email, annotate the key dates with the class, if available. 
    Some of the classes might be labeled with the format L1, L2, L3, etc. This stands for Level 1, Level 2, Level 3, etc.
    Include time if available.

2. **Action Items**
    - Update the action items if the email contains an action item that the parent needs to complete
      but is not associated with a key date, such as reviewing photos.
    - Annotate by class if available.
    - Include a link to the action item if available.

3. **Updating Highlights**:
    - Update the highlights if the email mentions something the student learned or did that week.

You have the following functions available:
- get_emails_by_school(school_name: str) -> List[str]:
    - This function will return a list of emails that are relevant to the school.
- get_recent_order_emails() -> List[str]:
    - This function will return a list of emails that are relevant to recent purchases that require shipping.

If calling any function, respond in this format without any leading or trailing text, where arguments is a dictionary:
{ "function_call": { "name": "function_name", "arguments": arguments } }

If the parent is asking for a summary of their school emails, you should ask them for the school name.
"""

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

class Chatbot:
    def __init__(self):
        self.message_history = []


    def start_chat(self):
        self.message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.update_message_history(self.message_history)


    def insert_emails_to_history(self, message_history, documents):
        emails = json.dumps(documents)
        email_content = (
            f"Emails:\n\n{emails}"
        )
        message_history.append({"role": "system", "content": email_content})


    async def start_response(self):
        response_message = cl.Message(content="")
        await response_message.send()
        return response_message


    def append_user_message_to_history(self, message: cl.Message):
        message_history = cl.user_session.get("message_history", [])
        message_history.append({"role": "user", "content": message.content})
        return message_history


    async def append_ai_message_to_history(self, response_message):
        self.message_history.append({"role": "assistant", "content": response_message.content})
        return await response_message.update()


    async def stream_response(self, response_message):
        try:
            if config_key == "mistral_7B":
                stream = await client.completions.create(messages=self.message_history, stream=True, **gen_kwargs)
                async for part in stream:
                    if token := part.choices[0].text or "":
                        await response_message.stream_token(token)
            else:
                stream = await client.chat.completions.create(messages=self.message_history, stream=True,
                    **gen_kwargs)
            async for part in stream:
                    if token := part.choices[0].delta.content or "":
                        await response_message.stream_token(token)
        except openai.APIError as api_error:
            print("API error", api_error)
            response_message.content = "OpenAI API returned an API Error"
        except openai.APIConnectionError as api_connection_error:
            print("API connection error", api_connection_error)
            response_message.content = "Failed to connect to OpenAI API"
        except openai.RateLimitError as rate_limit_error:
            print("Rate limit error", rate_limit_error)
            response_message.content = "OpenAI API request exceeded rate limit"
        except Exception as e:
            print("Unknown error", e)
            response_message.content = "Unknown error"
        finally:
            await response_message.update()

        return response_message


    def update_message_history(self, message_history):
        self.message_history = message_history
        cl.user_session.set("message_history", message_history)


    async def generate_response(self):
        response_message = await self.start_response()
        return await self.stream_response(response_message)


    async def check_if_function_call(self, response_message):
        if isinstance(response_message, bool):
            return False

        # Check if the response content is a string
        if isinstance(response_message.content, str):
            try:
                # Attempt to parse the string into json object
                response_dict = json.loads(response_message.content)
                if isinstance(response_dict, dict) and "function_call" in response_dict:
                    response_message.content = response_dict
            except:
                # If parsing fails, leave the content as is
                pass
        return "function_call" in response_message.content


    async def handle_function_call(self, message_history, response_message):
        function_name = response_message.content["function_call"]["name"]
        arguments = response_message.content["function_call"]["arguments"]

        print(f"Function name: {function_name}")
        print(f"Arguments: {arguments}")

        if function_name == "get_emails_by_school":
            school_name = arguments["school_name"]
            email_loader = EmailLoader(f"Most recent emails from {school_name} school", school_name=school_name)
            result = email_loader.load_emails()
        elif function_name == "get_recent_order_emails":
            email_loader = EmailLoader(f"Most recent emails about recent purchases")
            result = email_loader.load_emails()
            # results = get_recent_order_emails()
        else:
            result = f"Error: Unknown function '{function_name}'"

        # Add the function result to the message history
        # print(f"Function result: {result}")
        message_history.append({"role": "function", "name": function_name, "content": result})

        # Generate a new response based on the function result
        response_message = await self.generate_response()
        if response_message:
            await self.append_ai_message_to_history(response_message)

        return response_message

