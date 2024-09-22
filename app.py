from dotenv import load_dotenv
import chainlit as cl
from langsmith import traceable

import chat
from email_loader import load_emails

# Load environment variables
load_dotenv()


@cl.on_message
@traceable
async def on_message(message: cl.Message):
    message_history = chat.append_message_to_history(message)
    response_message = await chat.start_response()

    if chat.is_beginning_of_history(message_history):
        documents = load_emails("Recent school emails and purchase orders")
        chat.insert_system_prompt_to_history(message_history, documents)

    await chat.stream_response(message, message_history, response_message)
    await chat.record_ai_response(cl, message_history, response_message)
    await response_message.update()

if __name__ == "__main__":
    cl.main()
