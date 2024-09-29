import json
from dotenv import load_dotenv
import chainlit as cl
from langsmith import traceable

import chatbot
from email_loader import load_emails

# Load environment variables
load_dotenv()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="School email summary",
            message="Can you send me a summary of my school email content? Categorize them by key dates, action items, and highlights from the past week.",
            # icon="/public/idea.svg",
            ),

        cl.Starter(
            label="My recent orders",
            message="Find my recent purchase orders and help me track when they shipped.",
            # icon="/public/learn.svg",
            ),
    ]


@cl.on_chat_start
def start_chat():
    # documents = load_emails("Most recent emails from school")
    # chat.chat_start(documents)
    chatbot.start_chat([])


@traceable
@cl.on_message
async def on_message(message: cl.Message):
    message_history = chatbot.append_user_message_to_history(message)

    response_message = await chatbot.generate_response(message_history)

    while await chatbot.check_if_function_call(response_message):
        response_message = await chatbot.handle_function_call(message_history, response_message)

    if response_message:
        await chatbot.append_ai_message_to_history(message_history, response_message)

    chatbot.update_message_history(message_history)

if __name__ == "__main__":
    cl.main()
