import json
from dotenv import load_dotenv
import chainlit as cl
from langsmith import traceable

from chatbot import Chatbot

# Load environment variables
load_dotenv()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="School email summary",
            message="What is my weekly school email summary?",
            # icon="/public/idea.svg",
            ),

        cl.Starter(
            label="My recent orders",
            message="Find my recent purchase orders and help me track when they shipped.",
            # icon="/public/learn.svg",
            ),
    ]

chatbot = Chatbot()

@cl.on_chat_start
def start_chat():
    # documents = load_emails("Most recent emails from school")
    # chat.chat_start(documents)
    chatbot.start_chat()


@cl.on_message
@traceable
async def on_message(message: cl.Message):
    message_history = chatbot.append_user_message_to_history(message)

    response_message = await chatbot.generate_response()

    while await chatbot.check_if_function_call(response_message):
        response_message = await chatbot.handle_function_call(message_history, response_message)

    chatbot.update_message_history(message_history)

if __name__ == "__main__":
    cl.main()
