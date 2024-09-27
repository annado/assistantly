from dotenv import load_dotenv
import chainlit as cl
from langsmith import traceable

import chat
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
    documents = load_emails("Most recent emails from school")
    chat.chat_start(documents)

@cl.on_message
@traceable
async def on_message(message: cl.Message):
    message_history = chat.append_message_to_history(message)
    response_message = await chat.start_response()

    await chat.stream_response(message, message_history, response_message)
    await chat.record_ai_response(cl, message_history, response_message)
    await response_message.update()

if __name__ == "__main__":
    cl.main()
