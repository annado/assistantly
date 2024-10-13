import os
import json
import chainlit as cl
import openai

from typing import List
from llama_index.core.schema import Document
from email_loader import EmailLoader


EMAIL_AGENT_PROMPT = """\
You are a helpful assistant for busy parents. Your job is to read emails and create a summary. Your summary of the emails should be concise, \
  without losing fidelity of information.

- For each email, you should first read the current summary file, if it exists, then update the markdown-formatted file with \
  additional information from the latest email, removing any duplicate items.
- You should not include information regarding Middle School, or TK.

- For the contents of the markdown-formatted plan, create three sections, "Key Dates", "Action Items", and "Highlights."

You will use the following guidelines to update the summary:

1. **Key Dates**:
    - Do not include items that happened more than 2 days ago from {today}
    - If the key date comes from a school-related email, annotate the key dates with the class level or name, if available. 
    - Some of the classes might be labeled with the format L1, L2, L3, etc. This stands for Level 1, Level 2, Level 3, etc.
    - Include time if available.

2. **Action Items**
    - Update the action items if the email contains an action item that the parent needs to complete
      but is not associated with a key date, such as reviewing photos.
    - Include a link to the action item.
    - Annotate by class if available.

3. **Highlights**:
    - Add to highlights if the email mentions something the student learned or did that week.
    - Add to highlights if the email mentions something that happened at school that the parent should know about.

Action items should be formatted like this:

 - [ ] 1. This is the first action item
 - [ ] 2. This is the second action item
 - [ ] 3. This is the third action item

Once you have finished processing the emails, write the summary to the artifacts folder.
"""
class EmailAgent:
    """
    EmailAgent is an agent that reads emails and summarizes them.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "updateArtifact",
                "description": "Update an artifact file contains the summary. The name of the file should be today's date with the format of YYYY-MM-DD.txt",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The name of the file to update.",
                        },
                        "contents": {
                            "type": "string",
                            "description": "The contents to write to the file.",
                        },
                    },
                    "required": ["filename", "contents"],
                    "additionalProperties": False,
                },
            }
        },
    ]

    def __init__(self, client, gen_kwargs=None):
        self.name = "EmailAgent"
        self.client = client
        self.prompt = EMAIL_AGENT_PROMPT
        self.gen_kwargs = gen_kwargs or {
            "model": "gpt-4o",
            "temperature": 0.2
        }

    async def _response_stream(self, message_history, response_message):
        try:
            stream = await self.client.chat.completions.create(messages=message_history, stream=True, tools=self.tools, tool_choice="auto", **self.gen_kwargs)
        except openai.APIError as api_error:
            print("API error", api_error)
            print("Message history", message_history)
            print("response message", response_message)
        except openai.APIConnectionError as api_connection_error:
            print("API connection error", api_connection_error)
        except openai.RateLimitError as rate_limit_error:
            print("Rate limit error", rate_limit_error)
        except Exception as e:
            print("Unknown error", e)

        return stream

    async def execute(self, emails: List[Document], message_history: list, response_message = None):
        """
        Executes the agent's main functionality.
        """
        if not response_message:
            response_message = cl.Message(content="")
            await response_message.send()


        # email_loader = EmailLoader(f"Most recent emails from {school_name} school", school_name=school_name)
        # emails = email_loader.fetch_emails()

        copied_message_history = message_history.copy()

        # Check if the first message is a system prompt
        if copied_message_history and copied_message_history[0]["role"] == "system":
            # Replace the system prompt with the agent's prompt
            copied_message_history[0] = {"role": "system", "content": self._build_system_prompt(emails)}
        else:
            # Insert the agent's prompt at the beginning
            copied_message_history.insert(0, {"role": "system", "content": self._build_system_prompt(emails)})

        stream = await self._response_stream(copied_message_history, response_message)

        function_name = ""
        arguments = ""
        async for part in stream:
            if part.choices[0].delta.tool_calls:
                tool_call = part.choices[0].delta.tool_calls[0]
                function_name_delta = tool_call.function.name or ""
                arguments_delta = tool_call.function.arguments or ""

                if function_name_delta != "":
                    if function_name != "":
                        await self._handle_tool_call(function_name, arguments, copied_message_history, response_message)

                        stream = await self.client.chat.completions.create(messages=message_history, stream=True, **self.gen_kwargs)
                        async for part in stream:
                            if token := part.choices[0].delta.content or "":
                                await response_message.stream_token(token)
                        # reset state for next tool call
                        arguments = ""
                        function_name = ""

                    function_name = function_name_delta

                if arguments_delta != "":
                    arguments += arguments_delta

            if token := part.choices[0].delta.content or "":
                await response_message.stream_token(token)

        if function_name:
            await self._handle_tool_call(function_name, arguments, copied_message_history, response_message)

            stream = await self.client.chat.completions.create(messages=message_history, stream=True, **self.gen_kwargs)
            async for part in stream:
                if token := part.choices[0].delta.content or "":
                    await response_message.stream_token(token)
        else:
            print(f"[{self.__class__.__name__}]: No tool call")

        await response_message.update()

        return response_message.content

    async def call_agent(self, agent_name, message_history, response_message):
        pass

    async def _handle_tool_call(self, function_name, arguments, message_history, response_message):
        if function_name == "updateArtifact":
            try:
                arguments_dict = json.loads(arguments)
            except json.JSONDecodeError as e:
                print(f"[{self.__class__.__name__}]: updateArtifact- JSONDecodeError: {e}")
                return

            filename = arguments_dict.get("filename")
            contents = arguments_dict.get("contents")

            if filename and contents:
                self._update_artifact(filename, contents)

                # Add a message to the message history
                self._append_system_message(message_history, f"The artifact '{filename}' was updated.")

                # stream = await self.client.chat.completions.create(messages=message_history, stream=True, **self.gen_kwargs)
                # async for part in stream:
                #     if token := part.choices[0].delta.content or "":
                #         await response_message.stream_token(token)


    def _append_system_message(self, message_history, message):
        message_history.append({
            "role": "system",
            "content": message
        })

    def _update_artifact(self, filename, contents):
        os.makedirs("artifacts", exist_ok=True)
        with open(os.path.join("artifacts", filename), "w") as file:
            file.write(contents)


    def _build_system_prompt(self, emails: List[Document]):
        """
        Builds the system prompt including the agent's prompt and the contents of the artifacts folder.
        """
        artifacts_content = "<ARTIFACTS>\n"
        artifacts_dir = "artifacts"

        if os.path.exists(artifacts_dir) and os.path.isdir(artifacts_dir):
            for filename in os.listdir(artifacts_dir):
                file_path = os.path.join(artifacts_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as file:
                        file_content = file.read()
                        artifacts_content += f"<FILE name='{filename}'>\n{file_content}\n</FILE>\n"

        artifacts_content += "</ARTIFACTS>"

        email_content = "<EMAILS>\n"
        for email in emails:
          if email is not None:
              # print(email)
              email_content += f"<EMAIL id={email['metadata']['id']}>\n{email['content']}\n</EMAIL>\n"
        email_content += f"</EMAILS>"

        return f"{self.prompt}\n{artifacts_content}\n{email_content}"
