"""
Google Mail reader.
The original code is: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-google/llama_index/readers/google/gmail/base.py
Modifications made to create a cleaner document, and add metadata for filtering (to, from, subject, etc.)
"""

import base64
import email
from email import message_from_bytes
from email.utils import parseaddr
from typing import Any, List, Optional
from markdownify import markdownify as md

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from pydantic import BaseModel

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class CustomGmailReader(BaseReader, BaseModel):
    """Gmail reader.
    Reads emails
    Args:
        max_results (int): Defaults to 10.
        query (str): Gmail query. Defaults to None.
        service (Any): Gmail service. Defaults to None.
        results_per_page (Optional[int]): Max number of results per page. Defaults to 10.
        use_iterative_parser (bool): Use iterative parser. Defaults to False.
    """

    query: str = None
    use_iterative_parser: bool = False
    max_results: int = 10
    service: Any
    results_per_page: Optional[int]

    def load_data(self) -> List[Document]:
        """Load emails from the user's account."""
        from googleapiclient.discovery import build

        credentials = self._get_credentials()
        if not self.service:
            self.service = build("gmail", "v1", credentials=credentials)

        messages = self.search_messages()

        results = []
        for message in messages:
            text = message.pop("body")
            metadata = message

            results.append(Document(text=text, metadata=metadata or {}))

        return results

    def _get_credentials(self) -> Any:
        """Get valid user credentials from storage.
        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.
        Returns:
            Credentials, the obtained credential.
        """
        import os

        from google_auth_oauthlib.flow import InstalledAppFlow

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=8080)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def search_messages(self, query=None):
        query = query or self.query

        max_results = self.max_results
        if self.results_per_page:
            max_results = self.results_per_page

        results = (
            self.service.users()
            .messages()
            .list(userId="me", q=query, maxResults=int(max_results))
            .execute()
        )
        messages = results.get("messages", [])

        if len(messages) < self.max_results:
            # paginate if there are more results
            while "nextPageToken" in results:
                page_token = results["nextPageToken"]
                results = (
                    self.service.users()
                    .messages()
                    .list(
                        userId="me",
                        q=query,
                        pageToken=page_token,
                        maxResults=int(max_results),
                    )
                    .execute()
                )
                messages.extend(results["messages"])
                if len(messages) >= self.max_results:
                    break

        result = []
        try:
            for message in messages:
                message_data = self.get_message_data(message)
                if not message_data:
                    continue
                result.append(message_data)
        except Exception as e:
            raise Exception("Can't get message data" + str(e))

        return result

    def get_message_data(self, message):
        message_id = message["id"]
        message_data = (
            self.service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )

        headers = {header['name'].lower(): header['value'] for header in message_data['payload']['headers']}

        body = self.extract_message_body(message_data)
        # body2 = self.parse_multipart_email(message_data)
        # print(f"message_data: {body2}")
        return {
            "id": message_data["id"],
            "threadId": message_data["threadId"],
            "snippet": message_data.get("snippet", ""),
            "internalDate": message_data.get("internalDate", ""),
            "body": body,
            "from": headers.get('from', ""),
            "to": headers.get('to', ""),
            "subject": headers.get('subject', ""),
            "date": headers.get('date', ""),
        }

    def parse_multipart_email(self, raw_message):
        email_message = message_from_bytes(raw_message)
        
        html_content = ""
        plain_content = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    plain_content = part.get_payload(decode=True).decode()
                elif content_type == 'text/html':
                    html_content = part.get_payload(decode=True).decode()
        else:
            content_type = email_message.get_content_type()
            if content_type == 'text/plain':
                plain_content = email_message.get_payload(decode=True).decode()
            elif content_type == 'text/html':
                html_content = email_message.get_payload(decode=True).decode()

        # Prefer plain text, fall back to HTML converted to plain text
        body = plain_content or md(html_content)

        return body.strip()

    def extract_message_body(self, message_data):
        message_body = ''

        def get_text(payload):
            if 'body' in payload:
                data = payload['body'].get('data')
                if data:
                    decoded_data = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                    return decoded_data
            return ''

        def get_html(payload):
            if 'body' in payload:
                data = payload['body'].get('data')
                if data:
                    decoded_data = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
                    markdown = md(decoded_data)
                    return markdown
            return ''

        def find_plain_text(message_body, payload):
            if payload.get('mimeType') == 'text/html':
                return get_html(payload)
            if payload.get('mimeType') == 'text/plain':
                return get_text(payload)

            if payload.get('parts'):
                for part in payload.get('parts'):
                    text = find_plain_text(message_body, part)
                    if text:
                        message_body += text
            return message_body

        body = find_plain_text(message_body, message_data['payload'])
        return body if body else ""


if __name__ == "__main__":
    reader = CustomGmailReader(query="from:me after:2024-08-01")
    print(reader.load_data())
