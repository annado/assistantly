from datetime import datetime, timedelta
import json

from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
    QueryBundle
)
from llama_index.core.postprocessor import (
    FixedRecencyPostprocessor
)
from custom_gmail_reader import CustomGmailReader

PERSIST_DIR = "./storage"


class EmailLoader:

    email_loader: ChildProcessError
    query: str = ""

    def __init__(self, query=None, school_name=None):
        self.query = query
        self.school_name = school_name

        formatted_date = self._print_date_from_now(7)
        gmail_query = f"after:{formatted_date}"
        if self.school_name:
            gmail_query = f"{gmail_query} AND ({self.school_name} school) AND -subject:'Re:'"

        print(f"Gmail Query: {gmail_query}")
        self.email_loader = CustomGmailReader(
            query=gmail_query,
            max_results=100 if query else 500,
            results_per_page=50,
            service=None
        )


    def _print_date_from_now(self, days_ago):
        current_date = datetime.now()
        one_week_ago = current_date - timedelta(days=days_ago)
        formatted_date = one_week_ago.strftime("%Y-%m-%d")
        return formatted_date


    def _load_emails_from_storage(self):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            return index
        except:
            return None


    def fetch_emails(self):
        # formatted_date = self._print_date_from_now(7)
        # gmail_query = f"after:{formatted_date}"
        # if self.school_name:
        #     gmail_query = f"{gmail_query} AND (from:{self.school_name} OR subject:{self.school_name}) AND -subject:'Re:'"

        # print(f"Gmail Query: {gmail_query}")
        # Instantiate the CustomGmailReader
        # print("Creating email loader...")
        # loader = CustomGmailReader(
        #     query=gmail_query,
        #     max_results=100 if self.query else 500,
        #     results_per_page=50,
        #     service=None
        # )

        # Load the emails
        print("Loading emails...")
        documents = self.email_loader.load_data()
        print(f"Number of results: {len(documents)}")

        return documents

    def _fetch_and_create_index(self):
        documents = self.fetch_emails()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

    def _get_email_retriever(self, force_refetch=False, top_k=30):
        index = self._load_emails_from_storage()
        if not index or force_refetch:
            print("Refetching emails...")
            index = self._fetch_and_create_index()

        # Create a retriever to fetch relevant documents
        retriever = index.as_retriever(retrieval_mode='similarity', 
            k=top_k,
            similarity_top_k=top_k
        )
        return retriever

    def load_emails(self):
        # print(f"Query: {query}")
        retriever = self._get_email_retriever(True)

        # Retrieve relevant documents
        relevant_docs = retriever.retrieve(self.query)

        print(f"Number of relevant documents: {len(relevant_docs)}")
        print("\n" + "="*50 + "\n")

        ret = []
        for i, doc in enumerate(relevant_docs):
            # print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
            print(f"Document {i+1}: {doc.node.metadata['date']}")
            print(f"Subject: {doc.node.metadata['subject']}")
            print(f"Score: {doc.score}")
            print("="*50)
            ret.append({
                "metadata": doc.node.metadata,
                "content": doc.node.get_content()
            })

        return self._format_emails(ret)

    def get_emails_by_school(self, school_name):
        emails = self.load_emails(f"Most recent emails from {school_name} school")
        return emails

    def get_recent_order_emails(self):
        emails = self.load_emails("Most recent emails about recent purchases")
        return emails

    def _format_emails(self, documents) -> str:
        emails = json.dumps(documents)
        email_content = (
            f"Emails:\n\n{emails}"
        )
        return email_content
