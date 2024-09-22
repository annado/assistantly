from datetime import datetime, timedelta
import os

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import (
    load_index_from_storage,
    load_indices_from_storage,
    load_graph_from_storage,
    StorageContext,
)

from custom_gmail_reader import CustomGmailReader

PERSIST_DIR = "./storage"

def load_emails_from_storage():
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index
    return None

def fetch_emails():
    current_date = datetime.now()
    one_week_ago = current_date - timedelta(days=4)
    formatted_date = one_week_ago.strftime("%Y-%m-%d")

    # Instantiate the CustomGmailReader
    print("Creating email loader...")
    loader = CustomGmailReader(
        query=f"after:{formatted_date}",
        max_results=500,
        results_per_page=50,
        service=None
    )

    # Load the emails
    print("Loading emails...")
    documents = loader.load_data()
    print(f"Number of documents: {len(documents)}")
    return documents

def fetch_and_create_index():
    documents = fetch_emails()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def load_emails(query):
    index = load_emails_from_storage()
    if not index:
        print("No index found, fetching emails...")
        index = fetch_and_create_index()

    # Create a retriever to fetch relevant documents
    retriever = index.as_retriever(retrieval_mode='similarity', k=30, similarity_top_k=30)

    # Check for freshness
    latest_docs = retriever.retrieve("Do I have recent emails from within the last 24 hours?")
    if (len(latest_docs) == 0):
        print("No recent emails, refetching...")
        index = fetch_and_create_index()

    # Retrieve relevant documents
    relevant_docs = retriever.retrieve(query)

    print(f"Number of relevant documents: {len(relevant_docs)}")
    print("\n" + "="*50 + "\n")

    ret = []
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i+1}:")
        # print(f"Text sample: {doc.node.get_content()[:200]}...")  # Print first 200 characters
        print(f"Metadata: {doc.node.metadata}")
        print(f"Score: {doc.score}")
        print("\n" + "="*50 + "\n")
        ret.append({
            "metadata": doc.node.metadata,
            "content": doc.node.get_content()
        })

    return ret
