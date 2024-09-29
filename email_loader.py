from datetime import datetime, timedelta

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

def print_date_from_now(days_ago):
    current_date = datetime.now()
    one_week_ago = current_date - timedelta(days=days_ago)
    formatted_date = one_week_ago.strftime("%Y-%m-%d")
    return formatted_date


def load_emails_from_storage():
    try:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        return index
    except:
        return None

def fetch_emails():
    formatted_date = print_date_from_now(7)

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

def get_email_retriever(force_refetch=False, top_k=30):
    index = load_emails_from_storage()
    if not index or force_refetch:
        print("Refetching emails...")
        index = fetch_and_create_index()

    # Create a retriever to fetch relevant documents
    retriever = index.as_retriever(retrieval_mode='similarity', 
        k=top_k,
        similarity_top_k=top_k
    )
    return retriever

def load_emails(query):
    print(f"Query: {query}")
    retriever = get_email_retriever()

    # Check for freshness
    # formatted_date = print_date_from_now(1)
    # recency_query = f"Fetch me an email from {formatted_date}"
    # node_postprocessor = FixedRecencyPostprocessor(
    #     date_key="date",
    #     window=timedelta(days=1),
    # )
    # print(f"Recency Query: {recency_query}")
    # docs = retriever.retrieve(recency_query)
    # query_bundle = QueryBundle(
    #     query_str=recency_query
    # )
    # filtered_docs = node_postprocessor.postprocess_nodes(docs, query_bundle=query_bundle)
    # print(f"Number of latest documents: {len(filtered_docs)}")
    # if (len(filtered_docs) == 0):
    #     print("No recent emails, refetching...")
    #     index = fetch_and_create_index()
    # else:
    #     print(f"Found recent emails, using cached index. {filtered_docs[0].node.metadata}")

    # Retrieve relevant documents
    relevant_docs = retriever.retrieve(query)

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

    return ret

def get_emails_by_school(school_name):
    emails = load_emails(f"Most recent emails from {school_name}")
    return emails

def get_recent_order_emails():
    emails = load_emails("Most recent emails about recent purchases")
    return emails
