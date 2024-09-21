# AI Email Assistant
This Chainlit-based application serves as an assistant to give you a weekly digest of a few different types of emails:
* Summarize weekly school emails
* Summarize and track orders awaiting shipping

## Setup

### Prerequisites
* Python 3.7+
* API keys for OpenAI
* API keys for [Langchain](https://docs.smith.langchain.com)

### Installation & Setup

1. **Create virtual environment**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run**
   * Locally: `chainlit run app.py -w`
   * Server: `chainlit run app.py -h`
