from bs4 import BeautifulSoup
from markdownify import markdownify as md

def parse_email_to_markdown(file_path):
    # Open and read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    email = md(html_content)
    return email

def parse_email_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")

    for script in soup(["script", "style"]):
      script.extract() 

    text = [p.text for p in soup.find_all("p")]
    full_text = "\n".join(text)
    return full_text

def write_to_file(file_path):
  with open(file_path, 'w', encoding='utf-8') as file:
      file.write(text)
