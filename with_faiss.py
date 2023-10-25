import os
import sys
import pickle
import re
import logging
from abc import ABC, abstractmethod
from typing import List
import tiktoken
import requests
import mimetypes
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS as BaseFAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader as BaseWebBaseLoader,
)

from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

load_dotenv()
logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv('OPEN_AI_KEY')
WEBSITE_URL = os.getenv('WEBSITE_URLS')
WEBSITE_URLS = WEBSITE_URL.split(",")
MODEL_NAME = os.getenv('MODEL_NAME')
DOCUMENTATION_NAME = os.getenv('DOCUMENTATION_NAME')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT')
K_COUNT = int(os.getenv('K_COUNT'))
COUNT_FROM_SAME_SOURCE = os.getenv('COUNT_FROM_SAME_SOURCE')
if not COUNT_FROM_SAME_SOURCE:
    COUNT_FROM_SAME_SOURCE = K_COUNT
else:
    COUNT_FROM_SAME_SOURCE = int(COUNT_FROM_SAME_SOURCE)
MODE = os.getenv('MODE')
if not MODE:
    MODE = 'simple'
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP'))
BATCH_SIZE = 100  # Depending on your average document size, adjust this accordingly
RATE_LIMIT_TOKENS = 950000  # Setting it a bit lower than 1 million for safety

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, max_retries=3)
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-4')

class WebBaseLoader(BaseWebBaseLoader):

    def _fetch(
            self, url: str, selector: str = 'body', retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        for i in range(retries):
            try:
                webdriver_service = Service(
                    '/Users/timur/symfony/guestbook/vendor/symfony/panther/chromedriver-bin/chromedriver_mac64')  # Update this path
                options = webdriver.ChromeOptions()
                options.add_argument('headless')
                driver = webdriver.Chrome(service=webdriver_service, options=options)
                driver.get(url)

                # Wait until the specific element is visible on the page
                WebDriverWait(driver, timeout=500).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, selector))
                )

                content = driver.page_source
                driver.quit()
                return content
            except Exception as e:
                if i == retries - 1:
                    raise
                else:
                    logger.warning(
                        f"Error fetching {url} with attempt "
                        f"{i + 1}/{retries}: {e}. Retrying..."
                    )
                    time.sleep(cooldown * backoff ** i)
        raise ValueError("retry count exceeded")


class DocumentLoader(ABC):
    @abstractmethod
    def load_and_split(self) -> List[str]:
        pass


class FAISS(BaseFAISS):
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def addEmb(self, pages, embeddings, metadatas):
        self.__add(pages, embeddings, metadatas)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)


class URLHandler:
    @staticmethod
    def is_valid_url(url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    @staticmethod
    def extract_links(url):
        urls = set()
        domain_name = urlparse(url).netloc
        soup = BeautifulSoup(requests.get(url).content, "html.parser")

        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            href = urljoin(url, href)
            parsed_href = urlparse(href)
            if parsed_href.path.endswith(
                    ('.pdf', '.jpg', '.png', '.jpeg', '.gif', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
                continue
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not URLHandler.is_valid_url(href):
                continue
            if href in urls:
                continue
            if domain_name not in href:
                continue
            urls.add(href)
        return urls

    @staticmethod
    def extract_links_from_websites(websites):
        all_links = set()

        for website in websites:
            links = URLHandler.extract_links(website)
            all_links.update(links)

        return list(all_links)


def remove_phrase(data, phrase):
    # Create a pattern that matches the phrase and any words around it
    pattern = re.compile(r'\b\w*?\s*' + re.escape(phrase) + r'\s*\w*\b', re.IGNORECASE)

    # Remove the phrase and any surrounding words
    cleaned_data = re.sub(pattern, '', data)

    return cleaned_data


def get_loader(file_path_or_url):
    if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
        handle_website = URLHandler()
        return WebBaseLoader(handle_website.extract_links_from_websites([file_path_or_url]))
    else:
        mime_type, _ = mimetypes.guess_type(file_path_or_url)

        if mime_type == 'application/pdf':
            return PyPDFLoader(file_path_or_url)
        elif mime_type == 'text/csv':
            return CSVLoader(file_path_or_url)
        elif mime_type == 'text/plain':
            return TextLoader(file_path_or_url)
        elif mime_type in ['application/msword',
                           'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            return UnstructuredWordDocumentLoader(file_path_or_url)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")


def train_or_load_model(train, faiss_obj_path, file_paths):
    if train:
        phrase = "Machine Translated by Google"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                       chunk_overlap=CHUNK_OVERLAP
                                                       )
        #        text_splitter = CharacterTextSplitter(separator=".Статья ")
        docs = []
        for file_path in file_paths:
            file_path = file_path.strip(' \n')
            loader = get_loader(file_path)
            docs.extend(loader.load())

        pages = text_splitter.split_documents(docs)

        fixed_pages = []
        metadatas = []
        for i in pages:
            print("\n ______________")
            i.page_content = remove_phrase(i.page_content, phrase)  # remove if the data translate from google
            # i.page_content = structured_chunk(i.page_content)
            print(i.page_content)
            fixed_pages.append(i.page_content)
            metadatas.append(i.metadata)

        # Save pages to a text file
        with open('output.txt', 'w', encoding='utf-8') as f:
            sys.stdout = f  # Redirect standard output to the file
            print(pages)  # The output will be saved to 'output.txt'

            sys.stdout = sys.__stdout__  # Reset standard output

        # if os.path.exists(faiss_obj_path):
        #    faiss_index = FAISS.load(faiss_obj_path)
        #    new_embeddings = faiss_index.from_documents(pages, embeddings)
        #    new_embeddings.save(faiss_obj_path)
        # else:

        if MODE == 'batches':
            # embed_in_batches
            all_embeddings = embed_in_batches(fixed_pages, embeddings, BATCH_SIZE)
            faiss_index = FAISS.from_texts([SYSTEM_PROMPT], embeddings)
            faiss_index.addEmb(fixed_pages, all_embeddings, metadatas)
            faiss_index.save(faiss_obj_path)
            # end embed in batches
        else:
            faiss_index = FAISS.from_documents(pages, embeddings)
            faiss_index.save(faiss_obj_path)

        return FAISS.load(faiss_obj_path)
    else:
        return FAISS.load(faiss_obj_path)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_tokens_with_tiktoken(texts):
    sum = 0
    for text in texts:
        sum = sum + num_tokens_from_string(text)
    return sum


def embed_in_batches(pages, embeddings, batch_size):
    total_pages = len(pages)
    all_embeddings = []

    for start_idx in range(0, total_pages, batch_size):
        end_idx = min(start_idx + batch_size, total_pages)
        batch_pages = pages[start_idx:end_idx]

        if count_tokens_with_tiktoken(batch_pages) > RATE_LIMIT_TOKENS:
            raise ValueError("Individual batch exceeds rate limit!")

        batch_embeddings = embeddings.embed_documents(batch_pages)  # Assuming 'embed' is the function to get embeddings
        all_embeddings.extend(batch_embeddings)

        # Sleep if needed to avoid hitting the rate limit, especially if you're close to the limit.
        time.sleep(1)  # sleep for 1 second between batches

    return all_embeddings


def structured_chunk(message):
    messages = [SystemMessage(
        content="Please enhance and refine the following text to ensure clarity and standardization. Remove all "
                "extraneous components, including HTML tags, miscellaneous characters, and any segments "
                "translated by automatic systems like Google Translate."), HumanMessage(content=message)]

    ai_response = chat(messages).content
    return ai_response


def answer_questions(faiss_index):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

    while True:
        question = input("Ask a question (type 'stop' to end): ")
        if question.lower() == "stop":
            break

        docs = faiss_index.similarity_search(query=question, k=K_COUNT)
        # docs = faiss_index.max_marginal_relevance_search(query=question, k=K_COUNT, fetch_k=(K_COUNT*50))
        #print(docs)

        source_counts = dict()
        main_content = "Begin of " + DOCUMENTATION_NAME + "\n\n"
        for doc in docs:
            current_source = doc.metadata['source']
            source_counts[current_source] = source_counts.get(current_source, 0) + 1
            if source_counts[current_source] > COUNT_FROM_SAME_SOURCE:
                continue
            main_content += doc.page_content + "\n\n"
            print(doc)
        main_content += "End of " + DOCUMENTATION_NAME + "\n\nQuestion: " + question
        print(source_counts)

        messages.append(HumanMessage(content=main_content))
        ai_response = chat(messages).content
        messages.pop()
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=ai_response))

        print(ai_response)


def main():
    faiss_obj_path = "models/" + MODEL_NAME + ".pickle"
    file_paths = WEBSITE_URLS

    train = int(input("Do you want to train the model? (1 for yes, 0 for no): "))
    faiss_index = train_or_load_model(train, faiss_obj_path, file_paths)
    answer_questions(faiss_index)

if __name__ == "__main__":
    main()
