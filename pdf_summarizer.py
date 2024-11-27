import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


# Load environment variables from the .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("openai_api_key")

try:
    pdf_reader = PdfReader("PCSpeech.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
except Exception as e:
    print("Error reading PDF:", e)
    text = ""

# Create Document objects
docs = [Document(page_content=text)]
print(f"Loaded {len(docs)} document(s).")

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

template = '''Write a concise and short summary of the following speech.
Speech: `{text}`
'''
prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

chain = load_summarize_chain(
    llm,
    chain_type='stuff',
    prompt=prompt,
    verbose=False
)
output_summary = chain.run(docs)

output_summary
