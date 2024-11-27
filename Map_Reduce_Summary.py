import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Validate API key
api_key = os.getenv("openai_api_key")
os.environ["OPENAI_API_KEY"] = api_key

# Read PDF file
pdfreader = PdfReader('PCSpeech.pdf')

# Extract text from PDF
text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        text += content

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')  ## Initialize LLM

total_tokens = llm.get_num_tokens(text)
if total_tokens > 4096:   ## count tokens
    print(f"Warning: The input text exceeds the token limit ({total_tokens} tokens). Splitting into chunks.")

# Split text into manageable chunks
chunk_size = 10000
chunk_overlap = 20
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
    )
chunks = text_splitter.create_documents([text])
print(f"Number of chunks created: {len(chunks)}")

# MapReduce summarization
summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    verbose=True
)
summary = summary_chain.run(chunks)
print("Summary:", summary)

map_prompt = '''
Please summarize the below speech:
Speech: `{text}`
Summary:
'''

combine_prompt = '''
Provide a final summary of the entire speech with these important points:
1. Add a Generic Motivational Title.
2. Start the precise summary with an introduction.
3. Provide the summary in numbered points.

Speech: `{text}`
'''

map_prompt_template = PromptTemplate(input_variables=['text'], template=map_prompt)
combine_prompt_template = PromptTemplate(input_variables=['text'], template=combine_prompt)

# Custom summary chain
custom_summary_chain = load_summarize_chain(
    llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt_template,
    combine_prompt=combine_prompt_template,
    verbose=True
)
custom_summary = custom_summary_chain.run(chunks)
print("Custom Summary:", custom_summary)

