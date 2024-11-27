import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate


# Load environment variables from the .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("openai_api_key")
llm = ChatOpenAI(model_name='gpt-3.5-turbo')  # define LLM model

## Basic Prompt Summarization

speech="""
Over the last 11 years, I have witnessed firsthand the incredible work that UNICEF does for children around the world. Especially victims and survivors of child marriage, displacement, war, sexual violence.
But there is still so much work to do.
And for me, that is the fuel to my fire.
The reason I’m so committed to this cause and that is where my passion stems from because I know that a girl’s education not just empowers families but communities and economies.
A result of her education we all do better. It’s just as simple as that.
As entertainers and influencers sitting in this room I feel that is our social responsibility to be a voice for the voiceless, which is why I applaud each and every woman in this room for being such a badass.
For using your platform and your voice to contribute to change and for ensuring that there is not even one lost generation as long as we are alive.
I’d like to thank variety and all of you for encouraging me and all of us in this room to keep going and fighting on.
Thank you so much.
"""
tokens_used = llm.get_num_tokens(speech) # tokenize the text
print(f"Total tokens used for speech: {tokens_used}")

chat_messages = [
    SystemMessage(content='You are an expert assistant with expertise in summarizing speeches.'),
    HumanMessage(content=f'Please provide a short and concise summary of the following speech:\n\n{speech}')
]

# Get the summary
response = llm(chat_messages)
print("Summary:", response.content)


generic_template = '''
Write a summary of the following speech:
Speech: `{speech}`
Translate the precise summary to {language}.
'''

prompt = PromptTemplate(
    input_variables=['speech', 'language'],
    template=generic_template
)

# Format and create the complete prompt
complete_prompt = prompt.format(speech=speech, language='English')

# Calculate tokens for the complete prompt
tokens_used = llm.get_num_tokens(complete_prompt)
print(f"Total tokens for complete prompt: {tokens_used}")

# Generate summary in Hindi
llm_chain = LLMChain(llm=llm, prompt=prompt)
summary = llm_chain.run({'speech': speech})
print("Summary of Priyanka Chopra Speech:", summary)


