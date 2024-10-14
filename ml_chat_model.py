import os
import openai
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
_ = load_dotenv(find_dotenv()) # read local .env file

# TODO :--- Credentials should be stored in a .env file, see if you can manage to get the dotenv file to work, update the README if you do ---
# Retrieve the environment variables
# openai_api_key = os.getenv('OPENAI_API_KEY')
# langchain_tracing_v2 = os.getenv('LANGCHAIN_TRACING_V2')
# langchain_endpoint = os.getenv('LANGCHAIN_ENDPOINT')
# langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
#
# # Set environment variables
# os.environ['OPENAI_API_KEY'] = openai_api_key
# os.environ['LANGCHAIN_TRACING_V2'] = langchain_tracing_v2
# os.environ['LANGCHAIN_ENDPOINT'] = langchain_endpoint
# os.environ['LANGCHAIN_API_KEY'] = langchain_api_key

os.environ['OPENAI_API_KEY'] = "sk-proj-mZwfrXnSqmOXrmWt5ZOfT3BlbkFJytXcLAwKSBLt2JTfoNjU"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_56f4fc3fb17f4120a7d30cf22c201d50_a6dc18b3f4Y"

sys.path.append('../..')


class MLChatModel:
    def __init__(self, prompt: str):
        """
        Initialize the model with the prompt.
        """
        self.prompt = prompt

    def run(self):
        """
        Run the model and return the result.
        """
        # Load the PDF file
        loader = PyPDFLoader("../mental_health_in_uk.pdf")
        pages = loader.load()

        # Directory to persist vector embeddings and data to
        persist_directory = './TEST/'

        # Create embeddings object
        embedding = OpenAIEmbeddings()

        # Split documents into chunks and overlap for context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 150
        )
        splits = text_splitter.split_documents(pages)

        # Create vector database
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )

        # Create a memory buffer to store conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create vector db retriever
        retriever = vectordb.as_retriever()

        # Use conversational retrieval chain for interacting with the llm
        qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
            chain_type="stuff",
            retriever=retriever,
            memory=memory
        )

        # Send question to qa model
        question = self.prompt
        result = qa({"question": question})
        return result['answer']