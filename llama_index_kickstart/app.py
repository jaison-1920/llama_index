from llama_index.core import SimpleDirectoryReader,ServiceContext,StorageContext,VectorStoreIndex,load_index_from_storage,Settings
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser

#to use huggingface model to run on huggingface server import HFinferenceapi
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

#to run embeddings from huggingface, first we need to import langchainembedding
#and through langchainembedding we access HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN

#logging in  to huggingface hub
login(HF_TOKEN)

#Persist directory to use the same index for multiple runs
PERSIST_DIR = "./db"

llm = HuggingFaceInferenceAPI(
    model = "HuggingFaceH4/zephyr-7b-beta",
    api_key = HF_TOKEN,
    task="text-generation"
)

embed_model = LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        api_key=HF_TOKEN
    )
)
Settings.llm = llm
Settings.embed_model = embed_model

#if the persist directory does not exist, then create an new one
# load the document, then create the index
if not os.path.exists(PERSIST_DIR):
    document = SimpleDirectoryReader('./data').load_data()
    index = VectorStoreIndex.from_documents(document)
    #store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
#if the persist directory exists, then load the index    
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)


#query engine
query_engine = index.as_query_engine()
response = query_engine.query("Who is the friend mentioned in this speech?")
print(response)
