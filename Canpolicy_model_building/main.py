from llama_index.core import SimpleDirectoryReader,StorageContext,VectorStoreIndex
from llama_index.core import load_index_from_storage,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from dotenv import load_dotenv
from huggingface_hub import login
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore


load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

#login to huggingface
login(token=HF_TOKEN)

#set up LLM
Settings.llm = HuggingFaceInferenceAPI(
    model = "HuggingFaceH4/zephyr-7b-beta",
    api_key = HF_TOKEN,
    task="text-generation"
)

#set up embeddings
Settings.embed_model = LangchainEmbedding(
    HuggingFaceInferenceAPIEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        api_key=HF_TOKEN
    )
)
Settings.chunk_size = 1024
Settings.chunk_overlap = 256

#initialize qdrant client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    # path="qdrant.db"
)

#vector database 
vector_store = QdrantVectorStore(client=client, collection_name="llama_index")


file_directory = r"C:\\ML_projects\\llamaindex\\Canpolicy_model_building\\can_extracted_pdf_files_1"

print(f"File directory:{file_directory}")




documents = []
for filename in os.listdir(file_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(file_directory, filename)
        
        documents += SimpleDirectoryReader(input_files=[file_path]).load_data()

splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = splitter.get_nodes_from_documents(documents)

#storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
vector_index.storage_context.persist()

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
)

query = "what was the purpose of creating Online Streaming Act?"

try:
    response = query_engine.query(query)
    print(f"Original Query: {query}")
    print(f"Response: {response}")
except Exception as e:
    print(f"Error during query processing: {str(e)}")





