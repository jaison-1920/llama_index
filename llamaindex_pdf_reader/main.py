from llama_index.core import SimpleDirectoryReader,StorageContext,DocumentSummaryIndex
from llama_index.core import load_index_from_storage,Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os
from dotenv import load_dotenv
from huggingface_hub import login
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

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
)

PERSIST_DIR = "./db"
OUTPUT_DIR = "./output"

if not  os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#if the persist directory does not exist, then create an new one
# load the document, then create the index

if not os.path.exists(PERSIST_DIR):
    #if there is no persist directory, create one 
    documents = []
    for filename in os.listdir("./data"):
        if filename.endswith(".pdf"):
            filepath = os.path.join("./data", filename)
            #loading the pdf
            documents += SimpleDirectoryReader(input_files=[filepath]).load_data()
    
    #creating the nodes
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=256)
    nodes = splitter.get_nodes_from_documents(documents)

    #create the vector store
    vector_store = QdrantVectorStore(client=client, collection_name="llama_index")

    #storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #create the index and save to PERSIST_DIR
    doc_summ_index = DocumentSummaryIndex(nodes, storage_context=storage_context)
    doc_summ_index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    #load the index from storage
    vector_store = QdrantVectorStore(client=client, collection_name="llama_index")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR,vector_stores={"default": vector_store})
    doc_summ_index = load_index_from_storage(storage_context=storage_context)


#query engine
query_engine = doc_summ_index.as_query_engine(
    response_mode="tree_summarize")

#query the index
questions = [
    "What is the main premise or message of the book Ikigai?",
    "What are the four key questions the book suggests asking yourself to find your ikigai?",
    "How does the book describe the concept of ikigai and its relationship to finding purpose and meaning in life?",
    "What are some of the key principles or practices the book recommends for cultivating ikigai?",
    "Does the book provide any real-world examples or case studies of people who have discovered their ikigai?"
]

for query in questions:

    try:
        response = query_engine.query(query)
        print(f"query: {query}")
        print(f"response: {response}")

        #save the response as markdown
        markdown_file_path = os.path.join(OUTPUT_DIR,"ikigai_questions_answers.md")
        with open(markdown_file_path,"a") as file:
            file.write(f"##Question##: {query}\n\n")
            file.write(f"**response**: {response}\n\n")

    except Exception as e:
        print(f'error occured during processing')


    