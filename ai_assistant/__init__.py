import os

from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import config


def init_vector_store():
    os.environ['OPENAI_API_KEY'] = config.embedding('api-key')
    embeddings = OpenAIEmbeddings(model=config.embedding('model'))
    return Milvus(
        embedding_function=embeddings,
        connection_args={"uri": config.milvus('uri')},
        auto_id=True,
        collection_name=config.milvus('collection')
    )

vector_store: VectorStore = init_vector_store()


model = ChatOpenAI(
    model = config.chat_model('model'),
    temperature = config.chat_model('temperature'),
    max_tokens = config.chat_model('max_tokens')
)