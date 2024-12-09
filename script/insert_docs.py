from functools import reduce

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
import os

from config import config

metadata_fields = ['source', 'category']

documents_path_list = ['./data/Critique of Pure Reason.md', './data/The Republic.md']

def load_documents(file_path: str, splitter: TextSplitter) -> list[Document]:
    documents = UnstructuredMarkdownLoader(
        file_path,
        mode="elements",
        strategy="fast",
        text_splitter = splitter
    ).load()
    for doc in documents:
        keys_to_remove = [key for key in doc.metadata.keys() if key not in metadata_fields]
        for key in keys_to_remove:
            doc.metadata.pop(key)
    return documents

def load_and_merge_documents(file_paths: list[str], splitter: TextSplitter) -> list[Document]:
    documents_lists = map(lambda path: load_documents(path, splitter), file_paths)
    return reduce(lambda all_docs, current_docs: all_docs + current_docs, documents_lists)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.loader('chunk_size'),
    chunk_overlap=config.loader('chunk_overlap'),
    length_function=len,
    is_separator_regex=False,
)


def main():
    documents = load_and_merge_documents(documents_path_list, text_splitter)
    os.environ['OPENAI_API_KEY'] = config.embedding('api-key')
    embeddings = OpenAIEmbeddings(model=config.embedding('model'))
    vector_store = Milvus(embedding_function=embeddings,
                          connection_args={"uri": config.milvus('uri')},
                          auto_id=True,
                          collection_name=config.milvus('collection'),
                          enable_dynamic_field=True)
    vector_store.add_documents(documents)
