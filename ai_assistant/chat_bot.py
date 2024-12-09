from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ai_assistant import vector_store, model


class Chatbot:

    template = """
    Based on: {retrieved_docs}
    Question: {user_question}
    Chat history: {chat_history}
    """

    def answer_question(self, user_query:str, history: list[dict[str, str]]):
        retrieved_docs = vector_store.similarity_search(user_query, k = 1)
        prompt = ChatPromptTemplate.from_template(self.template)
        for doc in retrieved_docs: print(doc.page_content)
        chain = prompt | model | StrOutputParser()
        return chain.stream(
            {
                # "retrieved_docs": retrieved_docs,
                "user_question": user_query,
                "chat_history": history
            }
        )
