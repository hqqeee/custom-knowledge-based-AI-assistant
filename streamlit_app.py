import streamlit as st
from ai_assistant.chat_bot import Chatbot
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Custom-knowledge AI Assistant")

st.title("💬 Chatbot")



chatbot = Chatbot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

if prompt := st.chat_input("Type your message here..."):
    st.session_state.chat_history.append(HumanMessage(prompt))
    with st.chat_message("Human"):
        st.markdown(prompt)
    with st.chat_message("AI"):
        response = st.write_stream(chatbot.answer_question(prompt, st.session_state.chat_history))
    st.session_state.chat_history.append(AIMessage(response))

