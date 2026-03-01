import os
from dotenv import load_dotenv
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


load_dotenv()  

# Determine which collection to use based on the environment variable
if os.getenv("USE_BRIGHTDATA", "False") == "True":
    collection_name = os.getenv("COLLECTION_NAME_BRIGHTDATA")
else:
    collection_name = os.getenv("COLLECTION_NAME_FREE")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

# Initialize the language model
llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

# System prompt for the chatbot
system_prompt = """

You are a helpful Pokémon expert assistant with access to a knowledge base.

You will receive:
1. Retrieved information from the knowledge base (with source URLs).
2. A user question.

Your task:
- Provide concise and informative responses based on the retrieved information.
- Always cite sources at the end of your response in this format: "Source: <URL>".
- If the retrieved information doesn't help answer the question, respond with "I don't know" instead of guessing, without providing a source.

"""

# Streamlit app configuration
st.set_page_config(page_title="Stateless Pokémon Chatbot", page_icon="icon.png")

# Display icon and title
col1, col2 = st.columns([1, 9])
with col1:
    st.markdown('<div>', unsafe_allow_html=True)
    st.image("icon.png")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.title("Stateless Pokémon Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Create the text input box for the user to ask questions
user_question = st.chat_input("Ask me anything about Pokémon!")

if user_question:

    # Display the user question in the chat
    with st.chat_message("user"):
        st.markdown(user_question)

    # Show a spinner while retrieving information and generating the response
    with st.spinner("Searching knowledge base and generating response..."):
        # Retrieve relevant documents from vector store
        retrieved_docs = vector_store.similarity_search(user_question, k=2)

        # Format the retrieved information
        context = ""
        for doc in retrieved_docs:
            context += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n"

        # Build the messages list: system prompt + current message (context + question)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Retrieved Information:\n{context}\nUser Question: {user_question}")
        ]

        # Get response from LLM
        response = llm.invoke(messages)

    # Display the LLM response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    # Store messages in session state for display purposes only
    st.session_state.messages.append(HumanMessage(content=user_question))
    st.session_state.messages.append(AIMessage(content=response.content))
