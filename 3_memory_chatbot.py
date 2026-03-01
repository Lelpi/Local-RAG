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

# Query reformulation prompt
query_reformulation_prompt = """

Given the conversation history and a follow-up question, reformulate the follow-up question to be a standalone question that includes all necessary context.

If the question is already standalone (doesn't reference previous conversation), return it as is.

Examples:
- History: "What is Pikachu?" -> "Pikachu is an Electric-type Pokémon..."
  Follow-up: "What type is it?" -> Standalone: "What type is Pikachu?"
  
- History: "Tell me about Charizard" -> "Charizard is a Fire/Flying type..."
  Follow-up: "Does it evolve?" -> Standalone: "Does Charizard evolve?"
  
- Follow-up: "What is Ditto?" -> Standalone: "What is Ditto?" (already standalone)

Return ONLY the reformulated question, nothing else.

"""

# Streamlit app configuration
st.set_page_config(page_title="Stateful Pokémon Chatbot", page_icon="icon.png")

# Display icon and title
col1, col2 = st.columns([1, 9])
with col1:
    st.markdown('<div>', unsafe_allow_html=True)
    st.image("icon.png")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.title("Stateful Pokémon Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            # Display only the original question
            display_content = message.content.split(" Reformulated question:")[0].replace("Original question: ", "")
            st.markdown(display_content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Create the text input box for the user to ask questions
user_question = st.chat_input("Ask me anything about Pokémon!")

if user_question:

    # Display the original user question in the chat
    with st.chat_message("user"):
        st.markdown(user_question)

    # Reformulate the query if it's a follow-up question
    if len(st.session_state.messages) > 0:

        # Build context from recent conversation history (last 6 messages, i.e.: last 3 questions and answers)
        recent_history = st.session_state.messages[-6:]
        history_text = ""
        for msg in recent_history:
            if isinstance(msg, HumanMessage):
                # Extract only the reformulated question for context, not the original one
                history_text += f"User: {msg.content.split('Reformulated question: ')[-1]}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n\n"

        # Ask LLM to reformulate the query
        reformulation_messages = [
            SystemMessage(content=query_reformulation_prompt),
            HumanMessage(content=f"Conversation history:\n{history_text}\n\nFollow-up question: {user_question}\n\nStandalone question: ?")
        ]
        with st.spinner("Understanding your question..."):
            reformulated_response = llm.invoke(reformulation_messages)
            search_query = reformulated_response.content.strip()
    else:
        search_query = user_question

    # Show a spinner while retrieving information and generating the response
    with st.spinner("Searching knowledge base and generating response..."):
        # Retrieve relevant documents from vector store using the reformulated query
        retrieved_docs = vector_store.similarity_search(search_query, k=2)

        # Format the retrieved information
        context = ""
        for doc in retrieved_docs:
            context += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n"

        # Build the messages list: system prompt + current message (context + standalone question)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Retrieved Information:\n{context}\nUser Question: {search_query}")
        ]

        # Get response from LLM
        response = llm.invoke(messages)

    # Display the LLM response
    with st.chat_message("assistant"):
        st.markdown(response.content)

    # Store messages in session state for display purposes and to keep track of original vs reformulated questions for future reformulations
    st.session_state.messages.append(HumanMessage(content=f"Original question: {user_question} Reformulated question: {search_query}"))
    st.session_state.messages.append(AIMessage(content=response.content))
