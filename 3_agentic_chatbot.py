import os
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool


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

# Initialize the language model for the agent
llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

system_prompt = """
You are a helpful Pokémon expert assistant with access to a knowledge base.

TOOLS:
- You have access to a 'retrieve' tool that searches a Pokémon knowledge base.
- Always call 'retrieve' exactly once per user question, passing the user's question as a plain text string.
- For follow-up questions that reference previous context (e.g. "tell me more", "what about its evolutions?"), enrich the query with the relevant topic from the chat history before calling 'retrieve'.

RESPONDING:
- Base your answer strictly on the retrieved content and the conversation history.
- Be concise and informative.
- Always cite your source at the end in this format: "Source: <URL>"
- If the retrieved content is empty or irrelevant, respond with "I don't know" and omit the source.
- Never fabricate information or URLs.
"""

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Defining the tool that the agent will use to retrieve relevant information from the vector store based on the query.
# The tool takes a query as input, performs a similarity search in the vector store,
#   and returns a string containing the content and source URL of the k retrieved documents.
@tool
def retrieve(query) -> str:
    """Retrieve information related to a query. The query must be a plain text string with the user's question."""

    # Guard: LLM sometimes passes a schema dict instead of a plain string
    if isinstance(query, dict):
        for key in ("value", "query", "question", "text", "q", "input"):
            if key in query:
                query = query[key]
                break
        else:
            query = " ".join(str(v) for v in query.values() if isinstance(v, str))
    query = str(query).strip()

    context = ""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    for doc in retrieved_docs:
        context += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n"
    return context

# Create the agent and the agent executor
agent = create_tool_calling_agent(llm, [retrieve], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[retrieve], verbose=True, handle_parsing_errors=True)

# Streamlit app configuration
st.set_page_config(page_title="Pokémon Agentic Chatbot", page_icon="icon.png")

# Display icon and title
col1, col2 = st.columns([1, 9])
with col1:
    st.markdown('<div>', unsafe_allow_html=True)
    st.image("icon.png")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.title("Pokémon Agentic Chatbot")

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
        # Invoke the agent executor with the user's question and the chat history
        result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.messages})
        ai_message = result["output"]

    # Add the agent's response to the chat history and display it on the screen
    with st.chat_message("assistant"):
        st.markdown(ai_message)

    # Store messages in session state for display purposes and to keep track of the conversation history for the agent's context
    if ai_message.strip():
        st.session_state.messages.append(HumanMessage(user_question))
        st.session_state.messages.append(AIMessage(ai_message))
