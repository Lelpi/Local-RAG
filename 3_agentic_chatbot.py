import os
from dotenv import load_dotenv
import streamlit as st
from langchain.agents import AgentExecutor
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


load_dotenv()  

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

# Initialize the language model for the agent
llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

# Define the prompt template for the agent
prompt = PromptTemplate.from_template("""                                
You are a helpful assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store and provide a response.
For this you use the tool 'retrieve' to get the relevant information.
                                      
The query is as follows:                    
{input}

The chat history is as follows:
{chat_history}

Please provide a concise and informative response based on the retrieved information and also include the source.
If you don't know the answer, say "I don't know" and don't provide any source.
                                      
You can use the scratchpad to store any intermediate results or notes.
The scratchpad is as follows:
{agent_scratchpad}

Return text as follows:

<Answer to the question>
Source: source_url
""")


# Defining the tool that the agent will use to retrieve relevant information from the vector store based on the query.
# The tool takes a query as input, performs a similarity search in the vector store,
#   and returns a serialized string containing the content and source URL of the k retrieved documents.
@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    serialized = ""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    for doc in retrieved_docs:
        serialized += f"Source: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"
    return serialized


# Create the agent and the agent executor.
agent = create_tool_calling_agent(llm, [retrieve], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[retrieve], verbose=True)

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

    # Add the user's question to the chat history and display it on the screen
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    # Invoke the agent executor with the user's question and the chat history, and get the agent's response
    result = agent_executor.invoke({"input": user_question, "chat_history": st.session_state.messages})
    ai_message = result["output"]

    # Add the agent's response to the chat history and display it on the screen
    with st.chat_message("assistant"):
        st.markdown(ai_message)
        st.session_state.messages.append(AIMessage(ai_message))

