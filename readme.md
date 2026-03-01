<h1>Build a local RAG with Ollama</h1>

<h2>Inspired by Thomas Janssen's repository "Local-RAG-with-Ollama"</h2>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>

<h2>Installation</h2>
<h3>1. Clone the repository:</h3>

```
git clone https://github.com/Lelpi/Local-RAG.git
cd Local-RAG
```

<h3>2. Create a virtual environment</h3>

```
python -m venv venv
```

<h3>3. Activate the virtual environment</h3>

```
Windows: venv\Scripts\Activate
Mac/Linux: source venv/bin/activate
```

<h3>4. Install libraries</h3>

```
pip3 install -r requirements.txt
```

<h3>5. Rename the .env.example file to .env</h3>

<h3>6. This chatbot is supposed to be a Pokémon expert. In case you want to select another topic then modify the keywords.json file</h3>

<h3>7. Decide if you want to use the paid Wikipedia scraper or the free one</h3>

- **7.a** In case you want to use the BrighData API (paid version) then set USE_BRIGHTDATA = "True" in the .env file. Then you must get your API key, you can use this link to get a free trial: https://brdta.com/tomstechacademy (provided by Thomas Janssen). Finally, replace your real key in the BRIGHTDATA_API_KEY envrironment variable in the .env file
- **7.b** If you want to stick to the free version (works as good as the paid one) then you don't need to do anything in this section

<h2>Executing the scripts</h2>

- Open a terminal

- Execute the following command:

```
python3 1_scraping_wikipedia.py
python3 2_chunking_embedding_ingestion.py
streamlit run 3_chatbot.py (basic version — performs a direct similarity search on each question independently. Has no memory, so follow-up questions like "tell me more" will not work).
streamlit run 3_memory_chatbot.py (memory version — stores the full conversation history and uses the LLM to rewrite each follow-up question into a standalone query before searching, enabling coherent multi-turn conversations).
streamlit run 3_agentic_chatbot.py (agentic version — uses an agent that autonomously decides when and how to call a retrieval tool to fetch relevant knowledge base content. Handles follow-up questions by enriching the search query with context from the conversation history).
```