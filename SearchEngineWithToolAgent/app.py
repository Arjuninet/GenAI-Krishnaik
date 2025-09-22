import streamlit as st
from langchain_groq import ChatGroq
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

os.environ["GROQ_API_KEY "] = os.getenv("GROQ_API_KEY")

#wikipedia tool
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_char_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, verbose=True)
wiki.name

# arxiv tool
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_char_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, verbose=True)

# Search Duck Duck go tools
search = DuckDuckGoSearchRun(name="Search")


st.title("Langchain - Chat with search")

#Slide bar for setting
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your API Groq API key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant","content":"Hi I'm a chatbot"}]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine Learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)
    tools = [search,arxiv,wiki]

    search_agent = initialize_agent(tools,llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(prompt,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)



