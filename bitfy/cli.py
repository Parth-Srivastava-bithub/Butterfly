import click
import pyfiglet
import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings   # ✅ new import

# --- CONFIGURATION ---
CONFIG_FILE = "bitfy_config.json"
HISTORY_FILE = "bitfy_chat_history.json"
FAISS_INDEX_PATH = "bitfy_faiss_index"

# --- CORE FUNCTIONS ---

def save_config(config_data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chat_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

# --- MEMORY & VECTOR STORE FUNCTIONS ---

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ updated

def update_vector_store(history):
    if not history:
        return

    formatted_texts = [f"User asked: {item['prompt']}\nBitfy answered: {item['response']}" for item in history]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(formatted_texts)

    if not docs:
        return
        
    embeddings = get_embedding_model()

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(docs)
        except Exception as e:
            print(f"⚠️ Could not load existing FAISS index, creating a new one. Error: {e}")
            vector_store = FAISS.from_documents(docs, embedding=embeddings)
    else:
        vector_store = FAISS.from_documents(docs, embedding=embeddings)

    vector_store.save_local(FAISS_INDEX_PATH)

def retrieve_from_memory(query):
    history = load_chat_history()
    if not history:
        return None, None

    recent_chat = history[-1]
    recent_context = f"User asked: {recent_chat['prompt']}\nBitfy answered: {recent_chat['response']}"

    embeddings = get_embedding_model()
    relevant_context = ""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever(search_kwargs={'k': 2})
            retrieved_docs = retriever.invoke(query)
            relevant_context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"⚠️ Error retrieving from FAISS index: {e}")

    return recent_context, relevant_context

# --- CLI MAIN ENTRYPOINT ---

@click.command()
@click.option("--setapi", help="Set your Groq API key.")
@click.option("--ask", help="Ask a question to Bitfy AI Assistant.")
def main(setapi, ask):
    """Bitfy CLI - An AI Assistant with Memory"""
    if setapi:
        config = load_config()
        config["GROQ_API_KEY"] = setapi
        save_config(config)
        print("✅ API key saved successfully!")
        return

    if ask:
        click.echo(click.style(pyfiglet.figlet_format("Bitfy CLI", font="slant"), fg="cyan"))

        config = load_config()
        api_key = config.get("GROQ_API_KEY")
        if not api_key:
            print("❌ API key not found. Use '--setapi <your_key>' to set it.")
            return

        recent_memory, relevant_memory = retrieve_from_memory(ask)
        
        system_prompt = (
            "You are a CLI expert. Always answer as a Windows command-line specialist. "
            "Only give CLI relevant outputs. Be concise and accurate."
        )
        
        if relevant_memory:
            system_prompt += (
                f"\n\nHere is some potentially relevant context from our past conversations:\n"
                f"--- START OF RELEVANT CONTEXT ---\n{relevant_memory}\n--- END OF RELEVANT CONTEXT ---"
            )
        
        if recent_memory:
            system_prompt += (
                f"\n\nFor immediate context, here was our last exchange:\n"
                f"--- START OF LAST EXCHANGE ---\n{recent_memory}\n--- END OF LAST EXCHANGE ---"
            )

        try:
            chat = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=ask)
            ]

            print("🤖 Bitfy is thinking...")
            response = chat.invoke(messages)
            print("\n🤖 Bitfy:", response.content)

            history = load_chat_history()
            history.append({"prompt": ask, "response": response.content})
            save_chat_history(history)
            update_vector_store([history[-1]])

        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
