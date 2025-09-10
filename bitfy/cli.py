import click
import pyfiglet
import os
import json
import shutil
import subprocess
import re
import threading
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION & CONSTANTS ---
CONFIG_FILE = "bitfy_config.json"
HISTORY_FILE = "bitfy_chat_history.json"
FAISS_INDEX_PATH = "bitfy_faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
MAX_CONTEXT_LENGTH = 4000  # Increased slightly for better context

# --- GLOBAL CACHE ---
# ✅ PERFORMANCE: Cache the embedding model to avoid reloading it on every run.
_embedding_model = None

# --- CORE UTILITIES ---

def save_config(config_data: dict):
    """Saves the configuration data to a JSON file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

def load_config() -> dict:
    """Loads configuration data from a JSON file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chat_history(history: list):
    """Saves the chat history to a JSON file."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_chat_history() -> list:
    """Loads chat history from a JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

# --- MEMORY & VECTOR STORE FUNCTIONS ---

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the embedding model, using a global cache for performance.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model

def update_vector_store_async(new_history_item: dict):
    """
    ✅ PERFORMANCE: Runs the vector store update in a separate thread.
    """
    thread = threading.Thread(target=update_vector_store, args=(new_history_item,))
    thread.start()

def update_vector_store(new_history_item: dict):
    """Updates the FAISS vector store with the latest conversation."""
    if not new_history_item:
        return
    
    formatted_text = f"User asked: {new_history_item['prompt']}\nBitfy answered: {new_history_item['response']}"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.create_documents([formatted_text])
    if not docs:
        return
        
    embeddings = get_embedding_model()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(docs)
        except Exception:
            vector_store = FAISS.from_documents(docs, embedding=embeddings)
    else:
        vector_store = FAISS.from_documents(docs, embedding=embeddings)

    vector_store.save_local(FAISS_INDEX_PATH)

def retrieve_context(query: str) -> str:
    """Retrieves context from history and vector store, optimized for speed."""
    history = load_chat_history()
    if not history:
        return ""

    recent_context = f"Last exchange:\n- User: {history[-1]['prompt']}\n- Bitfy: {history[-1]['response']}"
    
    relevant_context = ""
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            embeddings = get_embedding_model()
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever(search_kwargs={'k': 3})
            retrieved_docs = retriever.invoke(query)
            if retrieved_docs:
                formatted_docs = [doc.page_content for doc in retrieved_docs]
                relevant_context = "Relevant past exchanges:\n" + "\n---\n".join(formatted_docs)
        except Exception:
            pass # Fail silently if context retrieval fails

    full_context = f"{relevant_context}\n\n{recent_context}"
    return full_context[-MAX_CONTEXT_LENGTH:]

# --- COMMAND EXECUTION ---

def run_command(command: str) -> str:
    """Executes a shell command and returns its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return f"✅ Command executed successfully:\n--- OUTPUT ---\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"❌ Command failed:\n--- ERROR ---\n{e.stderr}"

# --- CLI MAIN ENTRYPOINT ---

@click.command()
@click.option("--setapi", help="Set your Groq API key.", type=str)
@click.option("--ask", help="Ask a question to Bitfy AI Assistant.", type=str)
@click.option("--reset", is_flag=True, help="Reset chat history and memory.")
def main(setapi, ask, reset):
    """Bitfy CLI Agent - A supercharged AI assistant that can execute commands."""
    
    if setapi:
        config = load_config()
        config["GROQ_API_KEY"] = setapi
        save_config(config)
        click.secho("✅ API key saved successfully!", fg="green")
        return

    if reset:
        if click.confirm("Are you sure you want to delete all chat history and memory?"):
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            if os.path.exists(FAISS_INDEX_PATH): shutil.rmtree(FAISS_INDEX_PATH)
            click.secho("🗑️ Chat history and memory have been reset.", fg="yellow")
        return

    if not ask:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return
        
    click.echo(click.style(pyfiglet.figlet_format("Bitfy CLI", font="slant"), fg="cyan"))

    config = load_config()
    api_key = config.get("GROQ_API_KEY")
    if not api_key:
        click.secho("❌ API key not found. Use '--setapi <your_key>' to set it.", fg="red")
        return

    context = retrieve_context(ask)

    # ✅ NEW: The system prompt now instructs the model to use <execute> tags.
    system_prompt = (
        "You are Bitfy, a powerful AI assistant for the Windows command line. Your goal is to help the user accomplish tasks. "
        "1. First, think about the user's request. "
        "2. If a command is needed, provide ONLY the command inside <execute> tags. For example: <execute>tasklist</execute>. Do not add any other text or explanation. "
        "3. If the user is asking a question or the request doesn't need a command, provide a concise, helpful answer without any tags."
    )
    if context:
        system_prompt += f"\n\n## Context from past conversations:\n{context}"

    try:
        chat = ChatGroq(api_key=api_key, model_name=LLM_MODEL_NAME)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=ask)]

        click.echo("🤖 Bitfy is thinking...")
        response = chat.invoke(messages)
        response_content = response.content.strip()

        # ✅ NEW: Command execution logic
        execute_match = re.search(r"<execute>(.*?)</execute>", response_content, re.DOTALL)
        
        if execute_match:
            command_to_run = execute_match.group(1).strip()
            click.secho(f"\n🤖 Bitfy wants to run:", fg="yellow")
            click.secho(f"   {command_to_run}", fg="white", bold=True)
            
            if click.confirm("Do you want to execute this command?", default=True):
                command_output = run_command(command_to_run)
                click.secho("\n" + command_output, fg="cyan")
                final_response = command_output
            else:
                click.secho("Skipped command execution.", fg="red")
                final_response = "User chose not to execute the command."
        else:
            # No command found, just display the text response
            click.secho("\n🤖 Bitfy:", fg="green", bold=True)
            click.echo(response_content)
            final_response = response_content

        # 4. Update history and vector store asynchronously
        new_item = {"prompt": ask, "response": final_response}
        history = load_chat_history()
        history.append(new_item)
        save_chat_history(history)
        update_vector_store_async(new_item) # Non-blocking call

    except Exception as e:
        click.secho(f"❌ An error occurred: {e}", fg="red")

if __name__ == "__main__":
    main()