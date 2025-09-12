### **Imports**

```python
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
```

* `click`: Creates command-line interfaces (CLI) easily with options and flags.
* `pyfiglet`: Converts text to fancy ASCII art for terminal display.
* `os`: Handles operating system tasks like checking files, directories, paths.
* `json`: Reads/writes JSON files for config or chat history.
* `shutil`: Advanced file operations like deleting folders recursively.
* `subprocess`: Runs system shell commands and captures their outputs/errors.
* `re`: Regex library for searching/extracting patterns from text.
* `threading`: Runs tasks concurrently in background threads for performance.
* `langchain_groq.ChatGroq`: Connects with the Groq LLM model for AI responses.
* `langchain_core.messages`: Defines message types for LLM conversations.
* `RecursiveCharacterTextSplitter`: Splits large text into smaller chunks for vector embeddings.
* `FAISS`: Vector database for similarity search of embeddings.
* `HuggingFaceEmbeddings`: Converts text into embeddings using HuggingFace models.

---

### **Configuration Constants**

```python
CONFIG_FILE = "bitfy_config.json"
HISTORY_FILE = "bitfy_chat_history.json"
FAISS_INDEX_PATH = "bitfy_faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
MAX_CONTEXT_LENGTH = 4000
```

* `CONFIG_FILE`: Path to store API key and other config.
* `HISTORY_FILE`: Stores chat history as JSON.
* `FAISS_INDEX_PATH`: Folder where vector embeddings of conversation are saved.
* `EMBEDDING_MODEL_NAME`: HuggingFace model to convert text into embeddings.
* `LLM_MODEL_NAME`: The large language model Bitfy will use to generate responses.
* `MAX_CONTEXT_LENGTH`: Limits how much past context to include when asking AI (for speed & memory).

---

### **Global Cache**

```python
_embedding_model = None
```

* Global variable to **cache embedding model**, so you don’t reload it every time → speeds things up.

---

### **Configuration Helpers**

```python
def save_config(config_data: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}
```

* `save_config`: Writes a Python dict to `bitfy_config.json`.
* `load_config`: Reads config file if exists, otherwise returns empty dict.

---

### **Chat History Helpers**

```python
def save_chat_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_chat_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []
```

* `save_chat_history`: Stores all conversation items in JSON.
* `load_chat_history`: Loads JSON chat history safely; returns empty list if file missing or corrupted.

---

### **Embedding Model Loader**

```python
def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding_model
```

* Checks if `_embedding_model` exists:

  * If not, it **initializes the HuggingFace embedding model**.
  * Returns the cached model otherwise.

---

### **Vector Store (FAISS) Updates**

```python
def update_vector_store_async(new_history_item: dict):
    thread = threading.Thread(target=update_vector_store, args=(new_history_item,))
    thread.start()
```

* Runs `update_vector_store` in **background thread** → non-blocking so CLI stays responsive.

```python
def update_vector_store(new_history_item: dict):
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
```

* Converts new chat into a text format.
* Splits into chunks (1000 chars, 150 overlap).
* Loads or creates FAISS index.
* Adds new embeddings.
* Saves updated FAISS index locally.

---

### **Retrieve Context**

```python
def retrieve_context(query: str) -> str:
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
            pass 

    full_context = f"{relevant_context}\n\n{recent_context}"
    return full_context[-MAX_CONTEXT_LENGTH:]
```

* Loads recent chat history.
* Fetches **relevant past conversations** using FAISS search (top 3 similar).
* Combines recent and relevant context.
* Limits output to `MAX_CONTEXT_LENGTH`.

---

### **Run Shell Command**

```python
def run_command(command: str) -> str:
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
```

* Runs system commands safely.
* Captures stdout/stderr.
* Returns success or error formatted for CLI.

---

### **CLI Main Function**

```python
@click.command()
@click.option("--setapi", help="Set your Groq API key.", type=str)
@click.option("--ask", help="Ask a question to Bitfy AI Assistant.", type=str)
@click.option("--reset", is_flag=True, help="Reset chat history and memory.")
def main(setapi, ask, reset):
```

* Defines **Bitfy CLI entrypoint** with 3 options:

  * `--setapi`: Save API key.
  * `--ask`: Ask question to AI.
  * `--reset`: Clear chat & FAISS memory.

---

### **API Key Handling**

```python
if setapi:
    config = load_config()
    config["GROQ_API_KEY"] = setapi
    save_config(config)
    click.secho("✅ API key saved successfully!", fg="green")
    return
```

* Saves API key to config JSON and exits CLI.

---

### **Reset Option**

```python
if reset:
    if click.confirm("Are you sure you want to delete all chat history and memory?"):
        if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
        if os.path.exists(FAISS_INDEX_PATH): shutil.rmtree(FAISS_INDEX_PATH)
        click.secho("🗑️ Chat history and memory have been reset.", fg="yellow")
    return
```

* Deletes chat history JSON & FAISS index folder safely after confirmation.

---

### **Default Help**

```python
if not ask:
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    return
```

* If no input given, shows CLI help.

---

### **Display Banner**

```python
click.echo(click.style(pyfiglet.figlet_format("Bitfy CLI", font="slant"), fg="cyan"))
```

* Fancy ASCII banner for terminal display.

---

### **Load API Key**

```python
config = load_config()
api_key = config.get("GROQ_API_KEY")
if not api_key:
    click.secho("❌ API key not found. Use '--setapi <your_key>' to set it.", fg="red")
    return
```

* Checks for API key before calling Groq LLM.

---

### **System Prompt + Context**

```python
context = retrieve_context(ask)
system_prompt = (
    "You are Bitfy, a powerful AI assistant for the Windows command line. Your goal is to help the user accomplish tasks. "
    "1. First, think about the user's request. "
    "2. If a command is needed, provide ONLY the command inside <execute> tags. For example: <execute>tasklist</execute>. Do not add any other text or explanation. "
    "3. If the user is asking a question or the request doesn't need a command, provide a concise, helpful answer without any tags."
)
if context:
    system_prompt += f"\n\n## Context from past conversations:\n{context}"
```

* Creates LLM instruction + adds past context.
* `<execute>` tags guide AI for safe command execution.

---

### **Call LLM**

```python
chat = ChatGroq(api_key=api_key, model_name=LLM_MODEL_NAME)
messages = [SystemMessage(content=system_prompt), HumanMessage(content=ask)]

click.echo("🤖 Bitfy is thinking...")
response = chat.invoke(messages)
response_content = response.content.strip()
```

* Creates ChatGroq instance.
* Sends system + human messages.
* Gets AI response.

---

### **Parse Commands**

```python
execute_match = re.search(r"<execute>(.*?)</execute>", response_content, re.DOTALL)
        
if execute_match:
    command_to_run = execute_match.group(
```


1\).strip()
click.secho(f"\n🤖 Bitfy wants to run:", fg="yellow")
click.secho(f"   {command\_to\_run}", fg="white", bold=True)

```
if click.confirm("Do you want to execute this command?", default=True):
    command_output = run_command(command_to_run)
    click.secho("\n" + command_output, fg="cyan")
    final_response = command_output
else:
    click.secho("Skipped command execution.", fg="red")
    final_response = "User chose not to execute the command."
```

else:
click.secho("\n🤖 Bitfy:", fg="green", bold=True)
click.echo(response\_content)
final\_response = response\_content

````
- Extracts `<execute>` command using regex.  
- Asks user confirmation before running system commands.  
- If no command, just prints AI response.  

---

### **Update History & Vector Store**
```python
new_item = {"prompt": ask, "response": final_response}
history = load_chat_history()
history.append(new_item)
save_chat_history(history)
update_vector_store_async(new_item)
````

* Adds new Q\&A to JSON.
* Updates FAISS embeddings asynchronously.

---

### **Error Handling**

```python
except Exception as e:
    click.secho(f"❌ An error occurred: {e}", fg="red")
```

* Catches runtime errors gracefully.

---

### **Entry Point**

```python
if __name__ == "__main__":
    main()
```

* Runs CLI when script executed directly.

---

Bhai, basically ye script ek **fully-featured CLI AI assistant** hai:

* Can chat & remember past conversations.
* Can run commands safely via `<execute>` tags.
* Uses **vector embeddings** for relevant context.
* Async updates to keep CLI snappy.

---

