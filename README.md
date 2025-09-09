

# Bitfy CLI – AI Assistant with Memory

Bitfy CLI is an AI-powered command-line assistant designed to provide **accurate Windows command-line solutions** with persistent memory.
It uses **Groq’s LLaMA models** for fast responses and **MiniLM embeddings + FAISS** for semantic memory retrieval.

## ✨ Features

* 🖥️ **Windows CLI Specialist** → Always generates concise, accurate command-line solutions.
* 🧠 **Two-level memory** →

  * Short-term memory (last exchange for immediate context)
  * Long-term vector memory using FAISS for retrieving past conversations
* ⚡ **Groq API Integration** → Ultra-fast inference with LLaMA-3.1 models
* 📚 **Semantic Retrieval** → Splits, embeds, and indexes past Q\&A for smarter recall
* 🔑 **API Key Management** → Easily set and persist your Groq API key
* 🎨 **Interactive CLI** → Clean, figlet-based banner and seamless user experience

## 📦 Installation



Dependencies include:

* `click`
* `pyfiglet`
* `langchain_groq`
* `langchain_core`
* `langchain_huggingface`
* `langchain_community`

## 🔑 Configuration

Set your **Groq API Key** once:

```bash
python bitfy.py --setapi <YOUR_GROQ_API_KEY>
```

This saves the key in `bitfy_config.json`.

## 🚀 Usage

Ask a CLI-related question:

```bash
python bitfy.py --ask "How to list all processes running on Windows?"
```

Example Output:

```
Bitfy CLI

🤖 Bitfy is thinking...

🤖 Bitfy: 
Use the following command to list processes:
tasklist
```

## 💾 Memory System

* **Chat History** → Stored in `bitfy_chat_history.json`
* **Vector Store** → Stored locally in `bitfy_faiss_index/` using FAISS
* **Update cycle** → Every new Q\&A is embedded, indexed, and added to long-term memory

## 🛠️ Project Structure

```
bitfy-cli/
│
├── bitfy.py                  # Main CLI script
├── bitfy_config.json         # Stores API key
├── bitfy_chat_history.json   # Stores past conversations
├── bitfy_faiss_index/        # FAISS vector store for memory
└── requirements.txt          # Dependencies
```

## ⚡ Future Improvements

* Command auto-run mode with safety confirmations
* Multi-platform support (Linux/Mac commands)
* Session tagging for project-specific memories
* Richer semantic retrieval with better embedding models

## 📜 License

MIT License

