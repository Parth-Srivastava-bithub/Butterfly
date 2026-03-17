import click
import pyfiglet
import os
import json
import subprocess
import re
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# --- CONFIGURATION & CONSTANTS ---
CONFIG_FILE = "bitfy_config.json"
HISTORY_FILE = "bitfy_chat_history.json"   # Short-term memory (recent exchanges, JSON)
SUMMARY_FILE = "bitfy_memory.txt"          # Long-term memory (summarized past, plain text)
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_SHORT_TERM = 10   # How many recent exchanges to keep in short-term memory


# --- CONFIG ---

def save_config(config_data: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


# --- SHORT-TERM MEMORY (JSON) ---

def load_short_term() -> list:
    """Load recent chat exchanges from JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_short_term(history: list):
    """Save recent chat exchanges to JSON file (keeps last MAX_SHORT_TERM)."""
    trimmed = history[-MAX_SHORT_TERM:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(trimmed, f, indent=4)


# --- LONG-TERM MEMORY (TXT SUMMARY) ---

def load_long_term() -> str:
    """Load the long-term summary from a plain text file."""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def append_long_term(new_summary_line: str):
    """Append a new summary line to the long-term memory file."""
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        f.write(f"[{timestamp}] {new_summary_line}\n")

def summarize_and_archive(history: list, chat: ChatGroq):
    """
    When short-term memory is full, use the LLM to summarize the oldest half
    and move that summary into the long-term memory file.
    """
    if len(history) < MAX_SHORT_TERM:
        return history

    # Take the oldest half to archive
    to_archive = history[: MAX_SHORT_TERM // 2]
    keep = history[MAX_SHORT_TERM // 2 :]

    # Build a condensed text from the old exchanges
    exchanges_text = "\n".join(
        f"User: {item['prompt']}\nBitfy: {item['response']}" for item in to_archive
    )

    try:
        summary_messages = [
            SystemMessage(content="You are a summarizer. Summarize the following conversation in 1-3 concise sentences. Focus on key facts, decisions, or topics discussed."),
            HumanMessage(content=exchanges_text),
        ]
        summary_response = chat.invoke(summary_messages)
        summary_line = summary_response.content.strip().replace("\n", " ")
        append_long_term(summary_line)
    except Exception:
        # Fallback: just note the topics without LLM
        topics = ", ".join(set(item["prompt"][:40] for item in to_archive))
        append_long_term(f"Past topics discussed: {topics}")

    return keep


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
            text=True,
        )
        return f"✅ Command executed successfully:\n--- OUTPUT ---\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"❌ Command failed:\n--- ERROR ---\n{e.stderr}"


# --- BUILD CONTEXT FOR LLM ---

def build_context(short_term: list, long_term: str) -> str:
    """Combines long-term summary and recent short-term exchanges into a context string."""
    parts = []

    if long_term:
        parts.append(f"## Long-term memory (past conversation summaries):\n{long_term}")

    if short_term:
        recent_lines = []
        for item in short_term[-5:]:  # Only last 5 for brevity
            recent_lines.append(f"User: {item['prompt']}\nBitfy: {item['response']}")
        parts.append("## Recent conversation:\n" + "\n---\n".join(recent_lines))

    return "\n\n".join(parts)


# --- CLI ---

@click.command()
@click.option("--setapi", help="Set your Groq API key.", type=str)
@click.option("--setmodel", help=f"Set the LLM model to use (default: {DEFAULT_MODEL}).", type=str)
@click.option("--models", is_flag=True, help="Show available Groq model suggestions.")
@click.option("--ask", help="Ask a question or give a command to Bitfy.", type=str)
@click.option("--reset", is_flag=True, help="Reset all chat history and memory.")
@click.option("--memory", is_flag=True, help="View current short-term and long-term memory.")
def main(setapi, setmodel, models, ask, reset, memory):
    """Bitfy CLI Agent — AI assistant with memory and command execution."""

    # --- SET API KEY ---
    if setapi:
        config = load_config()
        config["GROQ_API_KEY"] = setapi
        save_config(config)
        click.secho("✅ API key saved successfully!", fg="green")
        return

    # --- SET MODEL ---
    if setmodel:
        config = load_config()
        config["MODEL"] = setmodel
        save_config(config)
        click.secho(f"✅ Model set to: {setmodel}", fg="green")
        return

    # --- SHOW MODELS ---
    if models:
        click.secho("\n📋 Popular Groq models you can use with --setmodel:\n", fg="cyan")
        model_list = [
            ("llama-3.1-8b-instant",   "Fast, lightweight — good for most tasks"),
            ("llama-3.3-70b-versatile","Powerful, slower — best for complex tasks"),
            ("llama3-8b-8192",         "Older Llama 3, solid general use"),
            ("llama3-70b-8192",        "Older Llama 3 large model"),
            ("mixtral-8x7b-32768",     "Mixtral, 32k context window"),
            ("gemma2-9b-it",           "Google Gemma 2, good instruction following"),
        ]
        for name, desc in model_list:
            click.secho(f"  {name:<35} ", fg="white", bold=True, nl=False)
            click.secho(desc, fg="bright_black")
        click.echo()
        return

    # --- RESET ---
    if reset:
        if click.confirm("⚠️  Delete ALL chat history and memory?"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            if os.path.exists(SUMMARY_FILE):
                os.remove(SUMMARY_FILE)
            click.secho("🗑️  History and memory cleared.", fg="yellow")
        return

    # --- VIEW MEMORY ---
    if memory:
        click.secho("\n📂 Short-term memory (recent exchanges):\n", fg="cyan", bold=True)
        short = load_short_term()
        if short:
            for i, item in enumerate(short, 1):
                click.secho(f"  [{i}] User: {item['prompt'][:80]}", fg="white")
                click.secho(f"       Bitfy: {item['response'][:80]}", fg="bright_black")
        else:
            click.secho("  (empty)", fg="bright_black")

        click.secho("\n📖 Long-term memory (summaries):\n", fg="cyan", bold=True)
        long = load_long_term()
        if long:
            click.echo(long)
        else:
            click.secho("  (empty)", fg="bright_black")
        click.echo()
        return

    # --- ASK ---
    if not ask:
        click.echo(click.style(pyfiglet.figlet_format("Bitfy CLI", font="slant"), fg="cyan"))
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        return

    click.echo(click.style(pyfiglet.figlet_format("Bitfy CLI", font="slant"), fg="cyan"))

    config = load_config()
    api_key = config.get("GROQ_API_KEY")
    if not api_key:
        click.secho("❌ API key not found. Use '--setapi <your_key>' to set it.", fg="red")
        return

    model_name = config.get("MODEL", DEFAULT_MODEL)
    click.secho(f"🔧 Using model: {model_name}\n", fg="bright_black")

    # Load memory
    short_term = load_short_term()
    long_term = load_long_term()
    context = build_context(short_term, long_term)

    system_prompt = (
        "You are Bitfy, a helpful AI assistant for the command line. "
        "You do two things:\n"
        "1. CHIT-CHAT: If the user is asking a question or having a conversation, reply naturally and helpfully.\n"
        "2. COMMANDS: If the user wants to run a system command, output ONLY the command wrapped in <execute> tags. "
        "Example: <execute>dir</execute>. Do NOT mix text and <execute> tags — pick one or the other.\n"
        "Be concise. Avoid unnecessary filler."
    )

    if context:
        system_prompt += f"\n\n{context}"

    try:
        chat = ChatGroq(api_key=api_key, model_name=model_name)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=ask)]

        click.secho("🤖 Bitfy is thinking...\n", fg="bright_black")
        response = chat.invoke(messages)
        response_content = response.content.strip()

        # --- COMMAND FLOW ---
        execute_match = re.search(r"<execute>(.*?)</execute>", response_content, re.DOTALL)

        if execute_match:
            command_to_run = execute_match.group(1).strip()
            click.secho("🤖 Bitfy wants to run:", fg="yellow")
            click.secho(f"   {command_to_run}\n", fg="white", bold=True)

            if click.confirm("Execute this command?", default=True):
                output = run_command(command_to_run)
                click.secho("\n" + output, fg="cyan")
                final_response = output
            else:
                click.secho("Skipped.", fg="red")
                final_response = "User skipped command execution."
        else:
            # --- CHIT-CHAT FLOW ---
            click.secho("🤖 Bitfy:", fg="green", bold=True)
            click.echo(response_content)
            final_response = response_content

        # --- UPDATE MEMORY ---
        short_term.append({"prompt": ask, "response": final_response})

        # Archive oldest entries to long-term if short-term is full
        short_term = summarize_and_archive(short_term, chat)

        save_short_term(short_term)

    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red")


if __name__ == "__main__":
    main()