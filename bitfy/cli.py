import click
import requests
import json
import os
from pathlib import Path
from collections import deque
import sys
import io
import google.generativeai as genai
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class BitfyTool:
    def __init__(self):
        self.api_key = None
        self.memory_file = Path.home() / ".bitfy_memory.txt"
        self.memory_size = 50  # Keep last 50 interactions
        self._initialize_api_key()
        self._initialize_memory()

    def _initialize_api_key(self):
        """Initialize API key from env or config file"""
        self.api_key = os.getenv("BITFY_API_KEY") or self._load_api_key_from_config()
        if not self.api_key:
            self._setup_api_key()

    def _setup_api_key(self):
        """Prompt user for API key and store it"""
        click.echo("API key not found. Let's set it up once.")
        api_key = click.prompt("Enter your Gemini API key: ").strip()
        if not api_key:
            raise ValueError("API key cannot be empty")
        
        # Store in environment variables (for current session)
        os.environ["BITFY_API_KEY"] = api_key
        
        # Store in config file for persistence
        config_path = Path.home() / ".bitfy_config"
        with open(config_path, "w") as f:
            json.dump({"api_key": api_key}, f)
        
        self.api_key = api_key
        click.echo("API key stored successfully!")

    def _load_api_key_from_config(self):
        """Load API key from config file"""
        config_path = Path.home() / ".bitfy_config"
        try:
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    return config.get("api_key")
        except json.JSONDecodeError:
            click.echo("⚠️ Config file corrupted. Please enter API key again.")
            os.remove(config_path)  # Remove corrupted config
        return None

    def _initialize_memory(self):
        """Initialize or load memory file"""
        if not self.memory_file.exists():
            self.memory_file.touch()
        self.memory = self._load_memory()

    def _load_memory(self):
        """Load last N lines from memory file"""
        try:
            with open(self.memory_file, 'r') as f:
                return deque(f.readlines()[-self.memory_size:], maxlen=self.memory_size)
        except:
            return deque(maxlen=self.memory_size)

    def _update_memory(self, prompt, response):
        """Store interaction in memory"""
        entry = f"User: {prompt}\nAI: {response}\n\n"
        self.memory.append(entry)
        with open(self.memory_file, 'a') as f:
            f.write(entry)

    def _get_memory_context(self):
        """Get recent memory as context"""
        return "Previous interactions:\n" + "".join(self.memory) if self.memory else ""

    def ask_gemini(self, prompt):
        if not self.api_key:
            raise ValueError("API key not set")
    
       
    
        # Proceed with Gemini for other queries
        full_prompt = f"{self._get_memory_context()}\nNew query: {prompt}"
    
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
            "parts": [{"text": full_prompt}]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            ai_response = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            self._update_memory(prompt, ai_response)
            return ai_response
        
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    
    def giving_help(self):
        click.echo("Here's some help:")
        click.echo("--bitfy: Show Bitfy intro")
        click.echo("--ask 'question': Ask Gemini a question")
        click.echo("--help: Show help")
        click.echo("--explain 'filepath' 'prompt': Explain a file")
        click.echo("--write 'directory' 'filename' 'prompt': Write to a file")
        click.echo("--isthere 'filename': Check if a file exists")
        click.echo("--replace 'directory' 'old_word' 'new_word': Replace a word in all files in a directory")

    
    def explain_this(self, filepath=None, start = 0, end = None, prompt="explain this"):
        if not filepath:
            click.echo("Please provide a file path")
            return

        if (end is None):
            with open(filepath, "r") as f:
                lines = f.readlines()
                end = len(lines) - 1

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = "".join([f.read() for i, l in enumerate(f) if start <= i <= end])
            explanation = self.ask_gemini(f"""{prompt}: This was written in file - {content}""")
            
            click.echo(explanation)
        except Exception as e:
            click.echo(f"⚠️ Error: {str(e)}")

    def check_file_exists(self, file):
        """
        Check if a specific file exists in the current directory.

        :param file: The file name to check for.
        :return: True if the file exists, False otherwise.
    """
        # Get the current working directory
        current_directory = os.getcwd()
    
        # Construct the full file path
        file_path = os.path.join(current_directory, file)
    
        # Check if the file exists at the given path
        return os.path.isfile(file_path)

    def replace_word(self, directory, old_word, new_word):
        if (old_word is None or new_word is None):
            click.echo("Please provide both old and new words")
            return

        with open(directory, "r") as old, open(directory, "w") as new:

            for line in old:
                new.write(line.replace(old_word, new_word))

    def write(self, directory=None, filename=None, prompt="write code about calculator"):
        if not directory or not filename:
            click.echo("Please provide both directory and filename")
            return
        
        try:
            # Use pathlib for safe path joining
            filepath = Path(directory) / filename
            content = self.ask_gemini(prompt)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            click.echo(f"File written successfully to {filepath}")
        except Exception as e:
            click.echo(f"Error writing file: {str(e)}")

@click.command()
@click.option("--bitfy", is_flag=True, help="Show Bitfy intro")
@click.option("--ask", help="Ask Gemini a question")
@click.option("--help", is_flag=True, help="Show help")
@click.option("--explain", nargs=2, type=str, help="Explain a file (provide filepath and prompt)")
@click.option("--write", nargs=3, type=str, help="Write to a file (provide directory, filename and prompt)")
@click.option("--isthere", nargs=1, type=str, help="Check if a file exists (provide filename)")
@click.option("--replace", nargs=3, type=str, help="Replace a word in a file (provide directory, old word and new word)")
def main(bitfy, ask, help, explain, write, isthere, replace):
    """Bitfy CLI - AI Assistant"""
    tool = BitfyTool()
    
    if bitfy:
        pass  # Already shown in init
    elif ask:
        response = tool.ask_gemini(ask)
        click.echo(f"🦋 Says:\n{response}")
    elif help:
        tool.giving_help()  
    elif explain:
        filepath, prompt = explain
        tool.explain_this(filepath, prompt)
    elif write:
        directory, filename, prompt = write
        tool.write(directory, filename, prompt)
    elif isthere:
        filename = isthere
        click.echo(tool.check_file_exists(filename))
    elif replace:
        directory, old_word, new_word = replace
        tool.replace_word(directory, old_word, new_word)
    else:
        tool.giving_help()

if __name__ == "__main__":
    main()