import click
import requests
import json
import os
from pathlib import Path
from collections import deque
import sys
import io
import google.generativeai as genai
from dotenv import load_dotenv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import subprocess

class BitfyTool:
    def __init__(self):
        self.api_key = None
        self.memory_file = Path.home() / ".bitfy_memory.txt"
        self.memory_size = 50  # Keep last 50 interactions
        self._initialize_api_key()
        self._initialize_memory()
        load_dotenv()
        self.configured = False
        self.safety_mode = True  # Fixed typo
        self._configure_api()
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
    def _initialize_api_key(self):
        """Initialize API key from env or config file"""
        self.api_key = os.getenv("BITFY_API_KEY") or self._load_api_key_from_config()
        if not self.api_key:
            self._setup_api_key()
    def change_api(self, new_api_key: str):
        """Change the API key and reconfigure the API"""
        if not new_api_key:
            raise ValueError("API key cannot be empty")
        
        # Update the API key
        self.api_key = new_api_key
        
        # Store in environment variables (for current session)
        os.environ["BITFY_API_KEY"] = new_api_key
        
        # Store in config file for persistence
        config_path = Path.home() / ".bitfy_config"
        with open(config_path, "w") as f:
            json.dump({"api_key": new_api_key}, f)
        
        # Reconfigure the API
        self._configure_api()
        
        if self.configured:
            click.echo("API key updated and API reconfigured successfully!")
        else:
            click.echo("Failed to reconfigure API with the new key.")

    def _configure_api(self):
        try:
            api_key = self.api_key
            if not api_key:
                raise ValueError("Missing API Key")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
            self.configured = True
        except Exception as e:
            click.echo(f"Error configuring API: {str(e)}")
            self.configured = False

    def execute_safely(self, command: str, dry_run: bool = False):
        """Execute command with enhanced validation"""
        command = self.natural_to_shell(command)
    
       
        
        click.echo(f"🔧 Generated Command: {command}")
        
        if dry_run:
            click.echo("🛑 Dry Run: Command not executed")
            return
        
        if self.safety_mode:
            confirm = click.prompt("🚨 Execute this command? (y/N)", default='n')
            if confirm.lower() != "y":
                click.echo("❌ Command cancelled")
                return

        try:
            # Use subprocess for better error handling
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            click.echo(f"✅ Success:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Execution Failed:\n{e.stderr}")


    
    def natural_to_shell(self, user_input: str) -> str:
        """Convert natural language to valid shell command"""
        if not self.configured:
            return "Error: API not configured"

        # Detect OS once
        is_windows = os.name == 'nt'
    
        # Improved prompt with OS-specific examples
        prompt = f"""You are a terminal expert. Convert this to a valid shell command for {'Windows' if is_windows else 'Unix'} systems.
        Current directory: {os.getcwd()}
        
        Examples:
        User: check pandas version → Command: {"pip show pandas" if is_windows else "pip3 show pandas"}
        User: list text files → Command: {"dir *.txt" if is_windows else "ls *.txt"}
        User: see directory contents → Command: {"dir" if is_windows else "ls"}
        User: what's my Python version → Command: {"python --version" if is_windows else "python3 --version"}
        
        User request: {user_input}
        Reply ONLY with the command, nothing else.
        Command:"""
    
        try:
            response = self.ask_gemini(prompt)
            command = response.strip()

            return command
        except Exception as e:
            return f"AI Error: {str(e)}"
    
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
    def _text_to_command(self, prompt):
        final_prompt = f"""
        You are text to shell command converter your output directly goes into shell command so convert 
        {prompt} to shell command so that we can get output
        dont add any extra text 
        For an eg:
        "Show me Pandas" : pip show pandas
        "is numpy exists their" : pip show numpy
        "download streamlit" pip install streamlit
        And user is on windows so no grep etc
        just one line command
        no decorators like ``` "" etc
        """

        try:
            response = self.model.generate_content(final_prompt)
            ai_response = response.text.strip()
            self._update_memory(prompt, ai_response)
            return ai_response
        
        except Exception as e:
            click.echo(f"There is some error u can try again {str(e)}")

    def control_shell(self, prompt):
        response = self._text_to_command(prompt)
        try:
            subprocess.run(response, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Command failed: {e}")

    def ask_gemini(self, prompt):
        if not self.api_key:
            raise ValueError("API key not set")
    
        full_prompt = f"{self._get_memory_context()}\nNew query: {prompt}"
    
        try:
            response = self.model.generate_content(full_prompt)
            ai_response = response.text.strip()
            self._update_memory(prompt, ai_response)
            return ai_response
        
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

    
    def giving_help(self):
        click.echo("Here's some help:")
        click.echo("--ask -a 'question': Ask Gemini a question")
        click.echo("--explain -e 'filepath' 'prompt': Explain a file")
        click.echo("--write -w 'directory' 'filename' 'prompt': Write to a file")
        click.echo("--shell -s 'command': Convert natural language to shell command")
        click.echo("--change -ca 'new_api_key': Change API key")
    
    def explain_this(self, filepath=None, prompt="explain this"):
        if not filepath:
            click.echo("Please provide a file path")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            explanation = self.ask_gemini(f"{prompt}: {content}")
            click.echo(explanation)
        except Exception as e:
            click.echo(f"⚠️ Error: {str(e)}")



    def write(self, directory=None, filename=None, prompt="write code about calculator"):
        if not directory or not filename:
            click.echo("Please provide both directory and filename")
            return
        
        try:
            # Use pathlib for safe path joining
            filepath = Path(directory) / filename
            content = self.ask_gemini(f"""{prompt}: This was written in file - {filepath}""")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            click.echo(f"File written successfully to {filepath}")
        except Exception as e:
            click.echo(f"Error writing file: {str(e)}")

@click.command()
@click.option("--ask", "-a", help="Ask Gemini a question")
@click.option("--explain", "-e", nargs=2, type=str, help="Explain a file (provide filepath and prompt)")
@click.option("--write", "-w", nargs=3, type=str, help="Write to a file (provide directory, filename and prompt)")
@click.option("--shell", "-s", help = "Taking input for to convert it into shell")
@click.option("--change", "-ca", help = "Change API key")
def main(ask, explain, write, shell, change):
    """Bitfy CLI - AI Assistant"""
    tool = BitfyTool()
    
    if ask:
        response = tool.ask_gemini(ask)
        click.echo(f"🦋 Says:\n{response}")
    elif explain:
        filepath, prompt = explain
        tool.explain_this(filepath, prompt)
    elif write:
        directory, filename, prompt = write
        tool.write(directory, filename, prompt)
    elif shell:
        tool.control_shell(shell)
    elif change:
        tool.change_api(change)
    else:
        tool.giving_help()

if __name__ == "__main__":
    main()