import os
from pathlib import Path

class PromptManager:
    def __init__(self, prompts_dir="prompts"):
        self.prompts_dir = prompts_dir
        self.prompts = {}
        self.load_prompts()
    
    def load_prompts(self):
        """Load all prompt files from the prompts directory"""
        prompts_path = Path(self.prompts_dir)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        for prompt_file in prompts_path.glob("*.txt"):
            prompt_name = prompt_file.stem
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.prompts[prompt_name] = f.read().strip()
    
    def get_prompt(self, prompt_name):
        """Get a specific prompt by name"""
        if prompt_name not in self.prompts:
            raise KeyError(f"Prompt not found: {prompt_name}")
        return self.prompts[prompt_name]
    
    def get_system_prompt(self):
        """Get the system prompt"""
        return self.get_prompt("system")
    
    def refresh_prompts(self):
        """Reload all prompts from disk"""
        self.prompts.clear()
        self.load_prompts() 