import frontmatter
from pathlib import Path
from jinja2 import Template
from .models import PromptMetadata

class PromptManager:
    def __init__(self, path: str = "./prompts"):
        self.base_path = Path(path)

    def get_prompt(self, task_name: str, version: str, variables: dict = None):
        """
        Loads a prompt file, validates metadata, and injects variables.
        """
        file_path = self.base_path / task_name / f"{version}.md"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No prompt found at {file_path}")

        # Parse YAML frontmatter and content
        post = frontmatter.load(file_path)
        
        # Validate metadata using our Pydantic model
        # post.metadata contains the YAML dict
        metadata = PromptMetadata(**post.metadata)
        
        # Use Jinja2 to inject variables into the prompt body
        template = Template(post.content)
        rendered_content = template.render(variables or {})

        return {
            "metadata": metadata,
            "content": rendered_content
        }