from pydantic import BaseModel, Field
from typing import Dict, Any

class PromptMetadata(BaseModel):
    version: str
    model: str
    temperature: float = 0.7
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)