from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class EasysearchConfig(BaseModel):
    collection_name: str = Field("mem0", description="Name of the index")
    host: str = Field("localhost", description="Easysearch host")
    port: int = Field(9200, description="Easysearch port")
    user: Optional[str] = Field(None, description="Username for authentication")
    password: Optional[str] = Field(None, description="Password for authentication")
    embedding_model_dims: int = Field(1536, description="Dimension of the embedding vector")
    verify_certs: bool = Field(False, description="Verify SSL certificates")
    use_ssl: bool = Field(False, description="Use SSL for connection")

    @model_validator(mode="before")
    @classmethod
    def validate_auth(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("host"):
            raise ValueError("Host must be provided for Easysearch")
        return values
