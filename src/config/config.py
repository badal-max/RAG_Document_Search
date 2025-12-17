"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG system"""

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.1-8b-instant"
    
    
    

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the Groq LLM model"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in the environment.")

        os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        return ChatGroq(
            model=cls.LLM_MODEL,
            temperature=0.2,   # good for RAG
        )
