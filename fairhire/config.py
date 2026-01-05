from fairhire.helpers import getenv

REDIS_URL = getenv("REDIS_URL", "redis://localhost:6379")
OLLAMA_URL = getenv("OLLAMA_URL", "http://localhost:11434")
DEBUG = getenv("DEBUG", 0)

DEFAULT_MODEL = getenv("FAIRHIRE_MODEL", "llama3.2")
PROTECTED_ATTRS = ["gender", "race", "age"]
