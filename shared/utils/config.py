from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DEFAULT_MODEL     = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "4096"))
