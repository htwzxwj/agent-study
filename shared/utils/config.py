from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    import sys
    print("WARNING: ANTHROPIC_API_KEY not set. Add it to .env file.", file=sys.stderr)

ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")
DEFAULT_MODEL      = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_TOKENS         = int(os.getenv("MAX_TOKENS", "4096"))
