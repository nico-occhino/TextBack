"""Print a small environment report for TextBack."""

import os
import sys


def main() -> None:
    """Check the main runtime dependencies without running an experiment."""
    dotenv_status = "ok"
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        dotenv_status = "not installed"

    print(f"Python version: {sys.version.split()[0]}")

    try:
        import torch

        print(f"torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Selected GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Selected GPU: none")
    except ImportError:
        print("torch version: not installed")
        print("CUDA available: unknown")
        print("Selected GPU: unknown")

    has_gemini_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_groq_key = bool(os.getenv("GROQ_API_KEY"))
    print(f"python-dotenv import: {dotenv_status}")
    print("Recommended TextGrad backend: Groq")
    print(f"Groq API key present: {has_groq_key}")
    print(f"Gemini API key present: {has_gemini_key}")

    try:
        import textgrad  # noqa: F401

        print("textgrad import: ok")
    except ImportError:
        print("textgrad import: failed")

    try:
        import diffusers  # noqa: F401

        print("diffusers import: ok")
    except ImportError:
        print("diffusers import: failed")


if __name__ == "__main__":
    main()
