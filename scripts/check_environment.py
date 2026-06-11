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

    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_hf_token = bool(os.getenv("HF_TOKEN"))
    print(f"python-dotenv import: {dotenv_status}")
    print("Recommended TextGrad backend: OpenAI")
    print(f"OpenAI API key present: {has_openai_key}")
    print(f"HF token present: {has_hf_token}")

    try:
        import robustbench  # noqa: F401
        from robustbench.utils import load_model  # noqa: F401

        print("robustbench import: ok")
    except Exception as error:
        print(f"robustbench import: failed - final classifier will not run ({error})")

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
