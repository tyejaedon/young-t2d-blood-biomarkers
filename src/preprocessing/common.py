"""
Common utilities and logging for preprocessing pipeline.
"""

def log_step(stage: str, message: str) -> None:
    """Standardized clean logging."""
    print(f"[{stage}] {message}")
