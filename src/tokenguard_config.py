"""
TokenGuard Configuration Schema
Production-ready configuration file defining models, token ceilings, circuit breaker thresholds,
and regex sanitization rules for bioinformatics workloads.
"""

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "1.0.0",
    "middleware_mode": "strict_enforcement",
    
    # Model Tier Assignment Matrix
    "models": {
        "micro_tier": "gemini-2.5-flash-lite",      # Syntax checks, table formatting, basic parsing
        "mid_tier": "gemini-2.5-flash",             # Script modifications, unit tests, summaries
        "frontier_tier": "gemini-2.5-pro",          # Multi-file architectural refactoring
        "pricing_per_1m_input": {
            "micro_tier": 0.075,
            "mid_tier": 0.15,
            "frontier_tier": 1.25
        }
    },
    
    # Circuit Breakers (Anti-Burn Safeguards)
    "circuit_breakers": {
        "max_autonomous_turns": 6,                  # Trip after 6 unverified back-and-forths
        "max_token_budget_per_task": 120000,        # Hard stop if session burns >120k tokens
        "max_identical_actions": 3,                 # Detect tool call / command ping-pong
        "error_oscillation_limit": 2                # Detect alternating ABAB error states
    },
    
    # Sliding Window Context Limits
    "context": {
        "max_recent_turns": 6,                      # Preserve last N raw turns
        "compaction_threshold_tokens": 12000,       # Trigger summary compression at this limit
        "hard_context_ceiling_tokens": 24000,       # Absolute context ceiling before hard pruning
        "preserve_anchor": True                     # Always preserve system prompt + initial objective
    },
    
    # Regex Exclusion Filtering Rules
    "sanitization": {
        "strip_ansi": True,
        "strip_tabulate_borders": True,
        "truncate_log_lines_threshold": 60,
        "max_preserved_head_lines": 30,
        "max_preserved_tail_lines": 10,
        "strip_code_comment_headers": True,
        "max_tabular_cell_chars": 80
    },
    
    # Modular System Prompts (DSP-T)
    "prompts": {
        "base_kernel": (
            "You are an expert bioinformatics AI software engineer. Provide concise, "
            "defensive, production-grade solutions. Omit pleasantries, conversational filler, "
            "and speculative prose."
        ),
        "task_modules": {
            "data_cleansing": (
                "Task: Dataset validation and GEO transcriptomic metadata cleansing.\n"
                "Constraints:\n"
                "- Handle missing series matrix files defensively without crashing.\n"
                "- Avoid printing full ASCII tabular dumps in stdout; log concise summary metrics.\n"
                "- Ensure consistent typing and path resolutions across datasets."
            ),
            "code_refactor": (
                "Task: Codebase optimization and refactoring.\n"
                "Constraints:\n"
                "- Optimize memory utilization and token efficiency.\n"
                "- Replace redundant tabulate calls with structured logging or lightweight summaries."
            ),
            "error_analysis": (
                "Task: Diagnose runtime errors.\n"
                "Constraints:\n"
                "- Inspect only root causes and the immediate call frame."
            )
        }
    }
}


def load_config(config_path: str = "tokenguard_config.json") -> Dict[str, Any]:
    """Loads configuration from JSON file or generates default if missing."""
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG


def save_default_config(config_path: str = "tokenguard_config.json"):
    """Saves the default configuration to disk."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)


if __name__ == "__main__":
    save_default_config()
    print(f"Default TokenGuard configuration written to {Path('tokenguard_config.json').resolve()}")
