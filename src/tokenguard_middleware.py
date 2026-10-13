"""
TokenGuard Middleware Protocol Implementation
Interception layer enforcing:
1. Dynamic System Prompt Truncation (DSP-T)
2. Sliding Window Context Management (SW-CM)
3. Context Exclusion Filtering (CE-RF via Regex)
4. Model Tier Routing Logic (MB-TR)
5. Agentic Loop Circuit Breakers (AL-CB)
"""

import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple


class CircuitBreakerTrip(RuntimeError):
    """Raised when an autonomous execution safeguard or circuit breaker is tripped."""
    pass


class TokenGuardMiddleware:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.total_tokens_consumed = 0
        self.consecutive_autonomous_turns = 0
        self.action_history: List[str] = []
        self.error_signature_history: List[str] = []

    # --------------------------------------------------------------------------
    # 1. CONTEXT EXCLUSION FILTERING (CE-RF)
    # --------------------------------------------------------------------------
    def sanitize_payload(self, text: str, is_code: bool = False) -> str:
        """
        Strips ANSI codes, repetitive tabulate grid borders, excessive whitespace,
        and compresses massive terminal outputs.
        """
        if not text:
            return ""

        # 1. Strip ANSI escape sequences & terminal progress controls
        ansi_regex = r'(?:\x1B[@-Z\\-_]|[\x80-\x9A\x9C-\x9F]|(?:\x1B\[|\x9B)[0-?]*[ -/]*[@-~])'
        text = re.sub(ansi_regex, '', text)

        # 2. Strip tabulate ASCII borders (+----+----+ or |====|====|)
        if self.config.get("sanitization", {}).get("strip_tabulate_borders", True):
            border_regex = r'^[\s\t]*[+\-|=]{3,}[\s\t]*$'
            text = re.sub(border_regex, '', text, flags=re.MULTILINE)

        # 3. Strip repetitive comment banners if code
        if is_code and self.config.get("sanitization", {}).get("strip_code_comment_headers", True):
            text = re.sub(r'^[ \t]*##[^\n]*\n', '', text, flags=re.MULTILINE)

        # 4. Collapse 3 or more empty lines into double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 5. Output Truncation Window (if logs exceed threshold)
        max_lines = self.config.get("sanitization", {}).get("truncate_log_lines_threshold", 60)
        head_lines = self.config.get("sanitization", {}).get("max_preserved_head_lines", 30)
        tail_lines = self.config.get("sanitization", {}).get("max_preserved_tail_lines", 10)

        lines = text.splitlines()
        if len(lines) > max_lines:
            omitted = len(lines) - (head_lines + tail_lines)
            head_part = "\n".join(lines[:head_lines])
            tail_part = "\n".join(lines[-tail_lines:])
            text = (
                f"{head_part}\n"
                f"... [TokenGuard CE-RF: {omitted} lines pruned to protect context budget] ...\n"
                f"{tail_part}"
            )

        return text.strip()

    # --------------------------------------------------------------------------
    # 2. DYNAMIC SYSTEM PROMPT TRUNCATION (DSP-T)
    # --------------------------------------------------------------------------
    def assemble_system_prompt(self, task_type: str) -> str:
        """
        Dynamically stitches a lean System Prompt kernel with task-specific modules,
        avoiding a monolithic prompt dump.
        """
        base_kernel = self.config.get("prompts", {}).get("base_kernel", "")
        task_module = self.config.get("prompts", {}).get("task_modules", {}).get(task_type, "")
        if task_module:
            return f"{base_kernel}\n\n[TASK CONTEXT - {task_type.upper()}]\n{task_module}".strip()
        return base_kernel.strip()

    # --------------------------------------------------------------------------
    # 3. MODEL TIER ROUTING DECISION TREE (MB-TR)
    # --------------------------------------------------------------------------
    def route_model(self, task_type: str, estimated_tokens: int) -> Tuple[str, str]:
        """
        Decision-tree algorithm routing tasks across Micro, Mid, and Frontier models.
        Returns: (model_name, tier_name)
        """
        models = self.config.get("models", {})
        micro_tier = models.get("micro_tier", "gemini-2.5-flash-lite")
        mid_tier = models.get("mid_tier", "gemini-2.5-flash")
        frontier_tier = models.get("frontier_tier", "gemini-2.5-pro")

        micro_tasks = {"format_cleanup", "syntax_check", "log_analysis", "table_pruning"}
        mid_tasks = {"data_cleansing", "unit_test_gen", "bugfix_targeted", "metadata_extraction"}
        frontier_tasks = {"architectural_refactor", "pipeline_redesign", "complex_synthesis"}

        if task_type in micro_tasks:
            return micro_tier, "micro_tier"
        elif task_type in frontier_tasks or estimated_tokens > 16000:
            return frontier_tier, "frontier_tier"
        elif task_type in mid_tasks:
            return mid_tier, "mid_tier"
        elif estimated_tokens < 1500:
            return micro_tier, "micro_tier"
        else:
            return mid_tier, "mid_tier"

    # --------------------------------------------------------------------------
    # 4. AGENTIC LOOP CIRCUIT BREAKERS (AL-CB)
    # --------------------------------------------------------------------------
    def verify_circuit_breakers(
        self,
        command_or_action: Optional[str] = None,
        last_error_message: Optional[str] = None
    ):
        cb_conf = self.config.get("circuit_breakers", {})
        max_turns = cb_conf.get("max_autonomous_turns", 6)
        max_budget = cb_conf.get("max_token_budget_per_task", 120000)
        max_identical = cb_conf.get("max_identical_actions", 3)

        self.consecutive_autonomous_turns += 1

        # 1. Turn count trip
        if self.consecutive_autonomous_turns > max_turns:
            raise CircuitBreakerTrip(
                f"[CIRCUIT BREAKER TRIPPED: LOOP_OVERFLOW] Agent exceeded {max_turns} "
                f"autonomous turns without human intervention. Halting to prevent burn."
            )

        # 2. Token budget trip
        if self.total_tokens_consumed > max_budget:
            raise CircuitBreakerTrip(
                f"[CIRCUIT BREAKER TRIPPED: BUDGET_EXCEEDED] Cumulative token burn "
                f"({self.total_tokens_consumed}) exceeded quota of {max_budget} tokens."
            )

        # 3. Action / Tool Ping-Pong Check
        if command_or_action:
            action_hash = hashlib.sha256(command_or_action.strip().encode()).hexdigest()
            self.action_history.append(action_hash)
            if self.action_history.count(action_hash) >= max_identical:
                raise CircuitBreakerTrip(
                    f"[CIRCUIT BREAKER TRIPPED: PING_PONG_ACTION] Detected {max_identical} "
                    f"identical repeated actions without state progression: {command_or_action[:80]}"
                )

        # 4. Error Oscillation Check
        if last_error_message:
            err_hash = hashlib.md5(last_error_message.strip().encode()).hexdigest()
            self.error_signature_history.append(err_hash)
            if len(self.error_signature_history) >= 4:
                if (
                    self.error_signature_history[-1] == self.error_signature_history[-3]
                    and self.error_signature_history[-2] == self.error_signature_history[-4]
                    and self.error_signature_history[-1] != self.error_signature_history[-2]
                ):
                    raise CircuitBreakerTrip(
                        "[CIRCUIT BREAKER TRIPPED: ERROR_OSCILLATION] Oscillating fix/break "
                        "loop detected between two error signatures. Disengaging autonomous agent."
                    )

    def reset_human_turn_counter(self):
        self.consecutive_autonomous_turns = 0

    # --------------------------------------------------------------------------
    # 5. SLIDING WINDOW & CONTEXT PRUNING PIPELINE (SW-CM)
    # --------------------------------------------------------------------------
    def process_request_envelope(
        self,
        raw_messages: List[Dict[str, str]],
        task_type: str = "data_cleansing",
        action_intent: Optional[str] = None,
        last_error: Optional[str] = None
    ) -> Dict[str, Any]:
        # Validate Circuit Breakers
        self.verify_circuit_breakers(command_or_action=action_intent, last_error_message=last_error)

        # Build Lean System Prompt (DSP-T)
        sys_prompt = self.assemble_system_prompt(task_type)
        dispatched_messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]

        # Extract User/Assistant History
        non_system = [m for m in raw_messages if m.get("role") != "system"]
        max_recent = self.config.get("context", {}).get("max_recent_turns", 6)

        # Preserve Anchor Turn (Initial Task Definition)
        if non_system:
            anchor_content = self.sanitize_payload(non_system[0]["content"], is_code=False)
            dispatched_messages.append({"role": non_system[0]["role"], "content": anchor_content})

        # Preserve Recent History Slice (Sliding Window Tail)
        recent_tail = non_system[1:][-max_recent:] if len(non_system) > 1 else []
        for turn in recent_tail:
            is_code_turn = turn.get("role") == "assistant"
            cleaned_text = self.sanitize_payload(turn.get("content", ""), is_code=is_code_turn)
            dispatched_messages.append({"role": turn["role"], "content": cleaned_text})

        # Token Estimation & Quota Accounting
        total_chars = sum(len(m["content"]) for m in dispatched_messages)
        estimated_input_tokens = total_chars // 4
        self.total_tokens_consumed += estimated_input_tokens

        # Model Tier Routing
        routed_model, tier_name = self.route_model(task_type, estimated_input_tokens)

        return {
            "status": "APPROVED",
            "model": routed_model,
            "tier": tier_name,
            "task_type": task_type,
            "messages": dispatched_messages,
            "estimated_tokens": estimated_input_tokens,
            "cumulative_tokens": self.total_tokens_consumed,
            "autonomous_turn": self.consecutive_autonomous_turns
        }
