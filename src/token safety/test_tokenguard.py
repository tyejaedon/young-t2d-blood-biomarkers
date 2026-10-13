"""
Verification test suite for TokenGuard Middleware and Clean Pipeline.
"""

from tokenguard_config import DEFAULT_CONFIG
from tokenguard_middleware import TokenGuardMiddleware, CircuitBreakerTrip

def test_tokenguard_pipeline():
    config = DEFAULT_CONFIG
    guard = TokenGuardMiddleware(config)

    # 1. Test Tabulate / ANSI Sanitization
    sample_raw_stdout = """
    \x1b[32m[DATA-CLEANSE] Pipeline Running\x1b[0m
    +-----------+----------+---------+
    | Dataset   | Status   |   Files |
    |-----------+----------+---------|
    | GSE26168  | READY    |       4 |
    +-----------+----------+---------+
    """
    clean_stdout = guard.sanitize_payload(sample_raw_stdout)
    assert "+-----------+" not in clean_stdout, "Sanitizer failed to strip tabulate border"
    assert "\x1b" not in clean_stdout, "Sanitizer failed to strip ANSI sequences"
    print("[PASS] CE-RF Regex Sanitizer stripped borders and ANSI codes.")

    # 2. Test Dynamic System Prompt Assembly (DSP-T)
    prompt = guard.assemble_system_prompt("data_cleansing")
    assert "GEO transcriptomic metadata" in prompt, "DSP-T failed to inject data_cleansing module"
    print("[PASS] DSP-T assembled lean modular system prompt.")

    # 3. Test Model Tier Routing (MB-TR)
    model, tier = guard.route_model("format_cleanup", 500)
    assert tier == "micro_tier", f"Expected micro_tier, got {tier}"
    
    model_adv, tier_adv = guard.route_model("architectural_refactor", 10000)
    assert tier_adv == "frontier_tier", f"Expected frontier_tier, got {tier_adv}"
    print(f"[PASS] MB-TR correctly routed micro task to {model} and heavy task to {model_adv}.")

    # 4. Test Circuit Breaker: Action Loop Detection (AL-CB)
    try:
        for _ in range(4):
            guard.verify_circuit_breakers(command_or_action="python run_cleansing.py")
        assert False, "Circuit breaker should have tripped on repetitive action"
    except CircuitBreakerTrip as e:
        print(f"[PASS] AL-CB successfully caught repetitive action loop: {e}")

    # 5. Full Envelope Processing
    messages = [
        {"role": "user", "content": "Run dataset validation for GSE26168"},
        {"role": "assistant", "content": "Validated. 4 files found."},
        {"role": "user", "content": "Now check GSE15932"}
    ]
    envelope = guard.process_request_envelope(messages, task_type="data_cleansing")
    assert envelope["status"] == "APPROVED"
    assert envelope["tier"] == "mid_tier"
    print(f"[PASS] Request envelope processed: model={envelope['model']}, estimated_tokens={envelope['estimated_tokens']}")

    print("\nALL TOKEN-GUARD PROTOCOL VERIFICATIONS PASSED.")

if __name__ == "__main__":
    test_tokenguard_pipeline()
