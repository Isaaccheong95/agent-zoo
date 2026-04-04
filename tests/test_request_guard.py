from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent if (CURRENT_FILE.parent / "src").exists() else CURRENT_FILE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_zoo.request_guard import (
    ALLOW_RESPONSE_TYPE,
    CLARIFICATION_RESPONSE_TYPE,
    OUT_OF_SCOPE_RESPONSE_TYPE,
    build_llm_request_guard,
    normalize_request_guard_decision,
)


class RequestGuardTestCase(unittest.TestCase):
    def test_normalize_request_guard_decision_parses_allow_json(self) -> None:
        decision = normalize_request_guard_decision(
            '{"response_type":"allow"}',
            default_refusal_message="fallback refusal",
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.response_type, ALLOW_RESPONSE_TYPE)

    def test_normalize_request_guard_decision_parses_out_of_scope_json(self) -> None:
        decision = normalize_request_guard_decision(
            '{"response_type":"out_of_scope","user_message":"Please ask about the dataset instead."}',
            default_refusal_message="fallback refusal",
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.response_type, OUT_OF_SCOPE_RESPONSE_TYPE)
        self.assertEqual(decision.user_message, "Please ask about the dataset instead.")

    def test_normalize_request_guard_decision_recovers_freeform_clarification(self) -> None:
        decision = normalize_request_guard_decision(
            """I need clarification on your request. Which metric do you want me to focus on?\n- age\n- fare\n- survival""",
            default_refusal_message="fallback refusal",
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.response_type, CLARIFICATION_RESPONSE_TYPE)
        self.assertEqual(decision.options, ["age", "fare", "survival"])

    def test_build_llm_request_guard_fails_open_on_guard_error(self) -> None:
        def raise_completion(**kwargs):
            raise RuntimeError("guard unavailable")

        guard = build_llm_request_guard(
            "openai/test-model",
            agent_label="a test agent",
            refusal_message="fallback refusal",
        )

        with patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=raise_completion)}):
            decision = guard("Analyze the result", "Allowed domain text")

        self.assertEqual(decision.response_type, ALLOW_RESPONSE_TYPE)


if __name__ == "__main__":
    unittest.main()