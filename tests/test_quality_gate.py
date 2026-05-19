#!/usr/bin/env python3
"""Validators for the stricter GPT2-BASIC text quality gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_gpt2_basic_quality import (  # noqa: E402
    QualityPrompt,
    boundary_error_count,
    score_completion,
)


class QualityGateTest(unittest.TestCase):
    def test_quality_gate_rejects_malformed_fragments(self) -> None:
        text = "The runtime is useful, but predas lowtsed stststs should not pass."
        self.assertGreater(boundary_error_count(text), 0)

    def test_quality_gate_rejects_high_phrase_repetition(self) -> None:
        prompt = QualityPrompt("probe", "To improve performance on real hardware", ("runtime", "memory"))
        completion = (
            "Use runtime memory. Use runtime memory. Use runtime memory. "
            "Use runtime memory. Use runtime memory."
        )
        result = score_completion(prompt, completion, len(completion.split()), 0.1)
        self.assertGreater(result.repeated_trigram_ratio, 0.18)
        self.assertFalse(result.passed)

    def test_quality_gate_accepts_clean_technical_completion(self) -> None:
        prompt = QualityPrompt("probe", "What makes this real inference?", ("logits", "weights"))
        completion = (
            "The DOS program loads trained weights, runs transformer layers, "
            "computes logits, and samples the next token from the checkpoint."
        )
        result = score_completion(prompt, completion, 32, 0.72)
        self.assertEqual(result.boundary_errors, 0)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
