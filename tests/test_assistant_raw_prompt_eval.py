from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AssistantRawPromptEvalTests(unittest.TestCase):
    def test_raw_prompt_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_raw_prompts.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_raw_prompt_eval_self_test=1", result.stdout)

    def test_chat_training_corpus_avoids_persona_leak(self) -> None:
        script = (ROOT / "scripts" / "train_assistant_pack_models.py").read_text(encoding="ascii")

        self.assertIn('return f"User: {query} Assistant:"', script)
        self.assertIn('assert "Use two brief sentences" not in chat_corpus', script)
        self.assertIn('assert "small friendly DOS chat assistant" not in chat_corpus', script)
        self.assertIn("instruction_leak", script)


if __name__ == "__main__":
    unittest.main()
