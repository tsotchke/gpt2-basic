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

    def test_assistant_consistency_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_consistency.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_consistency_self_test=1", result.stdout)

    def test_assistant_generalist_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_generalist_prompts.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_generalist_prompt_eval_self_test=1", result.stdout)

    def test_assistant_pack_retrieval_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_pack_retrieval.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_pack_retrieval_eval_self_test=1", result.stdout)

    def test_assistant_usefulness_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_usefulness.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_usefulness_eval_self_test=1", result.stdout)

    def test_assistant_kdb_index_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_kdb_index.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_kdb_index_eval_self_test=1", result.stdout)

    def test_assistant_kdb_binary_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_kdb_binary.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_kdb_binary_eval_self_test=1", result.stdout)

    def test_assistant_kdb_term_index_eval_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_assistant_kdb_term_index.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_kdb_term_index_eval_self_test=1", result.stdout)

    def test_assistant_kdb_builder_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_assistant_kdb.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_kdb_self_test=1", result.stdout)

    def test_assistant_pack_authoring_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_assistant_pack_authoring.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_pack_authoring_self_test=1", result.stdout)

    def test_assistant_note_import_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "import_assistant_notes.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_note_import_self_test=1", result.stdout)

    def test_assistant_pack_create_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "create_assistant_pack.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_pack_create_self_test=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
