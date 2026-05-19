#!/usr/bin/env python3
"""Focused validators for host-only subword prototype artifacts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_subword_prototype import (  # noqa: E402
    SubwordTokenizer,
    mark_host_only_export,
    prototype_export_tokenizer,
)


class TrainSubwordPrototypeTest(unittest.TestCase):
    def test_train_subword_prototype_validates_subwordvocab_and_host_marker(self) -> None:
        vocab_artifact = "SUBWORDVOCAB.json"
        host_marker = "HOST_ONLY_NOT_DOS_READY.txt"
        tokenizer = SubwordTokenizer.build("fixed point transformer memory runtime", 300)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            tokenizer.write(output / vocab_artifact)
            loaded = SubwordTokenizer.load(output / vocab_artifact)
            export_tokenizer = prototype_export_tokenizer(loaded)
            export_tokenizer.validate_for_vocab_size(loaded.vocab_size)
            export_tokenizer.write_vocab_bin(output / "VOCAB.BIN")

            (output / "PROFILE.TXT").write_text("tokenizer=byte\n", encoding="ascii")
            mark_host_only_export(output)

            profile = (output / "PROFILE.TXT").read_text(encoding="ascii")
            marker = (output / host_marker).read_text(encoding="ascii")

        self.assertIn("tokenizer=host-only-greedy-subword", profile)
        self.assertIn("dos_ready=0", profile)
        self.assertIn(f"host_tokenizer={vocab_artifact}", profile)
        self.assertIn("host evaluation only", marker)


if __name__ == "__main__":
    unittest.main()
