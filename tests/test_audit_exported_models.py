#!/usr/bin/env python3
"""Focused validators for exported-model inventory artifacts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_exported_models import (  # noqa: E402
    DEFAULT_OUTPUT,
    AuditRow,
    ExportedModel,
    markdown,
)
from era_performance_model import Config  # noqa: E402


class AuditExportedModelsTest(unittest.TestCase):
    def test_audit_exported_models_validates_exported_model_quality_inventory(self) -> None:
        artifact_name = "exported_model_quality_inventory.md"
        self.assertEqual(DEFAULT_OUTPUT.name, artifact_name)

        row = AuditRow(
            ExportedModel("MODEL_TEST", "root", ROOT / "assets" / "gpt2_basic" / "MODEL_TEST"),
            Config(
                profile="test",
                n_layer=2,
                n_embd=48,
                n_head=4,
                n_positions=192,
                hidden_dim=192,
                vocab_size=4096,
            ),
            True,
            ROOT / "qemu" / "evidence" / "model_inventory_model_test.log",
            (),
        )
        text = markdown([row], ROOT / "qemu" / "evidence")

        self.assertIn("# Exported Model Quality Inventory", text)
        self.assertIn("Models audited: `1`", text)
        self.assertIn("Artifact pass: `1/1`", text)
        self.assertIn("Missing quality evidence: `1`", text)


if __name__ == "__main__":
    unittest.main()
