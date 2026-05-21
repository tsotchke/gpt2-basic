from __future__ import annotations

import contextlib
import io
import unittest

from scripts import benchmark_assistant_recall


class AssistantRecallBenchmarkTests(unittest.TestCase):
    def test_self_test(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            benchmark_assistant_recall.self_test()

        self.assertIn("PROBE_OK assistant_recall_benchmark_self_test=1", output.getvalue())

    def test_rejects_slow_average_recall(self) -> None:
        records = [
            benchmark_assistant_recall.RecallRecord(
                case.pack,
                case.query,
                "kb2_term",
                99,
                10,
                " ".join(case.terms) + ".",
            )
            for case in benchmark_assistant_recall.CASES
        ]

        with self.assertRaises(SystemExit) as raised:
            benchmark_assistant_recall.validate_records(records, max_average_ms=1, max_single_ms=1500)

        self.assertIn("average_recall_ms", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
