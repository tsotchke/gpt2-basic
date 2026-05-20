import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AssistantShowcaseVideoTests(unittest.TestCase):
    def test_terminal_showcase_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "play_assistant_showcase_terminal.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_showcase_terminal_self_test=1", result.stdout)

    def test_video_builder_self_test(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_assistant_showcase_video.py"), "--self-test"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("PROBE_OK assistant_showcase_video_self_test=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
