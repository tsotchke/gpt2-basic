import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cast", ".log", ".md", ".txt"}
PUBLIC_TEXT_ROOTS = (
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "qemu" / "evidence",
    ROOT / "promo" / "renders",
)


def public_text_files() -> list[Path]:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    roots = tuple(root.relative_to(ROOT).as_posix() for root in PUBLIC_TEXT_ROOTS if root.is_dir())
    root_files = {root.relative_to(ROOT).as_posix() for root in PUBLIC_TEXT_ROOTS if root.is_file()}
    files: list[Path] = []
    for name in tracked:
        path = ROOT / name
        if name in root_files or (name.startswith(roots) and path.suffix.lower() in TEXT_SUFFIXES):
            files.append(path)
    return files


class PublicRepoHygieneTests(unittest.TestCase):
    def test_public_text_avoids_local_operator_paths(self) -> None:
        offender_pattern = re.compile(r"/Users/[^/\s]+/Desktop/gpt2-basic")
        offenders: list[str] = []
        for path in public_text_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if offender_pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual([], offenders)

    def test_public_copy_uses_product_not_personal_framing(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        promo = (ROOT / "docs" / "marketing" / "promo-kit.md").read_text(encoding="utf-8")

        self.assertNotIn("**Proof of Concept**", readme)
        self.assertNotIn("Contact: Tsotchke Corporation / project owner", promo)
        self.assertIn("**Practical Local AI**", readme)


if __name__ == "__main__":
    unittest.main()
