import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cast", ".log", ".md", ".txt"}
PUBLIC_TEXT_ROOTS = (
    ROOT / "README.md",
    ROOT / "gpt2_basic_documentation.md",
    ROOT / "docs",
    ROOT / "qemu" / "evidence",
    ROOT / "promo" / "renders",
)
PUBLIC_COPY_FILES = (
    ROOT / "README.md",
    ROOT / "gpt2_basic_documentation.md",
    ROOT / "docs" / "marketing" / "promo-kit.md",
    ROOT / "docs" / "marketing" / "video-plan.md",
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
        slash = "/"
        offender_pattern = re.compile(
            re.escape(slash.join(("", "Users", ""))) + r"[^/\s]+/Desktop/gpt2-basic"
        )
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
        self.assertIn("Portable Machine Intelligence in BASIC", readme)
        self.assertIn("not a frontier LLM", readme)
        self.assertIn("Physical returned board logs are still pending", readme)
        self.assertIn("Do not claim physical 486 speed", promo)

    def test_public_copy_avoids_novelty_demo_framing(self) -> None:
        banned = (
            "ai meets retrocomputing",
            "digital archaeology",
            "alternate history",
            "counterfactual",
            "where it has no business",
            "real mode vibes",
            "groundbreaking",
            "proof of concept",
        )
        offenders: list[str] = []
        for path in PUBLIC_COPY_FILES:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for phrase in banned:
                if phrase in text:
                    offenders.append(f"{path.relative_to(ROOT)}:{phrase}")

        self.assertEqual([], offenders)

    def test_substrate_portability_claim_is_qualified(self) -> None:
        substrate = (ROOT / "docs" / "substrate-portability.md").read_text(encoding="utf-8")

        self.assertIn("Minimum Substrate", substrate)
        self.assertIn("Pure assembly is a valid implementation path.", substrate)
        self.assertIn("Calculator-class BASIC", substrate)
        self.assertIn("TI-83 Plus-class system", substrate)
        self.assertIn("Do not use this yet:", substrate)
        self.assertIn("Runs on any microprocessor.", substrate)


if __name__ == "__main__":
    unittest.main()
