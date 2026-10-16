"""Published F007 report figures render into a separate directory."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.F007.results import plot_final_results as plot


class PublishedFigureTests(unittest.TestCase):
    def test_output_directory_preserves_published_sources(self):
        sources = [plot.REPORT] + [
            plot.HERE / f"{stem}.{extension}"
            for stem in ("heldout_quality_vs_flops", "heterogeneous_coverage")
            for extension in ("png", "svg")
        ]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sources}
        with tempfile.TemporaryDirectory(dir=plot.HERE.parents[2]) as folder:
            destination = Path(folder) / "figures"
            subprocess.run(
                [sys.executable, "-m", "experiments.F007.results.plot_final_results",
                 "--output-dir", str(destination)],
                cwd=plot.HERE.parents[2], check=True, capture_output=True, text=True)
            for source in sources[1:]:
                self.assertTrue((destination / source.name).is_file())
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sources})

    def test_report_hash_mismatch_still_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=plot.HERE.parents[2]) as folder:
            changed = Path(folder) / "report.json"
            changed.write_bytes(plot.REPORT.read_bytes() + b" ")
            with patch.object(plot, "REPORT", changed):
                with self.assertRaisesRegex(ValueError, "differs from the official artifact"):
                    plot.load_report()


if __name__ == "__main__":
    unittest.main()
