import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('sync', Path(__file__).parents[2] / 'scripts/import_overleaf.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class ImportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.remote = Path(self.tmp.name) / 'remote'
        self.paper = Path(self.tmp.name) / 'paper'
        self.remote.mkdir(); self.paper.mkdir()
        for root in [self.remote, self.paper]:
            (root / 'main.tex').write_bytes(b'original')
            (root / 'references.bib').write_bytes(b'references')
        (self.remote / 'latexmkrc').write_bytes(b'config')
        self.baseline()

    def baseline(self):
        self.state = {'files': {p.name: sync.digest(p.read_bytes()) for p in self.remote.iterdir() if p.name != '.github-sync.json'}}
        (self.remote / '.github-sync.json').write_text(json.dumps(self.state))

    def run_import(self):
        return sync.import_edits(self.remote, self.paper)

    def test_no_change_and_github_only(self):
        self.assertEqual(self.run_import(), 0)
        (self.paper / 'main.tex').write_bytes(b'GitHub edit')
        self.assertEqual(self.run_import(), 0)
        self.assertEqual((self.paper / 'main.tex').read_bytes(), b'GitHub edit')

    def test_import_add_delete_and_idempotence(self):
        (self.remote / 'main.tex').write_bytes(b'Overleaf edit')
        (self.remote / 'new.tex').write_bytes(b'new')
        (self.remote / 'references.bib').unlink()
        self.assertEqual(self.run_import(), 3)
        self.assertEqual((self.paper / 'main.tex').read_bytes(), b'Overleaf edit')
        self.assertTrue((self.paper / 'new.tex').exists())
        self.assertFalse((self.paper / 'references.bib').exists())
        self.assertEqual(self.run_import(), 0)

    def test_conflict_is_atomic(self):
        (self.remote / 'aaa.tex').write_bytes(b'new')
        (self.remote / 'main.tex').write_bytes(b'Overleaf edit')
        (self.paper / 'main.tex').write_bytes(b'GitHub edit')
        before = (self.remote / '.github-sync.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Both sides'):
            self.run_import()
        self.assertFalse((self.paper / 'aaa.tex').exists())
        self.assertEqual((self.remote / '.github-sync.json').read_bytes(), before)

    def test_retry_after_github_push(self):
        for root in [self.remote, self.paper]:
            (root / 'main.tex').write_bytes(b'already imported')
        self.assertEqual(self.run_import(), 1)
        self.assertEqual(self.run_import(), 0)

    def test_paths_roundtrip_and_config_guard(self):
        (self.paper / 'main.tex').write_bytes(b'\\input{../tables/a.tex}')
        (self.remote / 'main.tex').write_bytes(b'\\input{tables/a.tex}')
        self.baseline()
        self.assertEqual(self.run_import(), 0)
        (self.remote / 'main.tex').write_bytes(b'edit \\input{tables/a.tex}')
        self.run_import()
        self.assertEqual((self.paper / 'main.tex').read_bytes(), b'edit \\input{../tables/a.tex}')
        (self.remote / 'latexmkrc').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.run_import()

if __name__ == '__main__':
    unittest.main()
