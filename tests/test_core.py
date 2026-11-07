"""Real-format and mutation/permission regressions; entirely local synthetic data."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import pymupdf

from knowtrace.core import KnowledgeCore
from knowtrace.parsing import MAX_XML_BYTES, Parsed, Unit, chunks, parse


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "managed"
        self.core = KnowledgeCore(self.data)
        self.core.create_kb("Public", kb_id="public")
        self.core.create_kb("Private", kb_id="private")

    def tearDown(self):
        self.temporary.cleanup()

    def ingest(self, text, name="manual.md", kb="public", **kwargs):
        source = self.root / ("input-" + str(len(list(self.root.glob("input-*")))) + Path(name).suffix)
        source.write_text(text, encoding="utf-8")
        return self.core.ingest_file(source, name, kb, **kwargs), source

    def test_acl_none_empty_and_cross_scope_all_accessors(self):
        public, _ = self.ingest("Orchid roots need air.")
        private, _ = self.ingest("Secretorchid account recovery.", "secret.txt", "private")
        self.assertEqual(len(self.core.list_documents(None)), 2)
        self.assertEqual(self.core.list_kbs([]), [])
        self.assertEqual(self.core.list_documents([]), [])
        for scope in ([], ["public"]):
            self.assertIsNone(self.core.get_document(private["id"], scope))
            self.assertIsNone(self.core.source_path(private["id"], scope))
            self.assertEqual(self.core.get_document_chunks(private["id"], scope), [])
            self.assertFalse(self.core.delete_document(private["id"], scope))
            self.assertTrue(all(hit["kb_id"] in scope for hit in self.core.search("Secretorchid", scope)))
        self.assertIsNotNone(self.core.get_document(public["id"], ["public"]))
        self.assertTrue(self.core.ask("orchid", [])["refused"])

    def test_acl_filters_corpus_before_idf_and_ranking(self):
        public, _ = self.ingest("Orchid soil drains water. Orchid roots need air.")
        first = self.core.search("orchid soil", ["public"])
        for i in range(4):
            self.ingest(("Orchid soil water roots " * (10 + i)) + str(i), f"secret-{i}.txt", "private")
        second = self.core.search("orchid soil", ["public"])
        self.assertEqual(first, second)
        self.assertEqual(second[0]["document_id"], public["id"])

    def test_dedupe_is_per_kb_and_public_metadata_has_no_path(self):
        first, source = self.ingest("Stable checksum evidence.", "../nested/manual.txt")
        again = self.core.ingest_file(source, "different.txt", "public")
        other = self.core.ingest_file(source, "manual.txt", "private")
        self.assertFalse(first["duplicate"])
        self.assertTrue(again["duplicate"])
        self.assertEqual(first["id"], again["id"])
        self.assertNotEqual(first["id"], other["id"])
        self.assertFalse(other["duplicate"])
        self.assertEqual(first["name"], "manual.txt")
        managed = self.core.source_path(first["id"], None)
        self.assertEqual(managed.parent, self.data / "documents")
        self.assertEqual(managed.read_bytes(), source.read_bytes())
        self.assertNotIn("storage_name", first)
        self.assertNotIn("sha256", first)

    def test_markdown_true_lines_and_headings(self):
        text = "# Garden\nintro\n\n## Water\nOrchid watering is weekly.\n\n## Air\nRoots need ventilation."
        document, _ = self.ingest(text)
        pieces = self.core.get_document_chunks(document["id"], None)
        water = next(piece for piece in pieces if "watering" in piece["text"])
        self.assertEqual(water["section"], "Garden / Water")
        self.assertIsNone(water["page"])
        self.assertEqual((water["line_start"], water["line_end"]), (4, 5))
        lines = text.splitlines()
        self.assertEqual(water["text"], "\n".join(lines[3:6]).strip("\n"))
        self.assertEqual(self.core.get_document_chunks(document["id"], None, 1), [])

    def test_code_fence_does_not_become_markdown_section(self):
        document, _ = self.ingest("# Real\n```python\n# This is a comment\n```\nend")
        pieces = self.core.get_document_chunks(document["id"], None)
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0]["section"], "Real")

    def test_long_line_chunks_keep_original_source_line(self):
        parsed = Parsed([Unit("A" * 1801 + "\nsecond", None, 7, "Section")], "text-utf8", None)
        result = chunks(parsed)
        self.assertGreaterEqual(len(result), 3)
        self.assertTrue(all(len(piece["text"]) <= 900 for piece in result))
        self.assertEqual(result[0]["line_start"], 7)
        self.assertEqual(result[1]["line_start"], 7)
        self.assertEqual(result[-1]["line_end"], 8)

    def test_real_pdf_cites_second_physical_page(self):
        source = self.root / "two-pages.pdf"
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((60, 80), "FIRSTPAGE flower glossary.")
            pdf.new_page().insert_text((60, 80), "SECONDPAGEEVIDENCE orchid roots require airflow.")
            pdf.save(source)
        document = self.core.ingest_file(source, source.name, "public")
        self.assertEqual(document["page_count"], 2)
        self.assertEqual(document["parser"], "pdf-pypdf")
        hits = self.core.search("SECONDPAGEEVIDENCE", ["public"])
        self.assertEqual(hits[0]["page"], 2)
        self.assertEqual(len(self.core.get_document_chunks(document["id"], None, 2)), 1)
        self.assertIn("SECONDPAGEEVIDENCE", self.core.ask("SECONDPAGEEVIDENCE", None)["answer"])

    def test_docx_real_package_logical_lines_and_table_text(self):
        from docx import Document
        source = self.root / "document.docx"
        word = Document()
        word.add_heading("Care guide", level=1)
        word.add_paragraph("Orchid watering schedule is weekly.")
        word.add_table(rows=1, cols=1).cell(0, 0).text = "Table airflow evidence."
        word.save(source)
        document = self.core.ingest_file(source, source.name, "public")
        pieces = self.core.get_document_chunks(document["id"], None)
        self.assertIsNone(document["page_count"])
        self.assertTrue(all(piece["page"] is None for piece in pieces))
        self.assertIn("Table airflow evidence.", "\n".join(piece["text"] for piece in pieces))
        self.assertEqual(pieces[0]["section"], "Care guide")
        self.assertEqual((pieces[0]["line_start"], pieces[0]["line_end"]), (1, 3))

    def test_delete_invalidates_cached_index_and_keeps_input(self):
        document, original = self.ingest("Uniquedeletionmarker means removable evidence.")
        second_core = KnowledgeCore(self.data)
        self.assertTrue(self.core.search("Uniquedeletionmarker", None))
        self.assertTrue(second_core.search("Uniquedeletionmarker", None))
        managed = self.core.source_path(document["id"], None)
        self.assertTrue(self.core.delete_document(document["id"], ["public"]))
        self.assertTrue(original.exists())
        self.assertFalse(managed.exists())
        self.assertEqual(self.core.search("Uniquedeletionmarker", None), [])
        self.assertEqual(second_core.search("Uniquedeletionmarker", None), [])
        self.assertEqual(self.core.get_document_chunks(document["id"], None), [])
        self.assertFalse(self.core.delete_document(document["id"], None))
        with closing(sqlite3.connect(self.core.database)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0)

    def test_delete_file_cleanup_retries_after_interruption(self):
        document, _ = self.ingest("cleanupmarker pending delete")
        managed = self.core.source_path(document["id"], None)
        real_unlink = Path.unlink
        def deny_owned_file(path, *args, **kwargs):
            if path == managed:
                raise PermissionError("Synthetic open handle")
            return real_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", deny_owned_file):
            self.assertTrue(self.core.delete_document(document["id"], None))
        self.assertIsNone(self.core.get_document(document["id"], None))
        self.assertTrue(managed.exists())
        KnowledgeCore(self.data)
        self.assertFalse(managed.exists())

    def test_tampered_storage_path_never_reads_or_deletes_outside(self):
        document, original = self.ingest("Safe managed evidence.")
        with closing(sqlite3.connect(self.core.database)) as db, db:
            db.execute("UPDATE documents SET storage_name=? WHERE id=?", ("../../" + original.name, document["id"]))
        self.assertIsNone(self.core.source_path(document["id"], None))
        self.assertFalse(self.core.delete_document(document["id"], None))
        self.assertTrue(original.exists())

    def test_failed_parse_does_not_leave_rows_or_staged_copies(self):
        for filename, contents in (("bad.pdf", b"not PDF"), ("bad.docx", b"not ZIP"),
                                   ("binary.txt", b"abc\0def"), ("wrong.txt", b"\xff\xfe"),
                                   ("empty.txt", b" \n"), ("bad.exe", b"text")):
            source = self.root / filename
            source.write_bytes(contents)
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.core.ingest_file(source, filename, "public")
        self.assertEqual(self.core.list_documents(None), [])
        self.assertEqual(list((self.data / ".staging").iterdir()), [])
        self.assertEqual(list((self.data / "documents").iterdir()), [])

    def test_docx_decompression_limit_and_external_entities(self):
        oversized = self.root / "bomb.docx"
        with ZipFile(oversized, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", b"x" * (MAX_XML_BYTES + 1))
        self.assertLess(oversized.stat().st_size, 100_000)
        with self.assertRaisesRegex(ValueError, "上限"):
            self.core.ingest_file(oversized, oversized.name, "public")
        malicious = self.root / "entity.docx"
        with ZipFile(malicious, "w") as archive:
            archive.writestr("word/document.xml", b'<!DOCTYPE x [<!ENTITY a "secret">]><x>&a;</x>')
        with self.assertRaises(ValueError):
            self.core.ingest_file(malicious, malicious.name, "public")
        self.assertEqual(self.core.list_documents(None), [])

    def test_size_limit_rolls_back_staging(self):
        source = self.root / "limit.txt"
        source.write_text("123456789", encoding="utf-8")
        with patch("knowtrace.core.MAX_FILE_BYTES", 8), self.assertRaises(ValueError):
            self.core.ingest_file(source, source.name, "public")
        self.assertEqual(self.core.list_documents(None), [])
        self.assertEqual(list((self.data / ".staging").iterdir()), [])

    def test_concurrent_ingestion_dedupes_before_commit(self):
        source = self.root / "concurrent.txt"
        source.write_text("Concurrent unique evidence", encoding="utf-8")
        second_core = KnowledgeCore(self.data)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda core: core.ingest_file(source, source.name, "public"),
                                    (self.core, second_core)))
        self.assertEqual(results[0]["id"], results[1]["id"])
        self.assertEqual(sorted(result["duplicate"] for result in results), [False, True])
        self.assertEqual(len(self.core.list_documents(None)), 1)
        self.assertEqual(len(list((self.data / "documents").iterdir())), 1)

    def test_handwritten_queries_extract_verbatim_or_refuse(self):
        sentence = "Orchid roots require airflow. Water orchids weekly and let the soil drain."
        document, _ = self.ingest(sentence, "care.txt")
        for question in ("orchid roots", "What do orchid roots require?", "weekly orchid"):
            with self.subTest(question=question):
                response = self.core.ask(question, ["public"])
                self.assertFalse(response["refused"])
                self.assertEqual(response["answer_mode"], "extractive")
                self.assertIn(sentence, response["answer"])
                self.assertIn("[S1]", response["answer"])
                self.assertEqual(response["sources"][0]["document_id"], document["id"])
        response = self.core.ask("Who is the CEO of Orchid International?", ["public"])
        self.assertTrue(response["refused"])
        self.assertEqual(response["sources"], [])
        # Generic inflection normalization now matches water/watering/orchids;
        # this previously documented false refusal is a regression to prevent.
        self.assertTrue(self.core.search("orchid watering", ["public"]))
        self.assertFalse(self.core.ask("orchid watering", ["public"])["refused"])

    def test_modes_result_contract_stable_and_validated(self):
        document, _ = self.ingest("A botanical guide to orchid roots and airflow.", source_url="https://example.org/guide")
        required = {"chunk_id", "document_id", "kb_id", "document_name", "text", "page", "section",
                    "line_start", "line_end", "score", "bm25_score", "tfidf_score", "coverage", "source_url"}
        for mode in ("bm25", "tfidf", "hybrid"):
            hits = self.core.search("orchid roots", ["public"], mode)
            self.assertEqual(hits, self.core.search("orchid roots", ["public"], mode))
            self.assertTrue(required <= hits[0].keys())
            self.assertEqual(hits[0]["document_id"], document["id"])
            self.assertGreater(hits[0]["score"], 0)
        for kwargs in ({"top_k": 0}, {"top_k": True}, {"top_k": 21}, {"mode": "neural"}):
            with self.assertRaises(ValueError):
                self.core.search("orchid", None, **kwargs)
        for acl in ("public", 42, ["public", 1]):
            with self.assertRaises(ValueError):
                self.core.list_kbs(acl)


if __name__ == "__main__":
    unittest.main()
