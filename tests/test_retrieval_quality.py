"""Independent synthetic query regressions, not evaluation answer fixtures."""

from pathlib import Path
import tempfile
import unittest

from knowtrace.core import KnowledgeCore
from knowtrace.retrieval import tokens


class RetrievalQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.core = KnowledgeCore(self.root / "data")
        self.core.create_kb("Garden", kb_id="garden")
        self.core.create_kb("Operations", kb_id="ops")
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def add(self, text, kb="garden"):
        self.counter += 1
        file = self.root / f"source-{self.counter}.txt"
        file.write_text(text, encoding="utf-8")
        return self.core.ingest_file(file, file.name, kb)

    def test_chinese_word_boundaries_remove_function_words_not_negation(self):
        values = tokens("请问，这个服务的缓存应该如何清理呢？")
        self.assertIn("缓存", values)
        self.assertIn("清理", values)
        self.assertNotIn("的", values)
        self.assertNotIn("该如", values)
        self.assertTrue(any(part in ("不", "不能", "不可") for part in tokens("不支持删除")))
        self.assertIn("not", tokens("doesn't support deletion"))
        self.add("缓存清理：管理员每天清理服务的缓存。")
        result = self.core.ask("请问这个服务的缓存应该如何清理呢？", ["garden"])
        self.assertFalse(result["refused"])

    def test_generic_english_inflections_and_identifier_spelling(self):
        self.add("Orchid roots require airflow. Water orchids weekly and let the soil drain.")
        result = self.core.ask("orchid watering", ["garden"])
        self.assertFalse(result["refused"])
        self.assertEqual(tokens("caches cached caching"), ["cach", "cach", "cach"])
        self.assertEqual(tokens("USER-AGENT user_agent UserAgent"), ["useragent"] * 3)
        self.assertNotEqual(tokens("version 1.2"), tokens("version 12"))

    def test_unknown_technical_subject_is_not_discarded(self):
        self.add("RoseAPI returns JSON responses. RoseAPI supports a public endpoint.")
        response = self.core.ask("Does RoseAPI support QuasarAuth?", ["garden"])
        self.assertTrue(response["refused"])
        self.assertEqual(response["sources"], [])
        self.assertTrue(response["retrieval_candidates"])
        self.assertEqual(response["evidence_decision"]["status"], "refused")
        self.assertTrue(all(not hit["is_supporting_evidence"] for hit in response["retrieval_candidates"]))
        self.assertTrue(all("citation_id" not in hit for hit in response["retrieval_candidates"]))

    def test_date_constraint_does_not_silently_reuse_another_date(self):
        self.add("The release date for OrchidDesk is 2026-05-12. The release enables backups.")
        self.assertFalse(self.core.ask("OrchidDesk release 2026-05-12", ["garden"])["refused"])
        different = self.core.ask("OrchidDesk release 2029-05-12", ["garden"])
        self.assertTrue(different["refused"])
        self.assertEqual(different["sources"], [])

    def test_unknown_requested_fact_refuses_despite_familiar_anchor(self):
        self.add("OrchidDesk supports file uploads. Files are buffered in memory and then written to disk.")
        response = self.core.ask("What is the exact default maximum upload size in bytes for OrchidDesk?", ["garden"])
        self.assertTrue(response["refused"])
        self.assertTrue(response["retrieval_candidates"])

    def test_negated_prohibition_is_quoted_not_rewritten_to_permission(self):
        self.add("归档文件不可删除。历史记录必须保留。")
        response = self.core.ask("归档文件不可删除吗？", ["garden"])
        self.assertFalse(response["refused"])
        self.assertIn("归档文件不可删除。", response["answer"])
        self.assertNotIn("可以删除", response["answer"])

    def test_missing_explicit_negation_refuses_instead_of_accepting_overlap(self):
        self.add("Orchid roots require water.")
        response = self.core.ask("Do orchid roots not require water?", ["garden"])
        self.assertTrue(response["refused"])
        self.assertEqual(response["sources"], [])
        self.assertIn("missing_explicit_negation", response["retrieval_candidates"][0]["evidence_rejections"])
        self.add("归档文件支持删除。")
        self.assertTrue(self.core.ask("归档文件不支持删除吗？", ["garden"])["refused"])

    def test_qualitative_size_answer_is_not_mistaken_for_numeric_limit(self):
        self.add("上传时声明为 bytes 的文件会保存在内存，适合小文件。")
        result = self.core.ask("上传文件声明为 bytes 适合多大的文件？", ["garden"])
        # The remaining word-coverage gate may still refuse a paraphrase, but
        # it must not invent a requirement for a numeric answer to “多大”.
        self.assertNotIn("numeric_value_not_found", result["retrieval_candidates"][0]["evidence_rejections"])
        qualitative = self.core.ask("bytes 文件在内存，适合小文件吗？", ["garden"])
        self.assertFalse(qualitative["refused"])
        self.assertIn("适合小文件", qualitative["answer"])
        exact = self.core.ask("上传文件默认最大文件大小具体是多少字节？", ["garden"])
        self.assertTrue(exact["refused"])

    def test_no_unknown_terms_are_removed_just_because_they_are_absent(self):
        self.add("兰花每周浇水一次，盆土应排水良好。")
        response = self.core.ask("兰花的量子加密密钥如何轮换？", ["garden"])
        self.assertTrue(response["refused"])
        decision = response["evidence_decision"]
        self.assertEqual(decision["policy_version"], "lexical-v2")
        self.assertTrue(decision.get("query_terms"))
        self.assertTrue(any(term not in ("兰花",) for term in decision["query_terms"]))

    def test_cross_kb_candidates_and_statistics_cannot_leak(self):
        self.add("公开园艺信息：兰花每周浇水。")
        private = self.add("密钥轮换必须经过管理审批。SecretAuth accepts an owner token.", "ops")
        before = self.core.search("兰花 浇水", ["garden"])
        self.add("兰花 浇水 " * 40 + "SecretAuth", "ops")
        self.assertEqual(before, self.core.search("兰花 浇水", ["garden"]))
        response = self.core.ask("SecretAuth owner token", ["garden"])
        self.assertTrue(response["refused"])
        self.assertTrue(all(hit["kb_id"] == "garden" for hit in response["retrieval_candidates"]))
        self.assertNotIn(private["id"], str(response))
        self.assertIsNone(self.core.source_path(private["id"], ["garden"]))

    def test_only_quoted_support_sources_and_separate_candidates(self):
        for i in range(5):
            self.add(f"Orchid watering is weekly. Evidence version {i}.")
        response = self.core.ask("orchid watering", ["garden"], top_k=5)
        self.assertFalse(response["refused"])
        self.assertLessEqual(len(response["sources"]), 3)
        for source in response["sources"]:
            self.assertIn("[" + source["citation_id"] + "]", response["answer"])
            self.assertIn(source["text"], response["answer"])
        self.assertEqual(response["evidence_decision"]["status"], "accepted")
        self.assertTrue(response["evidence_decision"]["message"])
        self.assertEqual(len(response["retrieval_candidates"]), 5)

    def test_empty_acl_and_unknown_question_expose_no_support(self):
        self.add("Orchid watering is weekly.")
        response = self.core.ask("orchid watering", [])
        self.assertTrue(response["refused"])
        self.assertEqual(response["sources"], [])
        self.assertEqual(response["retrieval_candidates"], [])
        self.assertEqual(response["evidence_decision"]["status"], "refused")
        self.assertEqual(self.core.search("nebula cryptosystem", ["garden"], "bm25"), [])


if __name__ == "__main__":
    unittest.main()
