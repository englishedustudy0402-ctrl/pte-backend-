import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.scoring_engine import grade  # noqa: E402


def make_question(task, content, answer_data=None):
    return {"task": task, "content": content, "answer_data": answer_data or {}}


def make_attempt(task):
    return {"task": task}


class ScoringEngineTest(unittest.TestCase):
    def test_mcq_single(self):
        r = grade(make_question("mcq", {"correct": [2]}, {}), make_attempt("mcq"), {"answer": "[2]"})
        self.assertEqual(r["overall_score"], 90)

    def test_mcq_multi_wrong(self):
        r = grade(make_question("mcq_multi", {"correct": [0, 2]}, {}), make_attempt("mcq_multi"), {"answer": "[0,1]"})
        self.assertEqual(r["overall_score"], 45)

    def test_reorder_3_of_5(self):
        content = {"correct": [0, 1, 2, 3, 4]}
        r = grade(make_question("reorder", content), make_attempt("reorder"), {"answer_text": json.dumps([0, 1, 3, 2, 4])})
        self.assertEqual(r["overall_score"], 54)

    def test_fill_blanks(self):
        content = {"blanks": {"1": {"correct": "support"}, "2": {"correct": "funding"}}}
        r = grade(make_question("fill_blanks", content), make_attempt("fill_blanks"), {"answer_text": json.dumps({"1": "support", "2": "wrong"})})
        self.assertEqual(r["overall_score"], 45)
        self.assertEqual(r["meta"], {"hits": 1, "total": 2})

    def test_dictation_exact(self):
        r = grade(make_question("dictation", {"text": "The library will be closed this month."}), make_attempt("dictation"), {"answer": "The library will be closed this month."})
        self.assertEqual(r["overall_score"], 90)

    def test_repeat_partial(self):
        r = grade(make_question("repeat_sentence", {"text": "Students must submit their assignments before Friday."}), make_attempt("repeat_sentence"), {"answer": "Students submit assignments Friday"})
        self.assertGreater(r["overall_score"], 0)

    def test_answer_short(self):
        q = make_question("answer_short", {}, {"correct_answers": ["claustrophobia"]})
        r = grade(q, make_attempt("answer_short"), {"answer": "claustrophobia"})
        self.assertEqual(r["overall_score"], 90)

    def test_hcs_selection(self):
        q = make_question("hcs", {}, {"correct": 1})
        r = grade(q, make_attempt("hcs"), {"answer": "[1]"})
        self.assertEqual(r["overall_score"], 90)

    def test_missing_word_list_answer(self):
        q = make_question("missing_word", {}, {"correct": 0})
        r = grade(q, make_attempt("missing_word"), {"answer": "[0]"})
        self.assertEqual(r["overall_score"], 90)

    def test_incorrect_words_text_indices(self):
        q = make_question("incorrect_words", {}, {"wrong_indices": [2, 7]})
        r = grade(q, make_attempt("incorrect_words"), {"answer_text": "[2,7]"})
        self.assertEqual(r["overall_score"], 90)

    def test_sst_keyword_coverage(self):
        q = make_question("sst", {}, {"keywords": ["climate", "funding", "awareness"]})
        r = grade(q, make_attempt("sst"), {"answer": "climate funding needs awareness"})
        self.assertEqual(r["overall_score"], 90)

    def test_empty_not_attempted(self):
        q = make_question("essay", {}, {"keywords": []})
        r = grade(q, make_attempt("essay"), {"answer": ""})
        self.assertEqual(r["overall_score"], 0)

    def test_components_populated(self):
        r = grade(make_question("dictation", {"text": "a b c"}), make_attempt("dictation"), {"answer": "a b c"})
        self.assertEqual(r["engine"], "deterministic")
        self.assertTrue(r["components"])
        self.assertIn("component", r["components"][0])


if __name__ == "__main__":
    unittest.main(verbosity=1)