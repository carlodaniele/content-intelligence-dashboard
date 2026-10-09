import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from suggested_questions import load_question_config, normalize_question, save_question_config, validate_question_config


class SuggestedQuestionsTests(unittest.TestCase):
	def setUp(self):
		self.config = {"version": 1, "suggested_questions": [{"id": "first", "text": "Compare competitors?", "enabled": True}]}

	def test_shipped_questions(self):
		config = load_question_config(Path(__file__).resolve().parents[1] / "suggested_questions.json")
		self.assertEqual(len(config["suggested_questions"]), 10)

	def test_normalization(self):
		self.assertEqual(normalize_question("  Compare\n competitors?  "), "Compare competitors?")
		for text in (" ", None, "x" * 1001):
			with self.subTest(text=text), self.assertRaises(ValueError):
				normalize_question(text)

	def test_invalid_schema(self):
		for config in ({}, {"version": True, "suggested_questions": []}, {"version": 1, "suggested_questions": None}):
			with self.subTest(config=config), self.assertRaises(ValueError):
				validate_question_config(config)
		for changes in ({"id": "bad/id"}, {"enabled": "true"}, {"text": ""}):
			with self.subTest(changes=changes), self.assertRaises(ValueError):
				validate_question_config({"version": 1, "suggested_questions": [self.config["suggested_questions"][0] | changes]})

	def test_duplicates(self):
		for changes in ({"id": "first", "text": "Another question"}, {"id": "second", "text": " compare COMPETITORS? "}):
			with self.subTest(changes=changes), self.assertRaises(ValueError):
				validate_question_config({"version": 1, "suggested_questions": self.config["suggested_questions"] + [self.config["suggested_questions"][0] | changes]})

	def test_save_and_empty_list(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "questions.json"
			path.write_text(json.dumps(self.config), encoding="utf-8")
			empty = {"version": 1, "suggested_questions": []}
			save_question_config(path, empty, self.config)
			self.assertEqual(load_question_config(path), empty)
			with self.assertRaises(ValueError):
				save_question_config(path, self.config, self.config)

	def test_failed_write_preserves_original(self):
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "questions.json"
			path.write_text(json.dumps(self.config), encoding="utf-8")
			with patch("suggested_questions.os.replace", side_effect=OSError("Read-only storage")):
				with self.assertRaises(OSError):
					save_question_config(path, {"version": 1, "suggested_questions": []}, self.config)
			self.assertEqual(load_question_config(path), self.config)
			self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
	unittest.main()