import json
import os
import re
import tempfile
from pathlib import Path
from threading import Lock


_CONFIG_LOCK = Lock()
MAX_QUESTION_LENGTH = 1000


def normalize_question(text):
	if not isinstance(text, str):
		raise ValueError("Question text must be a string.")
	text = " ".join(text.split())
	if not text or len(text) > MAX_QUESTION_LENGTH:
		raise ValueError(f"Questions must contain 1 to {MAX_QUESTION_LENGTH} characters.")
	return text


def validate_question_config(config):
	if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] != 1:
		raise ValueError("Question configuration must use version 1.")
	questions = config.get("suggested_questions")
	if not isinstance(questions, list):
		raise ValueError("suggested_questions must be a list.")
	identifiers = set()
	texts = set()
	validated = []
	for question in questions:
		if not isinstance(question, dict):
			raise ValueError("Each question must be an object.")
		identifier = question.get("id")
		if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier) or identifier in identifiers:
			raise ValueError("Question IDs must be unique and contain only letters, numbers, underscores, or hyphens.")
		text = normalize_question(question.get("text"))
		if text.casefold() in texts:
			raise ValueError("Duplicate questions are not allowed.")
		if type(question.get("enabled")) is not bool:
			raise ValueError("Each question must have a boolean enabled value.")
		identifiers.add(identifier)
		texts.add(text.casefold())
		validated.append({"id": identifier, "text": text, "enabled": question["enabled"]})
	return {"version": 1, "suggested_questions": validated}


def load_question_config(path):
	with Path(path).open(encoding="utf-8") as config_file:
		return validate_question_config(json.load(config_file))


def save_question_config(path, config, expected_config):
	path = Path(path)
	config = validate_question_config(config)
	with _CONFIG_LOCK:
		if load_question_config(path) != expected_config:
			raise ValueError("The shared configuration changed. Close and reopen settings before saving.")
		temporary_path = None
		try:
			with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as config_file:
				temporary_path = Path(config_file.name)
				json.dump(config, config_file, ensure_ascii=True, indent=2)
				config_file.write("\n")
				config_file.flush()
				os.fsync(config_file.fileno())
			os.replace(temporary_path, path)
		finally:
			if temporary_path is not None:
				temporary_path.unlink(missing_ok=True)