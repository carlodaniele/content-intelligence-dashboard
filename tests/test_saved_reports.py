import ast
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import plotly.io as pio
from streamlit.testing.v1 import AppTest


class SavedReportsTests(unittest.TestCase):
	def setUp(self):
		self.module = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
		names = {"export_scope_details", "render_export_controls", "create_report_item", "report_export", "render_report", "render_report_actions", "queue_saved_report_operation", "queue_saved_chat_operation", "queue_saved_item_operation", "conversation_export", "clean_text", "render_conversation_actions", "remove_chat_exchange", "render_custom_report", "custom_report_content", "remove_custom_report_question", "split_category_cell", "explode_category_values", "build_dashboard_topic_chart", "build_dashboard_charts", "render_report_visuals", "article_length_series", "is_current_relevant_result", "generate_chat_answer"}
		self.nodes = [node for node in self.module.body if isinstance(node, ast.FunctionDef) and node.name in names]
		self.script = """import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from types import SimpleNamespace
from datetime import datetime, timezone
from uuid import uuid4
from html import escape
GEMINI_API_KEY = 'mock'
CHAT_MODEL = 'mock'
analysis_time_range = 'Last year'
frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})
st.session_state.setdefault('saved_reports_ready', True)
st.session_state.setdefault('saved_reports', [])
st.session_state.setdefault('saved_chats', [{'id': 'existing-chat'}])
def report_context(frame): return 'Mock context'
def clean_text(value): return '' if value is None else str(value)
def render_sources(sources, key_prefix='sources'): pass
def export_scope_details(item): return 'Scope details'
def article_document(row): return row['Document']
def topic_frequency_context(frame): return 'Complete dataset counts'
def get_genai_client(key):
    return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(text='## Report body')))
"""
		self.script += ast.unparse(ast.Module(body=self.nodes, type_ignores=[]))

	def test_report_save_and_identity(self):
		app = AppTest.from_string(self.script + "\nrender_report(frame)\n").run()
		next(button for button in app.button if button.label == "Generate Content Strategy Report").click().run()
		self.assertFalse(app.exception)
		report = app.session_state["strategy_report"]
		self.assertEqual(report["scope"], {"files": ["Kinsta.csv"], "time_range": "Last year", "articles": 1})
		app.run()
		self.assertEqual(app.session_state["strategy_report"]["id"], report["id"])
		app.button(key=f"report_save_{report['id']}").click().run()
		self.assertEqual(app.session_state["_saved_report_operation"]["item"], report)
		self.assertEqual(app.session_state["saved_chats"], [{"id": "existing-chat"}])
		app.session_state["_saved_report_operation"] = None
		app.session_state["saved_reports"] = [report]
		app.run()
		self.assertTrue(app.button(key=f"report_save_{report['id']}").disabled)
		next(button for button in app.button if button.label == "Generate Content Strategy Report").click().run()
		self.assertFalse(app.exception)
		self.assertNotEqual(app.session_state["strategy_report"]["id"], report["id"])
		self.assertEqual(app.session_state["saved_reports"], [report])

	def test_legacy_report_and_exports(self):
		app = AppTest.from_string(self.script + "\nst.session_state.setdefault('strategy_report', 'Legacy report')\nrender_report(frame)\n").run()
		self.assertFalse(app.exception)
		self.assertEqual(app.session_state["strategy_report"]["content"], "Legacy report")
		namespace = {}
		exec("from html import escape\nimport pandas as pd\n", namespace)
		exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "app.py", "exec"), namespace)
		item = {"id": "report", "title": "Report <script>", "content": "## Analysis\n<script>alert(1)</script>", "created_at": "2026-10-09", "scope": {"files": ["Kinsta.csv"], "time_range": "Last year", "articles": 10}}
		for file_format in ("Markdown", "Text", "HTML"):
			with self.subTest(file_format=file_format):
				exported = namespace["report_export"](item, file_format)
				self.assertIn("Kinsta.csv", exported)
				self.assertIn("Articles analyzed: 10", exported)
				self.assertIn("Last year", exported)
				self.assertIn("Analysis", exported)
		html = namespace["report_export"](item, "HTML")
		self.assertNotIn("<script>", html)
		self.assertIn("&lt;script&gt;", html)
		conversation = {"question": "Question", "answer": "Answer", "sources": []}
		self.assertIn("## Question", namespace["conversation_export"](conversation, "Markdown"))

	def test_custom_report_composes_answers_and_exports_charts(self):
		script = self.script.replace(
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})",
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv'], 'Sito': ['Kinsta'], 'Topics': ['AI'], 'Funnel stage': ['TOFU'], 'Word count': ['900'], 'Date published': [pd.Timestamp('2026-01-10')]})",
		)
		script += """
st.session_state.custom_report_answers = [{'id':'question-1','question':'How much AI content?','answer':'Kinsta has 1 AI topic article.','sources':[],'include':True}]
render_custom_report(frame)
"""
		app = AppTest.from_string(script).run()
		self.assertFalse(app.exception)
		next(button for button in app.button if button.label == "Generate custom report").click().run()
		self.assertFalse(app.exception)
		report = app.session_state["custom_strategy_report"]
		self.assertIn("Kinsta has 1 AI topic article.", report["content"])
		self.assertEqual(len(report["charts"]), 4)
		namespace = {"pio": pio, "escape": __import__("html").escape, "export_scope_details": lambda item: "Scope details"}
		exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "app.py", "exec"), namespace)
		html = namespace["report_export"](report, "HTML")
		self.assertIn("cdn.plot.ly", html)
		self.assertIn("Funnel Stage Distribution by Website", report["content"])

	def test_custom_report_question_form_uses_report_answer_mode(self):
		script = self.script.replace(
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv']})",
			"frame = pd.DataFrame({'_source_file': ['Kinsta.csv'], 'Sito': ['Kinsta'], 'Topics': ['AI'], 'Funnel stage': ['TOFU'], 'Word count': ['900'], 'Date published': [pd.Timestamp('2026-01-10')]})",
		)
		script += """
calls = []
get_indexed_collection = lambda: object()
generate_chat_answer = lambda question, collection, **kwargs: (f"report_mode={kwargs['report_mode']}; articles={len(kwargs['frame'])}", [])
render_custom_report(frame)
"""
		app = AppTest.from_string(script).run()
		app.text_area(key="custom_report_question_input").set_value("Count AI topics by website")
		next(button for button in app.button if button.label == "Generate answer").click().run()
		self.assertFalse(app.exception)
		self.assertTrue(
			app.session_state["custom_report_answers"],
			f"warnings={[item.value for item in app.warning]}; errors={[item.value for item in app.error]}",
		)
		self.assertEqual(
			app.session_state["custom_report_answers"][0]["question"], "Count AI topics by website",
			f"warnings={[item.value for item in app.warning]}; errors={[item.value for item in app.error]}",
		)
		self.assertEqual(app.session_state["custom_report_answers"][0]["answer"], "report_mode=True; articles=1")

	def test_chat_retrieval_rejects_irrelevant_stale_and_uncited_answers(self):
		import pandas as pd
		import re

		def ask(distance, indexed_document, generated_answer):
			current_document = "Current imported CSV article about product features."
			frame = pd.DataFrame({
				"_source_file": ["Kinsta.csv"], "Sito": ["Kinsta"], "Document": [current_document],
			})
			generation_calls = []
			collection = SimpleNamespace(
				count=lambda: 1,
				query=lambda **kwargs: {
					"ids": [["Kinsta.csv::0"]], "documents": [[indexed_document]],
					"metadatas": [[{"Sito": "Kinsta", "Title": "Imported article", "URL": "https://example.com"}]],
					"distances": [[distance]],
				},
			)
			client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: (
				generation_calls.append(kwargs) or SimpleNamespace(text=generated_answer)
			)))
			namespace = {
				"pd": pd, "re": re, "dataset": frame, "GEMINI_API_KEY": "mock", "CHAT_MODEL": "mock", "CHAT_RESULT_COUNT": 8,
				"MAX_RETRIEVAL_COSINE_DISTANCE": 0.65, "EMBEDDING_MODEL": "mock",
				"clean_text": lambda value: str(value or ""),
				"article_document": lambda row: row["Document"],
				"topic_frequency_context": lambda source_frame: "Complete dataset counts",
				"embed_texts": lambda *args: [[0.0]], "get_genai_client": lambda key: client,
			}
			exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "app.py", "exec"), namespace)
			result = namespace["generate_chat_answer"]("What does the data say?", collection, frame=frame)
			return result, generation_calls

		with self.subTest("irrelevant hit"):
			result, calls = ask(0.91, "Current imported CSV article about product features.", "Unsupported answer [Source 1]")
			self.assertIsNone(result)
			self.assertEqual(calls, [])
		with self.subTest("stale indexed text"):
			result, calls = ask(0.2, "Outdated article text.", "Unsupported answer [Source 1]")
			self.assertIsNone(result)
			self.assertEqual(calls, [])
		with self.subTest("missing citation"):
			result, calls = ask(0.2, "Current imported CSV article about product features.", "Unsupported answer")
			self.assertEqual(result, ("The selected data does not provide enough verified evidence to answer this question.", []))
			self.assertEqual(len(calls), 1)
		with self.subTest("valid citation"):
			result, calls = ask(0.2, "Current imported CSV article about product features.", "The source describes product features [Source 1].")
			self.assertEqual(result[0], "The source describes product features [Source 1].")
			self.assertEqual(len(result[1]), 1)
			self.assertEqual(len(calls), 1)

	def test_chat_controls_still_save_and_export(self):
		script = self.script + """
st.session_state.setdefault('saved_chats_ready', True)
item = {'id':'new-chat','question':'Question','answer':'Answer','sources':[]}
render_conversation_actions(item)
"""
		app = AppTest.from_string(script).run()
		self.assertFalse(app.exception)
		app.button(key="chat_save_new-chat").click().run()
		self.assertEqual(app.session_state["_saved_chat_operation"]["item"]["question"], "Question")
		self.assertNotIn("_saved_report_operation", app.session_state)
		app.selectbox(key="chat_format_new-chat").set_value("HTML").run()
		self.assertFalse(app.exception)
		self.assertEqual(app.selectbox(key="chat_format_new-chat").value, "HTML")

	@unittest.skipUnless(shutil.which("node"), "Node.js is needed to test browser storage JavaScript")
	def test_storage_isolation_and_failures(self):
		assignment = next(node for node in self.module.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "saved_chat_storage" for target in node.targets))
		javascript = next(ast.literal_eval(keyword.value) for keyword in assignment.value.keywords if keyword.arg == "js")
		script = "import assert from 'node:assert/strict';\n" + javascript.replace("export default function", "const mount = function") + """;
const store = new Map();
globalThis.window = {addEventListener() {}, removeEventListener() {}};
globalThis.localStorage = {getItem: key => store.get(key) || null, setItem: (key, value) => store.set(key, value)};
let snapshot;
let updates = 0;
function run(kind, operation = null, acknowledged = null) {
    mount({data: {kind, storage_key: `content-intelligence.saved-${kind}s.v1`, operation, snapshot: acknowledged},
        parentElement: {}, setStateValue: (name, value) => {snapshot = value; updates++;}});
    return snapshot;
}
const chat = {id:'chat',question:'Question',answer:'Answer',sources:[]};
const report = {id:'report',title:'Report',content:'Report body',scope:{files:['Kinsta.csv']}};
run('chat', {id:'chat-save', action:'save', item_id:chat.id, item:chat});
run('report', {id:'report-save', action:'save', item_id:report.id, item:report});
assert.equal(run('report').items[0].content, 'Report body');
const acknowledged = snapshot;
const before = updates;
run('report', null, acknowledged);
assert.equal(updates, before);
assert.equal(run('report', {id:'save-again',action:'save',item_id:report.id,item:report}).items.length, 1);
assert.equal(run('chat').items[0].question, 'Question');
assert.equal(run('report', {id:'delete',action:'delete',item_id:report.id}).items.length, 0);
assert.equal(run('chat').items.length, 1);
store.set('content-intelligence.saved-reports.v1', '{bad JSON');
assert.ok(run('report').error);
assert.equal(store.get('content-intelligence.saved-reports.v1'), '{bad JSON');
store.set('content-intelligence.saved-reports.v1', '[]');
localStorage.setItem = () => {throw new Error('Quota exceeded');};
assert.match(run('report', {id:'quota',action:'save',item_id:report.id,item:report}).error, /Quota/);
assert.equal(store.get('content-intelligence.saved-reports.v1'), '[]');
"""
		result = subprocess.run([shutil.which("node"), "--input-type=module"], input=script, text=True, capture_output=True)
		self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
	unittest.main()