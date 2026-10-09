import ast
import shutil
import subprocess
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class SavedReportsTests(unittest.TestCase):
	def setUp(self):
		self.module = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
		names = {"export_scope_details", "render_export_controls", "create_report_item", "report_export", "render_report", "render_report_actions", "queue_saved_report_operation", "queue_saved_chat_operation", "queue_saved_item_operation", "conversation_export", "clean_text", "render_conversation_actions", "remove_chat_exchange"}
		self.nodes = [node for node in self.module.body if isinstance(node, ast.FunctionDef) and node.name in names]
		self.script = """import streamlit as st
import pandas as pd
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