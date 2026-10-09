# Content Intelligence

## CSV Schema

CSV files in `data/` must provide these columns:

```text
Date published
URL
Title
Post type
Topics
Vertical
Buyer persona
Sales pitch
Strategic initiative
Features Mentioned
Main Pain Point
Solution
Word count
Funnel stage
Target audience
Summary
```

The filename determines the competitor name. The dashboard uses `Word count`
for article length and supports analyzing topics, post types, verticals, buyer
personas, strategic initiatives, mentioned features, and target audiences.

All populated CSV fields are included in article embeddings and retrieved AI
context. Strategy reports include distributions and article samples with the
new strategic fields. Chat still sends at most eight retrieved articles to Gemini.

After migrating from the old CSV schema, select the desired competitors and
run **Start Data Ingestion** to rebuild the search index. Old-schema indexes
are rejected rather than silently used with the new data. Rebuilding creates
new embeddings and incurs Gemini API usage; it is not done automatically.
The old `Topic`, `Category`, `Tags`, and `Lenght` schema is no longer supported.

## Suggested Questions

Shared team questions are stored in `suggested_questions.json`, not in Python.
The file uses version 1 and a `suggested_questions` array. Each question has a
unique `id`, a `text` value, and a boolean `enabled` value. Array order controls
display order. Empty lists are supported.

In the AI Assistant tab, expand **Suggested questions** and open **Shared
settings** to view the configuration. To enable adding, editing, deleting,
reordering, and enabling/disabling shared questions, set this in Streamlit secrets:

```toml
ALLOW_SHARED_QUESTION_EDITING = true
```

Alternatively, set `ALLOW_SHARED_QUESTION_EDITING=true` in the environment or
`.env`. Restart the server after changing this setting. Editing is disabled by
default. This flag is not authentication: enabling it allows all app users to
edit shared questions. Enable it only in a trusted deployment or behind access
controls.

The editor saves validated JSON with atomic file replacement. Invalid or
duplicate questions, stale edits, and storage errors leave the existing file
unchanged. Missing or invalid shared configuration does not disable the chat
or personal favorites.

For deployments, use a persistent writable directory containing the JSON file.
Set `SUGGESTED_QUESTIONS_CONFIG` to its file path to override the default
workspace file. The directory must allow temporary file creation and atomic
replacement. A repository redeploy can otherwise overwrite shared edits.

## Personal Preferences

**Add favorite** adds a personal question. The trash icon removes a personal
favorite or hides a shared suggestion only in the current browser.
**Restore shared questions** restores hidden shared suggestions without
removing favorites. Selecting a question submits it and collapses the box.
The list displays at most five question rows and scrolls internally.

Favorites and hidden shared question IDs are stored under
`content-intelligence.question-preferences.v1` in browser `localStorage`.
They survive reloads and dataset filter changes but are specific to the
browser and app origin. Clearing site data removes them. They do not change
the shared JSON or saved conversations.

## Saved Reports

Generated strategy reports have **Save** and **Download** controls with the same
Markdown, plain text, and HTML formats as chat conversations. Saved reports
appear under **Saved reports** in the sidebar and can be opened or deleted
individually. Opening a saved report does not call Gemini or regenerate its
content. Each saved report retains its generation time, competitor files,
analysis time range, and article count.

Reports use the separate `content-intelligence.saved-reports.v1` localStorage
key. They survive reloads and filter changes but are tied to the browser and
app origin. Clearing site data removes them. Saving or deleting reports does
not change saved conversations or question preferences. Export HTML can be
printed to PDF through the browser; there is no direct PDF export.

## Tests

```sh
./.venv/bin/python -m unittest discover -s tests -v
```