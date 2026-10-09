# Content Intelligence

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

## Tests

```sh
./.venv/bin/python -m unittest discover -s tests -v
```