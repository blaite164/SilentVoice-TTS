How to add a new UI translation for SilentVoice:

1. Copy `en.json` in this folder (`locales/`).
2. Rename the copied file using a 2-letter language code, e.g. `it.json`, `pl.json`, `de.json`, `fr.json`, `es.json`.
3. (Optional) Update `_meta` at the top of the file:
   "_meta": {
     "code": "it",
     "short": "IT",
     "name": "Italiano"
   }
4. Translate all string values on the right side (keep the keys on the left side and any `{placeholders}` like `{name}` unchanged).
5. Restart SilentVoice — the new language will automatically appear in the UI language list.
