import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCALES_DIR = ROOT / "locales"
EXPECTED_LOCALES = ("de", "en", "es", "fr", "pl", "ru", "uk")
PH_RE = re.compile(r"\{[a-zA-Z0-9_]+(?::[^}]*)?\}")
TR_RE = re.compile(r"""\btr\(\s*["']([a-zA-Z0-9_.-]+)["']""")
CYR_RE = re.compile(r"[а-яА-ЯіІїЇєЄґҐ]")


class LocaleCompletenessTests(unittest.TestCase):
    def test_all_locales_have_matching_keys_and_placeholders(self) -> None:
        data_by_lang = {
            code: json.loads((LOCALES_DIR / f"{code}.json").read_text(encoding="utf-8"))
            for code in EXPECTED_LOCALES
        }
        uk_keys = set(data_by_lang["uk"].keys()) - {"_meta"}

        for code, data in data_by_lang.items():
            keys = set(data.keys()) - {"_meta"}
            self.assertEqual(keys, uk_keys, f"Locale {code}.json key mismatch")
            for k in uk_keys:
                self.assertEqual(
                    sorted(PH_RE.findall(str(data[k]))),
                    sorted(PH_RE.findall(str(data_by_lang["uk"][k]))),
                    f"Placeholder mismatch in {code}.json for {k}",
                )
            if code in ("de", "en", "es", "fr", "pl"):
                for k in uk_keys:
                    if k == "ui_216":
                        # ui_216 intentionally shows an example of Cyrillic QWERTY mismatch
                        continue
                    self.assertIsNone(
                        CYR_RE.search(str(data[k])),
                        f"Unexpected Cyrillic text in {code}.json[{k!r}]",
                    )

    def test_all_static_tr_keys_exist_in_all_locales(self) -> None:
        used_keys = set()
        py_files = [ROOT / "main.py"] + sorted((ROOT / "ui").glob("*.py")) + sorted((ROOT / "core").glob("*.py"))
        for path in py_files:
            for match in TR_RE.finditer(path.read_text(encoding="utf-8")):
                used_keys.add(match.group(1))

        for code in EXPECTED_LOCALES:
            data = json.loads((LOCALES_DIR / f"{code}.json").read_text(encoding="utf-8"))
            missing = sorted(used_keys - (set(data.keys()) - {"_meta"}))
            self.assertEqual(missing, [], f"Missing tr() keys in {code}.json: {missing}")


if __name__ == "__main__":
    unittest.main()
