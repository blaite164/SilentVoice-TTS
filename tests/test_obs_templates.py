import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.config import config
from core.obs_server import (
    BLAITE_CUSTOM_TEST_CSS,
    BLAITE_CUSTOM_TEST_HTML,
    BLAITE_CUSTOM_TEST_NAME,
    DEFAULT_CUSTOM_OBS_CSS,
    DEFAULT_CUSTOM_OBS_HTML,
    OBSSubtitleServer,
    ensure_custom_obs_templates,
    format_custom_template_label,
    list_custom_obs_templates,
    render_custom_page,
)


class ObsCustomTemplateTests(unittest.TestCase):
    def test_ensure_creates_and_preserves_custom_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpl_dir = Path(tmp) / "obs_templates"
            created = ensure_custom_obs_templates(tpl_dir)
            self.assertEqual(created, tpl_dir)
            html_file = tpl_dir / "index.html"
            css_file = tpl_dir / "style.css"
            blaite_html = tpl_dir / f"{BLAITE_CUSTOM_TEST_NAME}.html"
            blaite_css = tpl_dir / f"{BLAITE_CUSTOM_TEST_NAME}.css"
            self.assertTrue(html_file.exists())
            self.assertTrue(css_file.exists())
            self.assertTrue(blaite_html.exists())
            self.assertTrue(blaite_css.exists())
            self.assertEqual(html_file.read_text(encoding="utf-8"), DEFAULT_CUSTOM_OBS_HTML)
            self.assertEqual(css_file.read_text(encoding="utf-8"), DEFAULT_CUSTOM_OBS_CSS)
            self.assertEqual(blaite_html.read_text(encoding="utf-8"), BLAITE_CUSTOM_TEST_HTML)
            self.assertEqual(blaite_css.read_text(encoding="utf-8"), BLAITE_CUSTOM_TEST_CSS)
            self.assertIn('<html lang="en">', DEFAULT_CUSTOM_OBS_HTML)
            self.assertIn("HUMAN & AI GUIDE", DEFAULT_CUSTOM_OBS_HTML)
            self.assertIn("Human guide:", DEFAULT_CUSTOM_OBS_CSS)
            self.assertNotRegex(DEFAULT_CUSTOM_OBS_HTML, r"[а-яА-ЯіІїЇєЄґҐ]")
            self.assertNotRegex(DEFAULT_CUSTOM_OBS_CSS, r"[а-яА-ЯіІїЇєЄґҐ]")

            # User edits must NOT be overwritten when overwrite=False
            css_file.write_text("#subtitle-box { color: #00FF88; }", encoding="utf-8")
            html_file.write_text("<html><head><style>{{CSS}}</style></head><body>CUSTOM</body></html>", encoding="utf-8")
            ensure_custom_obs_templates(tpl_dir, overwrite=False)

            rendered = render_custom_page(tpl_dir)
            self.assertIn("#00FF88", rendered)
            self.assertIn("CUSTOM", rendered)
            self.assertNotIn("{{CSS}}", rendered)

            # Explicit reset overwrites back to defaults
            ensure_custom_obs_templates(tpl_dir, overwrite=True)
            self.assertEqual(css_file.read_text(encoding="utf-8"), DEFAULT_CUSTOM_OBS_CSS)

    def test_discovery_formats_underscores_and_renders_named_templates(self) -> None:
        self.assertEqual(format_custom_template_label("Blaite_castom_test"), "Blaite castom test")
        self.assertEqual(format_custom_template_label("Yri1"), "Yri1")
        self.assertEqual(format_custom_template_label("Yri_1"), "Yri 1")

        with tempfile.TemporaryDirectory() as tmp:
            tpl_dir = Path(tmp)
            (tpl_dir / "Yri1.html").write_text("<html><head><style>{{CSS}}</style></head><body>YRI1</body></html>", encoding="utf-8")
            (tpl_dir / "Yri1.css").write_text(".y1 { color: red; }", encoding="utf-8")
            (tpl_dir / "Yri_1.html").write_text("<html><head><style>{{CSS}}</style></head><body>YRI_US</body></html>", encoding="utf-8")
            (tpl_dir / "Yri_1.css").write_text(".y2 { color: blue; }", encoding="utf-8")

            entries = list_custom_obs_templates(tpl_dir)
            labels_by_mode = {e["mode"]: e["label"] for e in entries}
            self.assertEqual(entries[0]["mode"], f"custom:{BLAITE_CUSTOM_TEST_NAME}")
            self.assertEqual(labels_by_mode[f"custom:{BLAITE_CUSTOM_TEST_NAME}"], "Blaite castom test")
            self.assertEqual(labels_by_mode["custom:Yri1"], "Yri1")
            self.assertEqual(labels_by_mode["custom:Yri_1"], "Yri 1")

            page = render_custom_page(tpl_dir, mode="custom:Yri_1")
            self.assertIn("YRI_US", page)
            self.assertIn(".y2 { color: blue; }", page)

    def test_server_render_page_switches_builtin_vs_custom_and_border_radius(self) -> None:
        srv = OBSSubtitleServer()
        with tempfile.TemporaryDirectory() as tmp:
            tpl_dir = Path(tmp)
            (tpl_dir / "index.html").write_text(
                "<!doctype html><html><head><style>{{CSS}}</style></head><body>MY_OBS_HTML</body></html>",
                encoding="utf-8",
            )
            (tpl_dir / "style.css").write_text(".my-cls { font-size: 42px; }", encoding="utf-8")

            with patch("core.obs_server.OBS_TEMPLATES_DIR", tpl_dir):
                with patch.object(config, "get", side_effect=lambda k, d=None: 22 if k == "obs_border_radius" else ("builtin" if k == "obs_template_mode" else d)):
                    self.assertEqual(srv.get_style_payload()["border_radius"], 22)
                    builtin_html = srv.render_page()
                    self.assertIn("SilentVoice OBS Subtitles", builtin_html)
                    self.assertNotIn("MY_OBS_HTML", builtin_html)

                with patch.object(config, "get", side_effect=lambda k, d=None: "custom" if k == "obs_template_mode" else d):
                    custom_html = srv.render_page()
                    self.assertIn("MY_OBS_HTML", custom_html)
                    self.assertIn(".my-cls { font-size: 42px; }", custom_html)

                with patch.object(config, "get", side_effect=lambda k, d=None: f"custom:{BLAITE_CUSTOM_TEST_NAME}" if k == "obs_template_mode" else d):
                    blaite_html = srv.render_page()
                    self.assertIn("VOICE TRANSMISSION", blaite_html)


if __name__ == "__main__":
    unittest.main()
