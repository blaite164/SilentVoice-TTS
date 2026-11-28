import unittest
from unittest.mock import MagicMock, patch

from core.ai_corrector import (
    DEFAULT_AI_PROMPTS,
    _call_ai_item,
    _clean_ai_output,
    _get_live_ai_timeout,
    correct_text_with_ai,
    get_default_ai_prompt,
)


class AiCorrectorPromptTests(unittest.TestCase):
    def test_all_seven_locales_have_non_intrusive_proofreading_prompts(self) -> None:
        expected = ("uk", "ru", "en", "pl", "de", "fr", "es")
        for lang in expected:
            self.assertIn(lang, DEFAULT_AI_PROMPTS)
            prompt = get_default_ai_prompt(lang)
            self.assertTrue(prompt.strip())
            # Ensure we keep digits/numbers intact instead of scrubbing "junk"
            self.assertIn("1, 100, 50%", prompt)


class AiCorrectorLiveCallTests(unittest.TestCase):
    def test_live_timeouts_match_provider_latency(self) -> None:
        self.assertGreaterEqual(_get_live_ai_timeout({"provider": "local_api"}), 30.0)
        self.assertGreaterEqual(_get_live_ai_timeout({"provider": "ollama"}), 45.0)
        self.assertGreaterEqual(_get_live_ai_timeout({"provider": "groq"}), 12.0)

    def test_clean_ai_output_strips_closed_and_unclosed_think_blocks(self) -> None:
        self.assertEqual(_clean_ai_output("<think>reasoning</think>\nПривет!"), "Привет!")
        self.assertEqual(_clean_ai_output("<think>truncated reasoning without closing tag"), "")
        self.assertEqual(_clean_ai_output("Привет!<think>trailing"), "Привет!")

    def test_local_api_uses_generous_max_tokens_for_thinking_models(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "<think>ok</think>Привет, как дела?"}}]
        }
        mock_sess = MagicMock()
        mock_sess.post.return_value = mock_resp

        item = {
            "id": "ai_local_1",
            "provider": "local_api",
            "title": "LM Studio",
            "base_url": "http://localhost:1234/v1",
            "model": "qwen3-4b",
        }
        with patch("core.ai_corrector._get_ai_http_session", return_value=mock_sess):
            out = _call_ai_item(item, "привет как дила", "prompt", timeout=30.0)

        self.assertEqual(out, "Привет, как дела?")
        sent_payload = mock_sess.post.call_args.kwargs["json"]
        self.assertGreaterEqual(sent_payload["max_tokens"], 1024)
        self.assertEqual(mock_sess.post.call_args.kwargs["timeout"], 30.0)

    def test_correct_text_with_ai_uses_live_timeout_for_local_api(self) -> None:
        item = {
            "id": "ai_local_1",
            "provider": "local_api",
            "title": "LM Studio",
            "base_url": "http://localhost:1234/v1",
            "model": "qwen3-4b",
        }
        fake_cfg = {
            "ai_spellcheck_enabled": True,
            "ai_items": [item],
            "ai_primary_id": "ai_local_1",
            "ai_fallback_id": "",
            "app_language": "ru",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Привет, мир!"}}]
        }
        mock_sess = MagicMock()
        mock_sess.post.return_value = mock_resp
        with (
            patch("core.ai_corrector.config.get", side_effect=lambda k, d=None: fake_cfg.get(k, d)),
            patch("core.ai_corrector._extract_protected_reaction_spans", return_value=(["привет мир"], [])),
            patch("core.ai_corrector._get_ai_http_session", return_value=mock_sess),
        ):
            res = correct_text_with_ai("привет мир")

        self.assertEqual(res, "Привет, мир!")
        self.assertEqual(mock_sess.post.call_args.kwargs["timeout"], 30.0)

    def test_correct_text_with_ai_preserves_russian_slash_tags_with_lmstudio_provider(self) -> None:
        """_extract_protected_reaction_spans respects app_language='ru' and restores [laughter]/[whisper]."""
        item = {
            "id": "ai_lms_1",
            "provider": "local_api",
            "title": "LM Studio",
            "url": "http://127.0.0.1:1234/v1",
            "model": "gemma-3-4b-it",
        }
        fake_cfg = {
            "ai_spellcheck_enabled": True,
            "ai_items": [item],
            "ai_primary_id": "ai_lms_1",
            "ai_fallback_id": "",
            "app_language": "ru",
            "reaction_omnivoice_enabled": True,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "⟪T0⟫ Привет, ⟪T1⟫ меня зовут Дима"}}]
        }
        mock_sess = MagicMock()
        mock_sess.post.return_value = mock_resp
        with (
            patch("core.ai_corrector.config.get", side_effect=lambda k, d=None: fake_cfg.get(k, d)),
            patch("core.ai_corrector._get_ai_http_session", return_value=mock_sess),
        ):
            res = correct_text_with_ai(r"\хаха привет, \тихо меня зовут дима")

        self.assertEqual(res, "[laughter] Привет, [whisper] меня зовут Дима")


if __name__ == "__main__":
    unittest.main()
