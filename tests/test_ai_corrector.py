import unittest

from core.ai_corrector import DEFAULT_AI_PROMPTS, get_default_ai_prompt


class AiCorrectorPromptTests(unittest.TestCase):
    def test_all_seven_locales_have_non_intrusive_proofreading_prompts(self) -> None:
        expected = ("uk", "ru", "en", "pl", "de", "fr", "es")
        for lang in expected:
            self.assertIn(lang, DEFAULT_AI_PROMPTS)
            prompt = get_default_ai_prompt(lang)
            self.assertTrue(prompt.strip())
            # Ensure we keep digits/numbers intact instead of scrubbing "junk"
            self.assertIn("1, 100, 50%", prompt)


if __name__ == "__main__":
    unittest.main()
