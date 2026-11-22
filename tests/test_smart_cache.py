import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.tts_cache import TTSAudioCache, hash_cache_key
from core.tts_manager import tts_manager


class TestTTSAudioCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmp.name)
        self._orig_cache = tts_manager._audio_cache
        tts_manager._audio_cache = TTSAudioCache(self.cache_dir)

    def tearDown(self) -> None:
        tts_manager._audio_cache = self._orig_cache
        self._tmp.cleanup()

    def test_ram_hit_and_copy_isolation(self) -> None:
        cache = TTSAudioCache(None, ram_max_items=10)
        samples = np.linspace(-0.4, 0.4, 1600, dtype=np.float32)
        key = ("куку", "edge_tts", "uk-UA-PolinaNeural", 1.0, 0, "uk")

        cache.store(key, samples, 24000, "куку")
        hit1 = cache.lookup(key)
        self.assertIsNotNone(hit1)
        out1, sr1, title1 = hit1  # type: ignore[misc]
        self.assertEqual(sr1, 24000)
        self.assertEqual(title1, "куку")
        self.assertTrue(np.allclose(out1, samples, atol=1e-5))

        # Mutating returned buffer must not corrupt cached audio
        out1[:] = 0.0
        hit2 = cache.lookup(key)
        self.assertIsNotNone(hit2)
        out2, _, _ = hit2  # type: ignore[misc]
        self.assertTrue(np.allclose(out2, samples, atol=1e-5))

    def test_lru_eviction_in_ram_only_mode(self) -> None:
        cache = TTSAudioCache(None, ram_max_items=2)
        a = np.full(400, 0.1, dtype=np.float32)
        b = np.full(400, 0.2, dtype=np.float32)
        c = np.full(400, 0.3, dtype=np.float32)

        cache.store(("a",), a, 24000, "a")
        cache.store(("b",), b, 24000, "b")
        # Touch 'a' so 'b' becomes least-recently-used
        self.assertIsNotNone(cache.lookup(("a",)))
        cache.store(("c",), c, 24000, "c")

        self.assertIsNotNone(cache.lookup(("a",)))
        self.assertIsNone(cache.lookup(("b",)))
        self.assertIsNotNone(cache.lookup(("c",)))

    def test_disk_persistence_across_restart(self) -> None:
        key = ("привіт", "edge_tts", "uk-UA-OstapNeural")
        samples = (0.3 * np.sin(np.linspace(0, 12 * np.pi, 2400))).astype(np.float32)

        cache1 = TTSAudioCache(self.cache_dir, ram_max_items=5, disk_recent_limit=10)
        cache1.store(key, samples, 24000, "привіт")
        cache1.flush()

        # New instance simulates app restart (cold RAM)
        cache2 = TTSAudioCache(self.cache_dir, ram_max_items=5, disk_recent_limit=10)
        hit = cache2.lookup(key)
        self.assertIsNotNone(hit)
        loaded, sr, title = hit  # type: ignore[misc]
        self.assertEqual(sr, 24000)
        self.assertEqual(title, "привіт")
        # PCM_16 quantization tolerance
        self.assertTrue(np.allclose(loaded, samples, atol=2e-4))

    def test_frequency_pinning_protects_from_recent_eviction(self) -> None:
        freq_key = ("дякую", "edge_tts", "v1")
        freq_samples = np.full(800, 0.25, dtype=np.float32)

        cache = TTSAudioCache(
            self.cache_dir,
            ram_max_items=1,  # force disk path when looking up older items
            disk_recent_limit=3,
            disk_pinned_limit=5,
            pin_threshold=3,
        )
        # Store once (hits=1), then lookup twice -> hits=3 => pinned!
        cache.store(freq_key, freq_samples, 24000, "дякую")
        self.assertIsNotNone(cache.lookup(freq_key))
        self.assertIsNotNone(cache.lookup(freq_key))
        self.assertGreaterEqual(cache.stats()["pinned_items"], 1)

        # Store 5 distinct unpinned phrases (more than disk_recent_limit=3)
        for i in range(5):
            k = (f"одноразова_{i}", "edge_tts", "v1")
            cache.store(k, np.full(800, 0.05 * (i + 1), dtype=np.float32), 24000, f"одноразова_{i}")

        stats = cache.stats()
        self.assertEqual(stats["pinned_items"], 1)
        self.assertLessEqual(stats["recent_items"], 3)
        cache.flush()

        # Reload from disk after restart: frequent phrase survived eviction, oldest unpinned did not
        reloaded = TTSAudioCache(
            self.cache_dir,
            ram_max_items=1,
            disk_recent_limit=3,
            disk_pinned_limit=5,
            pin_threshold=3,
        )
        self.assertIsNotNone(reloaded.lookup(freq_key))
        self.assertIsNone(reloaded.lookup(("одноразова_0", "edge_tts", "v1")))
        self.assertIsNotNone(reloaded.lookup(("одноразова_4", "edge_tts", "v1")))

    def test_tts_manager_key_and_flush_integration(self) -> None:
        k1 = tts_manager._build_audio_cache_key("куку", "edge_tts", "uk-UA-PolinaNeural")
        k2 = tts_manager._build_audio_cache_key("куку", "edge_tts", "uk-UA-OstapNeural")
        k3 = tts_manager._build_audio_cache_key("куку", "silero", "uk-UA-PolinaNeural")
        self.assertNotEqual(hash_cache_key(k1), hash_cache_key(k2))
        self.assertNotEqual(hash_cache_key(k1), hash_cache_key(k3))

        samples = np.full(1000, 0.15, dtype=np.float32)
        tts_manager._cache_store(k1, samples, 24000, "куку")
        tts_manager.flush_audio_cache()
        self.assertTrue((self.cache_dir / "index.json").exists())
        self.assertIsNotNone(tts_manager._cache_lookup(k1))

        tts_manager.clear_audio_cache()
        self.assertIsNone(tts_manager._cache_lookup(k1))


if __name__ == "__main__":
    unittest.main()
