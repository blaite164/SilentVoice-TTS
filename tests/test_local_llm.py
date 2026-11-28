import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import core.llama_tts as llama_tts
import core.local_llm as local_llm


class LocalLlmGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls._qt_app = QApplication.instance() or QApplication([])

    def test_ensure_llama_server_reuses_tts_vulkan_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_dir = root / "llama_cpp"
            tts_dir = root / "llama_tts"
            tts_dir.mkdir()
            tts_srv = tts_dir / "llama-server.exe"
            tts_srv.write_bytes(b"srv")
            (tts_dir / "ggml-vulkan.dll").write_bytes(b"vk")

            with (
                patch.object(local_llm, "LLAMA_DIR", llama_dir),
                patch.object(local_llm, "LLAMA_EXE", llama_dir / "llama-server.exe"),
                patch.object(llama_tts, "LLAMA_TTS_DIR", tts_dir),
                patch.object(llama_tts, "LLAMA_TTS_SERVER_EXE", tts_srv),
            ):
                exe = local_llm.ensure_llama_server()
                self.assertEqual(exe, tts_srv)

    def test_ensure_llama_server_downloads_vulkan_zip_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_dir = root / "llama_cpp"
            tts_dir = root / "llama_tts"

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                zf.writestr("llama-server.exe", b"srv-bin")
                zf.writestr("ggml-vulkan.dll", b"vk-bin")
            zip_bytes = zip_buf.getvalue()

            def _fake_dl(url, dest, *args, **kwargs):
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(zip_bytes)

            with (
                patch.object(local_llm, "LLAMA_DIR", llama_dir),
                patch.object(local_llm, "LLAMA_EXE", llama_dir / "llama-server.exe"),
                patch.object(llama_tts, "LLAMA_TTS_DIR", tts_dir),
                patch.object(llama_tts, "LLAMA_TTS_SERVER_EXE", tts_dir / "llama-server.exe"),
                patch.object(llama_tts, "resolve_llama_cpp_win_asset_url", return_value="https://example.com/llama-win-vulkan-x64.zip"),
                patch("core.tts_manager.tts_manager._download_url_with_progress", side_effect=_fake_dl),
            ):
                exe = local_llm.ensure_llama_server()
                self.assertEqual(exe, llama_dir / "llama-server.exe")
                self.assertTrue((llama_dir / "llama-server.exe").is_file())
                self.assertTrue((llama_dir / "ggml-vulkan.dll").is_file())

    def test_start_llama_cpp_server_uses_ngl_99_and_falls_back_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "llama-server.exe"
            exe.write_bytes(b"srv")
            (root / "ggml-vulkan.dll").write_bytes(b"vk")
            model = root / "model.gguf"
            model.write_bytes(b"gguf")

            seen_cmds = []

            def _fake_popen(cmd, **kwargs):
                seen_cmds.append(list(cmd))
                proc = MagicMock()
                if cmd[cmd.index("-ngl") + 1] == "99":
                    proc.poll.return_value = 1
                else:
                    proc.poll.return_value = None
                return proc

            def _fake_models_at(url):
                return [] if len(seen_cmds) < 2 else ["model.gguf"]

            with (
                patch.object(local_llm, "DATA_DIR", root),
                patch.object(local_llm, "ensure_llama_server", return_value=exe),
                patch.object(local_llm, "_models_at", side_effect=_fake_models_at),
                patch("core.local_llm.subprocess.Popen", side_effect=_fake_popen),
                patch("core.local_llm.time.sleep"),
            ):
                try:
                    url, models = local_llm.start_llama_cpp_server(str(model), port=45678)
                finally:
                    local_llm.stop_managed_llama_server()

            self.assertEqual(url, "http://127.0.0.1:45678/v1")
            self.assertEqual(models, ["model.gguf"])
            self.assertEqual(seen_cmds[0][seen_cmds[0].index("-ngl") + 1], "99")
            self.assertEqual(seen_cmds[1][seen_cmds[1].index("-ngl") + 1], "0")

    def test_list_and_download_recommended_lmstudio_model_handles_prompt_and_100_pct(self) -> None:
        """list_lmstudio_models preserves path and lms import confirms Y/n prompts without stalling at 99%."""
        sample_json = (
            'Notice\n[{"type":"llm","modelKey":"gemma-3-4b-it","displayName":"Gemma 3 4B Instruct",'
            '"path":"lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf"}]'
        )
        completed = MagicMock(returncode=0, stdout=sample_json, stderr="")
        with patch.object(local_llm, "_run_lms", return_value=completed):
            listed = local_llm.list_lmstudio_models()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["key"], "gemma-3-4b-it")
        self.assertIn("gemma-3-4b-it-GGUF", listed[0]["path"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            emitted = []

            class _FakeSignal:
                def emit(self, eng: str, pct: int, lbl: str) -> None:
                    emitted.append((eng, pct, lbl))

            fake_mgr = MagicMock()
            fake_mgr.download_progress = _FakeSignal()

            def _fake_download(url, dest, engine_id, start, end, label, cancel_event=None):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"GGUF-DATA")
                emitted.append((engine_id, 99, label))

            fake_mgr._download_url_with_progress.side_effect = _fake_download

            import io
            fake_proc = MagicMock()
            fake_proc.stdin = io.StringIO()
            fake_proc.stdout = io.StringIO("ok\n")
            fake_proc.stderr = io.StringIO("")
            fake_proc.poll.side_effect = [None, 0]
            fake_proc.returncode = 0

            with (
                patch.object(local_llm, "DATA_DIR", root),
                patch.object(local_llm, "find_lms_cli", return_value="lms.exe"),
                patch("core.tts_manager.tts_manager", fake_mgr),
                patch("core.local_llm.subprocess.Popen", return_value=fake_proc) as mock_popen,
                patch("core.local_llm.time.sleep"),
            ):
                repo = local_llm.download_recommended_lmstudio_model(
                    "https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
                    "gemma-3-4b-it-Q4_K_M.gguf",
                    "lmstudio-community/gemma-3-4b-it-GGUF",
                )

            self.assertEqual(repo, "lmstudio-community/gemma-3-4b-it-GGUF")
            self.assertEqual(emitted[-1][1], 100)
            self.assertTrue(any(pct == 99 for _, pct, _ in emitted))
            self.assertFalse((root / "model_downloads" / "lmstudio" / "gemma-3-4b-it-Q4_K_M.gguf").exists())
            self.assertEqual(mock_popen.call_args.kwargs["stdin"], local_llm.subprocess.PIPE)

    def test_local_models_dialog_selects_lms_model_by_repo_path(self) -> None:
        """LocalModelsDialog selects gemma-3-4b-it when _preferred_lms_model_key holds repo path."""
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from ui.dialogs import LocalModelsDialog

        with patch("ui.dialogs.list_lmstudio_models", return_value=[]):
            dlg = LocalModelsDialog()
        try:
            dlg._preferred_lms_model_key = "lmstudio-community/gemma-3-4b-it-GGUF"
            dlg._on_lmstudio_models_fetched(
                [
                    {"key": "qwen/qwen3.5-2b", "label": "Qwen", "path": "qwen/qwen3.5-2b"},
                    {
                        "key": "gemma-3-4b-it",
                        "label": "Gemma 3 4B Instruct",
                        "path": "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf",
                    },
                ],
                "",
            )
            self.assertEqual(dlg.model_combo.currentData(), "gemma-3-4b-it")
        finally:
            dlg.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
