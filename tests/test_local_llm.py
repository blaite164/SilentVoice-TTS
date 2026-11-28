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
            dlg.reject()
            dlg.close()
            app.processEvents()

    def test_is_lmstudio_installed_checks_cli_and_executable(self) -> None:
        import os

        with patch.object(local_llm, "find_lms_cli", return_value=r"C:\lms.exe"):
            self.assertTrue(local_llm.is_lmstudio_installed())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_patch = {"LOCALAPPDATA": str(root / "local"), "ProgramFiles": str(root / "pf")}
            with (
                patch.object(local_llm, "find_lms_cli", return_value=None),
                patch.dict(os.environ, env_patch, clear=False),
                patch("core.local_llm.os.name", "nt"),
            ):
                self.assertFalse(local_llm.is_lmstudio_installed())
                exe = root / "local" / "Programs" / "LM Studio" / "LM Studio.exe"
                exe.parent.mkdir(parents=True)
                exe.write_bytes(b"mz")
                self.assertTrue(local_llm.is_lmstudio_installed())

    def test_prune_ollama_when_local_model_active(self) -> None:
        items = [
            {"id": "ollama_1", "provider": "ollama", "model": "gemma3:4b"},
            {"id": "lms_1", "provider": "local_api", "managed_backend": "lmstudio", "model": "gemma-3-4b-it"},
            {"id": "groq_1", "provider": "groq", "model": "llama-3.3-70b"},
        ]
        next_items, prim, fb, removed = local_llm.prune_ollama_when_local_model_active(
            items,
            "ollama_1",
            "ollama_1",
            preferred_local_id="lms_1",
        )
        self.assertEqual(removed, ["ollama_1"])
        self.assertEqual([x["id"] for x in next_items], ["lms_1", "groq_1"])
        self.assertEqual(prim, "lms_1")
        self.assertEqual(fb, "")

        only_remote = [
            {"id": "ollama_1", "provider": "ollama", "model": "gemma3:4b"},
            {"id": "groq_1", "provider": "groq", "model": "llama-3.3-70b"},
        ]
        untouched, prim2, fb2, removed2 = local_llm.prune_ollama_when_local_model_active(
            only_remote,
            "ollama_1",
            "groq_1",
        )
        self.assertEqual(removed2, [])
        self.assertEqual([x["id"] for x in untouched], ["ollama_1", "groq_1"])
        self.assertEqual((prim2, fb2), ("ollama_1", "groq_1"))

    def test_local_models_dialog_visibility_for_lmstudio_installed_and_missing(self) -> None:
        from PySide6.QtWidgets import QApplication, QSizePolicy

        app = QApplication.instance() or QApplication([])
        from ui.dialogs import LocalModelsDialog

        installed_entries = [
            {
                "key": "gemma-3-4b-it",
                "label": "Gemma 3 4B Instruct",
                "path": "lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf",
            }
        ]
        with (
            patch("ui.dialogs.is_lmstudio_installed", return_value=True),
            patch("ui.dialogs.list_lmstudio_models", return_value=installed_entries),
        ):
            dlg = LocalModelsDialog()
            try:
                self.assertTrue(dlg.install_button.isHidden())
                self.assertFalse(dlg.detect_button.isHidden())
                self.assertEqual(dlg.detect_button.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed)
                self.assertFalse(dlg.recommended_section.isHidden())
                self.assertFalse(dlg.model_combo.isHidden())
                self.assertFalse(dlg.refresh_button.isHidden())

                dlg._on_lmstudio_models_fetched(installed_entries, "")
                self.assertTrue(dlg.install_button.isHidden())
                self.assertFalse(dlg.download_model_button.isEnabled())
            finally:
                dlg.reject()
                dlg.close()
                app.processEvents()

        with patch("ui.dialogs.is_lmstudio_installed", return_value=False):
            dlg = LocalModelsDialog()
            try:
                self.assertFalse(dlg.install_button.isHidden())
                self.assertTrue(dlg.detect_button.isHidden())
                self.assertTrue(dlg.recommended_section.isHidden())
                self.assertTrue(dlg.model_combo.isHidden())
                self.assertTrue(dlg.refresh_button.isHidden())
            finally:
                dlg.reject()
                dlg.close()
                app.processEvents()

    def test_open_local_models_dialog_and_toggle_remove_ollama(self) -> None:
        from PySide6.QtWidgets import QDialog
        import ui.main_window as main_window_mod

        class _DictCfg:
            def __init__(self, initial: dict) -> None:
                self._data = dict(initial)

            def get(self, key: str, default=None):
                return self._data.get(key, default)

            def set(self, key: str, value) -> None:
                self._data[key] = value

            def update(self, values: dict) -> None:
                self._data.update(values)

        fake_cfg = _DictCfg(
            {
                "ai_items": [
                    {"id": "ollama_1", "provider": "ollama", "title": "Ollama", "model": "gemma3:4b"},
                ],
                "ai_primary_id": "ollama_1",
                "ai_fallback_id": "ollama_1",
                "ai_spellcheck_enabled": False,
            }
        )
        timer = MagicMock()
        dummy = MagicMock()
        dummy._ai_debounce_timers = {"ollama_1": timer}
        dummy._ollama_connections = {"ollama_1": True}
        dummy._ollama_model_pull_active = {"ollama_1": True}
        dummy._ollama_download_widgets = {"ollama_1": {}}
        dummy._ollama_start_in_progress = False
        dummy._ollama_executable = lambda: r"C:\Program Files\Ollama\ollama.exe"
        dummy._cleanup_ollama_item_state = main_window_mod.MainWindow._cleanup_ollama_item_state.__get__(dummy, main_window_mod.MainWindow)
        dummy._prune_ollama_if_local_model_active = main_window_mod.MainWindow._prune_ollama_if_local_model_active.__get__(dummy, main_window_mod.MainWindow)
        dummy._open_local_models_dialog = main_window_mod.MainWindow._open_local_models_dialog.__get__(dummy, main_window_mod.MainWindow)
        dummy._on_ai_spellcheck_toggled = main_window_mod.MainWindow._on_ai_spellcheck_toggled.__get__(dummy, main_window_mod.MainWindow)
        dummy._ensure_ollama_server = main_window_mod.MainWindow._ensure_ollama_server.__get__(dummy, main_window_mod.MainWindow)

        fake_dlg = MagicMock()
        fake_dlg.exec.return_value = QDialog.DialogCode.Accepted
        fake_dlg.server_url = "http://127.0.0.1:1234/v1"
        fake_dlg.api_key = ""
        fake_dlg.selected_model = "gemma-3-4b-it"
        fake_dlg.available_models = ["gemma-3-4b-it"]
        fake_dlg.managed_backend = "lmstudio"
        fake_dlg.model_path = ""
        fake_dlg.backend_title = "LM Studio"
        fake_dlg.use_as_primary = True

        with (
            patch.object(main_window_mod, "config", fake_cfg),
            patch("ui.main_window.LocalModelsDialog", return_value=fake_dlg),
        ):
            dummy._open_local_models_dialog()

        remaining = fake_cfg.get("ai_items")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["provider"], "local_api")
        self.assertEqual(fake_cfg.get("ai_primary_id"), remaining[0]["id"])
        self.assertEqual(fake_cfg.get("ai_fallback_id"), "")
        timer.stop.assert_called_once()
        timer.deleteLater.assert_called_once()
        self.assertEqual(dummy._ollama_connections, {})

        # Enabling AI when both Ollama and LM Studio exist also removes Ollama and skips Ollama server launch.
        fake_cfg2 = _DictCfg(
            {
                "ai_items": [
                    {"id": "ollama_2", "provider": "ollama", "title": "Ollama", "model": "gemma3:4b"},
                    {"id": "lms_2", "provider": "local_api", "managed_backend": "lmstudio", "model": "gemma-3-4b-it"},
                ],
                "ai_primary_id": "ollama_2",
                "ai_fallback_id": "",
                "ai_spellcheck_enabled": False,
            }
        )
        dummy2 = MagicMock()
        dummy2._ai_debounce_timers = {}
        dummy2._ollama_connections = {"ollama_2": False}
        dummy2._ollama_model_pull_active = {}
        dummy2._ollama_download_widgets = {}
        dummy2._ollama_start_in_progress = False
        dummy2._ollama_executable = lambda: r"C:\Program Files\Ollama\ollama.exe"
        dummy2._cleanup_ollama_item_state = main_window_mod.MainWindow._cleanup_ollama_item_state.__get__(dummy2, main_window_mod.MainWindow)
        dummy2._prune_ollama_if_local_model_active = main_window_mod.MainWindow._prune_ollama_if_local_model_active.__get__(dummy2, main_window_mod.MainWindow)
        dummy2._on_ai_spellcheck_toggled = main_window_mod.MainWindow._on_ai_spellcheck_toggled.__get__(dummy2, main_window_mod.MainWindow)
        dummy2._ensure_ollama_server = main_window_mod.MainWindow._ensure_ollama_server.__get__(dummy2, main_window_mod.MainWindow)

        with (
            patch.object(main_window_mod, "config", fake_cfg2),
            patch("ui.main_window.subprocess.Popen") as mock_popen,
        ):
            dummy2._on_ai_spellcheck_toggled(True)

        self.assertEqual([x["id"] for x in fake_cfg2.get("ai_items")], ["lms_2"])
        self.assertEqual(fake_cfg2.get("ai_primary_id"), "lms_2")
        mock_popen.assert_not_called()
        dummy2._start_saved_local_llm_servers.assert_called_once()

    def test_shared_gguf_discovery_and_reuse_between_lmstudio_and_llama_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llama_models = root / "llama_models"
            lms_models = root / "lms_models"
            repo_dir = lms_models / "lmstudio-community" / "gemma-3-4b-it-GGUF"
            repo_dir.mkdir(parents=True)
            (repo_dir / "mmproj-gemma.gguf").write_bytes(b"proj")
            shared_gguf = repo_dir / "gemma-3-4b-it-Q4_K_M.gguf"
            shared_gguf.write_bytes(b"GGUF-SHARED")

            fake_mgr = MagicMock()
            with (
                patch.object(local_llm, "DATA_DIR", root),
                patch.object(local_llm, "LLAMA_MODEL_DIR", llama_models),
                patch.object(local_llm, "get_lmstudio_models_dirs", return_value=[lms_models]),
                patch("core.tts_manager.tts_manager", fake_mgr),
            ):
                found = local_llm.find_local_gguf_model(
                    "gemma-3-4b-it-Q4_K_M.gguf",
                    "lmstudio-community/gemma-3-4b-it-GGUF",
                )
                self.assertEqual(found, shared_gguf)
                self.assertIsNone(local_llm.find_local_gguf_model("mmproj-gemma.gguf"))
                self.assertEqual(local_llm.get_preferred_gguf_browse_dir(), lms_models)

                dest = local_llm.download_recommended_gguf(
                    "https://example.com/gemma-3-4b-it-Q4_K_M.gguf",
                    "gemma-3-4b-it-Q4_K_M.gguf",
                )
                self.assertEqual(dest, shared_gguf)
                fake_mgr._download_url_with_progress.assert_not_called()

            # When model exists in llama.cpp folder first, LM Studio imports it with --copy without redownloading.
            llama_models.mkdir(parents=True, exist_ok=True)
            llama_gguf = llama_models / "Qwen3.5-4B-Q4_K_M.gguf"
            llama_gguf.write_bytes(b"GGUF-LLAMA")
            fake_proc = MagicMock()
            fake_proc.stdin = io.StringIO()
            fake_proc.stdout = io.StringIO("ok\n")
            fake_proc.stderr = io.StringIO("")
            fake_proc.poll.side_effect = [None, 0]
            fake_proc.returncode = 0
            fake_mgr2 = MagicMock()

            with (
                patch.object(local_llm, "DATA_DIR", root),
                patch.object(local_llm, "LLAMA_MODEL_DIR", llama_models),
                patch.object(local_llm, "find_lms_cli", return_value="lms.exe"),
                patch("core.tts_manager.tts_manager", fake_mgr2),
                patch("core.local_llm.subprocess.Popen", return_value=fake_proc) as mock_popen,
                patch("core.local_llm.time.sleep"),
            ):
                repo = local_llm.download_recommended_lmstudio_model(
                    "https://example.com/Qwen3.5-4B-Q4_K_M.gguf",
                    "Qwen3.5-4B-Q4_K_M.gguf",
                    "lmstudio-community/Qwen3.5-4B-GGUF",
                )
            self.assertEqual(repo, "lmstudio-community/Qwen3.5-4B-GGUF")
            fake_mgr2._download_url_with_progress.assert_not_called()
            cmd = mock_popen.call_args.args[0]
            self.assertIn("--copy", cmd)
            self.assertEqual(Path(cmd[2]), llama_gguf)
            self.assertTrue(llama_gguf.is_file())

    def test_local_models_dialog_browse_button_and_llama_reuses_shared_gguf(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from ui.dialogs import LocalModelsDialog

        with tempfile.TemporaryDirectory() as tmp:
            lms_gguf = Path(tmp) / "gemma-3-4b-it-Q4_K_M.gguf"
            lms_gguf.write_bytes(b"GGUF")

            def _fake_find(filename: str, _repo: str = ""):
                return lms_gguf if filename == "gemma-3-4b-it-Q4_K_M.gguf" else None

            with (
                patch("ui.dialogs.is_lmstudio_installed", return_value=True),
                patch("ui.dialogs.list_lmstudio_models", return_value=[]),
                patch("ui.dialogs.find_local_gguf_model", side_effect=_fake_find),
                patch("ui.dialogs.QDesktopServices.openUrl", return_value=False) as mock_open_url,
                patch("ui.dialogs.webbrowser.open") as mock_wb_open,
            ):
                dlg = LocalModelsDialog()
                try:
                    self.assertIn("padding: 0px", dlg.browse_models_button.styleSheet())
                    self.assertEqual(dlg.browse_models_button.size().width(), 34)
                    self.assertEqual(dlg.browse_models_button.size().height(), 34)
                    dlg.browse_models_button.click()
                    self.assertEqual(mock_open_url.call_args.args[0].toString(), "https://lmstudio.ai/models")
                    mock_wb_open.assert_called_with("https://lmstudio.ai/models")

                    llama_idx = dlg.backend_combo.findData("llama_cpp")
                    dlg.backend_combo.setCurrentIndex(llama_idx)
                    gemma_idx = dlg.recommended_model_combo.findData("gemma3_4b")
                    dlg.recommended_model_combo.setCurrentIndex(gemma_idx)
                    app.processEvents()

                    self.assertEqual(dlg.model_file_edit.text(), str(lms_gguf))
                    self.assertTrue(dlg.refresh_button.isEnabled())
                    self.assertFalse(dlg.download_model_button.isEnabled())

                    dlg.browse_models_button.click()
                    self.assertIn("huggingface.co/models", mock_wb_open.call_args.args[0])
                finally:
                    dlg.reject()
                    dlg.close()
                    app.processEvents()


if __name__ == "__main__":
    unittest.main()
