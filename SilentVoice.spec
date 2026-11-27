# -*- mode: python ; coding: utf-8 -*-
"""
SilentVoice.spec — Оптимізована конфігурація PyInstaller (onedir) для SilentVoice TTS.
Видаляє CUDA/NVIDIA DLL, невикористовувані модулі PySide6 (WebEngine/QML/Quick/3D/Pdf/Charts),
кешовані моделі ruaccent (661 МБ), сторонні пакети (PyQt5/6, cv2, matplotlib, silero-pip, тощо),
а також dev/test/header/lib артефакти для меншого інсталятора та швидшого старту.
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve()

# Базові дані додатку (без персонального data/config.json, data/crash.log та __pycache__)
app_datas = [
    (str(ROOT / "assets" / "icon.ico"), "assets"),
    (str(ROOT / "assets" / "edit.svg"), "assets"),
    (str(ROOT / "assets" / "delete.svg"), "assets"),
    (str(ROOT / "assets" / "easter_egg.ogg"), "assets"),
    (str(ROOT / "locales" / "*.json"), "locales"),
    (str(ROOT / "data" / "samples"), "data/samples"),
]

# Необхідні дані бібліотек
for pkg in (
    "certifi",
    "piper",
    "ukrainian_word_stress",
    "ukrainian_accentor",
    "ipa_uk",
    "num2words",
    "razdel",
    "holos_onnx",
    "TeraTTS",
):
    app_datas += collect_data_files(pkg, include_py_files=False)

# transliterate шукає мовні пакети через os.listdir(transliterate/contrib/languages)
app_datas += collect_data_files("transliterate", include_py_files=True)

# Метадані пакетів, які перевіряються через importlib.metadata
for meta_pkg in (
    "numpy",
    "scipy",
    "torch",
    "torchaudio",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "safetensors",
    "tqdm",
    "packaging",
    "regex",
    "requests",
    "filelock",
    "aiohttp",
    "edge-tts",
    "piper-tts",
    "holos-tts-onnx",
    "noisereduce",
    "sounddevice",
    "soundfile",
):
    try:
        app_datas += copy_metadata(meta_pkg)
    except Exception:
        pass

# Динамічні бібліотеки звуку та ONNX CPU
app_binaries = []
for bin_pkg in ("sounddevice", "_sounddevice_data", "soundfile", "_soundfile_data", "piper"):
    try:
        app_binaries += collect_dynamic_libs(bin_pkg)
    except Exception:
        pass

hidden_imports = [
    # Потрібно для рендерингу SVG-іконок інтерфейсу (assets/edit.svg, assets/delete.svg) через QIcon
    "PySide6.QtSvg",
    # Мережа та онлайн-провайдери
    "aiohttp",
    "aiohttp_socks",
    "python_socks",
    "python_socks.async_.asyncio",
    "socks",
    "edge_tts",
    "telethon",
    "telethon.sessions",
    "telethon.tl.types",
    "telethon.tl.functions",
    "core.secret_store",
    "qrcode",
    # Локальні TTS та обробка тексту
    "piper",
    "piper.voice",
    "piper.phonemize_espeak",
    "onnxruntime",
    "core.llama_tts",
    "core.omnivoice_tts",
    "core.omnivoice_tags",
    "core.kokoro_onnx_tts",
    "ui.reaction_commands",
    "holos_onnx",
    "ipa_uk",
    "ukrainian_word_stress",
    "TeraTTS",
    "TeraTTS.infer_onnx",
    "TeraTTS.tokenizer",
    "ruaccent",
    "ruaccent.ruaccent",
    "ruaccent.koziev",
    "ruaccent.koziev.rupostagger.rupostagger",
    "ruaccent.koziev.rulemma.rulemma",
    "razdel",
    "num2words",
    "transliterate",
    "transliterate.contrib.languages.ru",
    "transliterate.contrib.languages.ru.translit_language_pack",
    # DSP / ML
    "noisereduce",
    "noisereduce.spectralgate",
    "noisereduce.spectralgate.stationary",
    "scipy.signal",
    "scipy.ndimage",
    "scipy.io",
    "scipy.io.wavfile",
    "scipy._cyutility",
    # Потрібні моделі transformers (Holos Albert + RUAccent AutoTokenizer)
    "transformers.models.albert",
    "transformers.models.albert.modeling_albert",
    "transformers.models.albert.configuration_albert",
    "transformers.models.bert",
    "transformers.models.roberta",
    "transformers.models.electra",
    "transformers.models.deberta",
    "transformers.models.deberta_v2",
    "transformers.models.xlm_roberta",
    "transformers.models.auto",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.modeling_auto",
]
hidden_imports += collect_submodules("ruaccent")
hidden_imports += collect_submodules("holos_onnx")
hidden_imports += collect_submodules("transliterate.contrib.languages.ru")

excludes = [
    # Інші GUI-фреймворки та системні інструменти
    "PyQt5",
    "PyQt6",
    "tkinter",
    "_tkinter",
    "customtkinter",
    "tkinterdnd2",
    # Невикористовувані модулі PySide6 (додаток використовує лише QtCore, QtGui, QtWidgets)
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DAnimation",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetwork",
    "PySide6.QtNetworkAuth",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtPrintSupport",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialBus",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtSvgWidgets",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtUiTools",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtHttpServer",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtDBus",
    "PySide6.QtXml",
    "PySide6.QtConcurrent",
    # CUDA / GPU та сторонні важкі бібліотеки
    "nvidia",
    "torchvision",
    "silero",
    "cv2",
    "matplotlib",
    "mpl_toolkits",
    "noisereduce.plotting",
    "librosa.display",
    "PIL",
    "easyocr",
    "faster_whisper",
    "ctranslate2",
    "av",
    "yt_dlp",
    "spotdl",
    "spotipy",
    "twitchio",
    "rapidocr_onnxruntime",
    "pytesseract",
    "shapely",
    "pyclipper",
    "uvicorn",
    "fastapi",
    "starlette",
    "openai",
    "groq",
    "ddgs",
    "duckduckgo_search",
    "pykakasi",
    "jaconv",
    # Зайві важкі модулі (librosa замінено легким runtime-шимом на scipy.signal)
    "librosa",
    "numba",
    "llvmlite",
    "sklearn",
    "piper.train",
    "hf_xet",
    "lxml",
    "tiktoken",
    "sentencepiece",
    "pydantic",
    "pydantic_core",
    "typer",
    "tensorboard",
    "torch.utils.tensorboard",
    "torch.utils.benchmark",
    # Dev / test / packaging інструменти
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "doctest",
    "pip",
    "setuptools",
    "wheel",
    "pyinstaller",
    "pefile",
    "altgraph",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=app_binaries,
    datas=app_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "installer" / "pyi_rth_librosa_shim.py")],
    excludes=excludes,
    # Зберігаємо Python-модулі у PYZ-архіві замість тисяч окремих .pyc.
    # Це помітно зменшує кількість операцій читання файлів під час кожного запуску.
    noarchive=False,
    optimize=1,
)

# ----------------------------------------------------------------------
# Жорстка постобробка: видаляємо непотрібні бінарники, кешовані моделі та dev-файли
# ----------------------------------------------------------------------
_ALLOWED_PYSIDE6_DLLS = {
    "qt6core.dll",
    "qt6gui.dll",
    "qt6svg.dll",
    "qt6widgets.dll",
    "pyside6.abi3.dll",
    "shiboken6.abi3.dll",
}

_ALLOWED_QT_PLUGINS = (
    "platforms\\qwindows.dll",
    "platforms/qwindows.dll",
    "styles\\qmodernwindowsstyle.dll",
    "styles/qmodernwindowsstyle.dll",
    "imageformats\\qico.dll",
    "imageformats/qico.dll",
    "imageformats\\qsvg.dll",
    "imageformats/qsvg.dll",
    "iconengines\\qsvgicon.dll",
    "iconengines/qsvgicon.dll",
)

def _is_unwanted_entry(dest: str, src: str) -> bool:
    d = dest.replace("/", "\\").lower()
    s = src.replace("/", "\\").lower()

    # Ніколи не видаляємо Python-байткод модулів, якщо він потрапив до a.datas.
    if d.endswith(".pyc"):
        return False

    # 1. CUDA / TensorRT / NVIDIA GPU бібліотеки (не чіпаємо .py/.pyc модулі на кшталт torch.backends.cudnn)
    if "\\nvidia\\" in d or (
        not d.endswith((".py", ".pyc"))
        and any(tok in d for tok in (
            "onnxruntime_providers_cuda",
            "onnxruntime_providers_tensorrt",
            "cublas", "cudnn", "cufft", "curand", "cusparse", "cudart", "nvrtc", "nvjitlink", "nccl",
        ))
    ):
        return True

    # 2. Зайві Qt6 DLL / WebEngine / QML / OpenGL software / переклади Qt
    if d.startswith("pyside6\\"):
        if "\\translations\\" in d or "\\qml\\" in d or "\\resources\\" in d:
            return True
        if "\\plugins\\" in d:
            if not any(d.endswith(p.replace("/", "\\")) for p in _ALLOWED_QT_PLUGINS):
                return True
        elif d.endswith(".dll"):
            dll_name = d.rsplit("\\", 1)[-1]
            if dll_name not in _ALLOWED_PYSIDE6_DLLS and not dll_name.startswith(("vcruntime", "msvcp", "concrt")):
                return True

    # 3. Зайві файли PyTorch / Numba / Sklearn (заголовки, .lib, cmake, protoc, тести — без видалення Python-модулів .pyc/.py)
    if d.endswith((".lib", ".a", ".pdb", ".h", ".hpp", ".c", ".cpp", ".cu", ".pyx", ".pxd", ".pxi", ".cmake", ".pyi")):
        return True
    if "\\include\\" in d:
        return True
    if not d.endswith((".py", ".pyc")) and (
        "\\test\\" in d or "\\tests\\" in d or ("\\testing\\" in d and not d.startswith("torch\\"))
    ):
        return True
    if d.endswith("protoc.exe"):
        return True

    # 4. Кешовані моделі ruaccent із site-packages (завантажуються при потребі в %APPDATA%)
    if d.startswith("ruaccent\\"):
        if any(seg in d for seg in (
            "\\nn\\",
            "\\dictionary\\",
            "\\koziev\\rupostagger\\database\\",
            "ruword2tags.dat",
            "rulemma.dat",
        )):
            return True

    # 5. Невикористовувані моделі івриту/арабської в Piper (25 МБ) та тренувальний код
    if d.startswith("piper\\") and (
        "\\hebrew\\nakdimon.onnx" in d
        or "\\tashkeel\\model.onnx" in d
        or "\\train\\" in d
    ):
        return True

    # 6. Важкі підмодулі SciPy та нецільові бінарники PortAudio (32bit/ARM64/macOS)
    if d.startswith("scipy\\") and (
        "\\io\\_fast_matrix_market\\" in d
        or "\\datasets\\" in d
    ):
        return True
    if d.startswith("_sounddevice_data\\") and any(
        tok in d for tok in ("32bit", "arm64", ".dylib")
    ):
        return True

    # 7. Кеш-логи або випадкові файли проєкту
    if d.endswith(("crash.log", "config.json")) and "data\\" in d and "samples" not in d:
        return True

    return False

a.binaries = [x for x in a.binaries if not _is_unwanted_entry(x[0], x[1])]
a.datas = [x for x in a.datas if not _is_unwanted_entry(x[0], x[1])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SilentVoice",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico"),
    version=str(ROOT / "installer" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SilentVoice",
)
