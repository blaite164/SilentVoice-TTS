"""
core/obs_animations.py — Каталог анімацій появи та руху для OBS-субтитрів
(витягнуто з Test з урахуванням наявних налаштувань тексту та блоку).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

ENTRY_ANIMATIONS: List[Dict[str, Any]] = [
    {
        "id": "typewriter",
        "i18n_key": "ui_339",
        "name": "Друкарська машинка (синхронно з голосом)",
        "category": "Друкарські",
        "description": "Поступовий набір по літерах синхронно з озвучкою.",
        "supports_split": False,
        "defaults": {},
        "controls": [],
    },
    {
        "id": "fade-in",
        "i18n_key": "obs.anim.fade_in",
        "name": "Плавна поява (Fade In)",
        "category": "Плавні",
        "description": "Класична м'яка поява тексту через прозорість.",
        "supports_split": True,
        "defaults": {"duration": 1000, "easing": "ease-out", "split": "words", "stagger": 200},
        "split_defaults": {
            "whole": {"duration": 700, "stagger": 45},
            "words": {"duration": 1000, "stagger": 200},
            "chars": {"duration": 700, "stagger": 45},
        },
        "controls": [],
    },
    {
        "id": "blur-focus",
        "i18n_key": "obs.anim.blur_focus",
        "name": "Фокус з розмиття (Blur Focus)",
        "category": "Плавні",
        "description": "Текст проявляється з м'якого оптичного розмиття.",
        "supports_split": True,
        "defaults": {
            "duration": 850,
            "easing": "cubic-bezier(0.22, 1, 0.36, 1)",
            "split": "words",
            "stagger": 65,
            "blur": 16,
            "scaleFrom": 1.08,
        },
        "controls": [
            {"key": "blur", "i18n": "obs.ctrl.blur", "label": "Сила розмиття", "type": "range", "min": 4, "max": 32, "step": 2, "unit": "px"},
            {"key": "scaleFrom", "i18n": "obs.ctrl.scale_from", "label": "Початковий масштаб", "type": "range", "min": 0.8, "max": 1.4, "step": 0.02, "unit": "x"},
        ],
    },
    {
        "id": "zoom",
        "i18n_key": "obs.anim.zoom",
        "name": "Масштабування (Zoom)",
        "category": "Масштаб",
        "description": "Поява тексту зі збільшенням або зменшенням масштабу.",
        "supports_split": True,
        "defaults": {
            "duration": 680,
            "easing": "cubic-bezier(0.22, 1, 0.36, 1)",
            "split": "whole",
            "stagger": 40,
            "zoomType": "in",
            "scaleAmount": 0.45,
        },
        "controls": [
            {
                "key": "zoomType",
                "i18n": "obs.ctrl.zoom_type",
                "label": "Режим масштабу",
                "type": "select",
                "options": [
                    {"value": "in", "i18n": "obs.opt.zoom_in", "label": "Збільшення (Zoom In)"},
                    {"value": "out", "i18n": "obs.opt.zoom_out", "label": "Зменшення (Zoom Out)"},
                ],
            },
            {"key": "scaleAmount", "i18n": "obs.ctrl.scale_amount", "label": "Сила масштабу", "type": "range", "min": 0.1, "max": 0.9, "step": 0.05, "unit": ""},
        ],
    },
    {
        "id": "elastic-pop",
        "i18n_key": "obs.anim.elastic_pop",
        "name": "Пружинний вибух (Elastic Pop)",
        "category": "Масштаб",
        "description": "Енергійний пружний стрибок із легким перельотом.",
        "supports_split": True,
        "defaults": {
            "duration": 850,
            "easing": "cubic-bezier(0.34, 1.56, 0.64, 1)",
            "split": "chars",
            "stagger": 30,
            "overshoot": 1.22,
        },
        "controls": [
            {"key": "overshoot", "i18n": "obs.ctrl.overshoot", "label": "Сила відскоку", "type": "range", "min": 1.05, "max": 1.5, "step": 0.03, "unit": "x"},
        ],
    },
    {
        "id": "flip-y",
        "i18n_key": "obs.anim.flip_y",
        "name": "3D Переворот по Y (Flip Y)",
        "category": "3D та Обертання",
        "description": "Об'ємний поворот по вертикальній осі.",
        "supports_split": True,
        "defaults": {
            "duration": 800,
            "easing": "cubic-bezier(0.22, 1, 0.36, 1)",
            "split": "chars",
            "stagger": 35,
            "angle": 90,
        },
        "controls": [
            {"key": "angle", "i18n": "obs.ctrl.angle", "label": "Початковий кут", "type": "range", "min": 30, "max": 180, "step": 10, "unit": "°"},
        ],
    },
    {
        "id": "wipe-mask",
        "i18n_key": "obs.anim.wipe_mask",
        "name": "Розкриття маскою (Wipe)",
        "category": "Маски",
        "description": "Геометричне відкриття тексту з центру або по діагоналі.",
        "supports_split": False,
        "defaults": {
            "duration": 800,
            "easing": "cubic-bezier(0.22, 1, 0.36, 1)",
            "split": "whole",
            "stagger": 0,
            "wipeStyle": "center",
            "scaleFrom": 0.94,
        },
        "controls": [
            {
                "key": "wipeStyle",
                "i18n": "obs.ctrl.wipe_style",
                "label": "Форма маски",
                "type": "select",
                "options": [
                    {"value": "center", "i18n": "obs.opt.wipe_center", "label": "З центру вбоки"},
                    {"value": "diagonal", "i18n": "obs.opt.wipe_diagonal", "label": "Діагональний зріз"},
                ],
            },
            {"key": "scaleFrom", "i18n": "obs.ctrl.scale_from", "label": "Початковий масштаб", "type": "range", "min": 0.7, "max": 1.0, "step": 0.02, "unit": "x"},
        ],
    },
    {
        "id": "mask-rise",
        "i18n_key": "obs.anim.mask_rise",
        "name": "Підйом з-під маски (Editorial Rise)",
        "category": "Маски",
        "description": "Слова виринають знизу з невидимої лінії.",
        "supports_split": True,
        "defaults": {
            "duration": 750,
            "easing": "cubic-bezier(0.22, 1, 0.36, 1)",
            "split": "words",
            "stagger": 70,
            "distance": 105,
        },
        "controls": [
            {"key": "distance", "i18n": "obs.ctrl.mask_distance", "label": "Глибина маски (%)", "type": "range", "min": 40, "max": 140, "step": 5, "unit": "%"},
        ],
    },
    {
        "id": "wave-stagger",
        "i18n_key": "obs.anim.wave_stagger",
        "name": "Хвиля літер (Wave Stagger)",
        "category": "Друкарські",
        "description": "Кожна літера підстрибує хвилею одна за одною.",
        "supports_split": True,
        "defaults": {
            "duration": 700,
            "easing": "cubic-bezier(0.34, 1.56, 0.64, 1)",
            "split": "chars",
            "stagger": 35,
            "waveHeight": 28,
        },
        "controls": [
            {"key": "waveHeight", "i18n": "obs.ctrl.wave_amp", "label": "Амплітуда хвилі", "type": "range", "min": 10, "max": 60, "step": 2, "unit": "px"},
        ],
    },
    {
        "id": "skew-slide",
        "i18n_key": "obs.anim.skew_slide",
        "name": "Швидкісний нахил (Skew Slide)",
        "category": "Креативні",
        "description": "Динамічний вліт із кінетичним нахилом.",
        "supports_split": True,
        "defaults": {
            "duration": 700,
            "easing": "cubic-bezier(0.22, 1, 0.36, 1)",
            "split": "whole",
            "stagger": 45,
            "distance": 70,
            "skewDeg": 22,
        },
        "controls": [
            {"key": "distance", "i18n": "obs.ctrl.distance", "label": "Дистанція", "type": "range", "min": 20, "max": 140, "step": 5, "unit": "px"},
            {"key": "skewDeg", "i18n": "obs.ctrl.skew_deg", "label": "Кут нахилу", "type": "range", "min": 5, "max": 45, "step": 2, "unit": "°"},
        ],
    },
    {
        "id": "glitch-reveal",
        "i18n_key": "obs.anim.glitch_reveal",
        "name": "Глітч-поява (Glitch Reveal)",
        "category": "Креативні",
        "description": "Цифровий збій зі зрізами, RGB-аберацією та кіберпанк-спалахами.",
        "supports_split": True,
        "defaults": {
            "duration": 700,
            "easing": "linear",
            "split": "whole",
            "stagger": 35,
            "glitchPalette": "cyber",
            "intensity": 5,
        },
        "controls": [
            {
                "key": "glitchPalette",
                "i18n": "obs.ctrl.glitch_palette",
                "label": "Колірна схема",
                "type": "select",
                "options": [
                    {"value": "cyber", "i18n": "obs.opt.glitch_cyber", "label": "Кіберпанк (Cyan / Pink)"},
                    {"value": "rgb", "i18n": "obs.opt.glitch_rgb", "label": "Класичний RGB"},
                    {"value": "matrix", "i18n": "obs.opt.glitch_matrix", "label": "Матриця (Зелений)"},
                ],
            },
            {"key": "intensity", "i18n": "obs.ctrl.intensity", "label": "Інтенсивність зсуву", "type": "range", "min": 2, "max": 14, "step": 1, "unit": "px"},
        ],
    },
    {
        "id": "neon-flicker",
        "i18n_key": "obs.anim.neon_flicker",
        "name": "Неонове ввімкнення (Neon Flicker)",
        "category": "Креативні",
        "description": "Спалахування неонової вивіски з м'яким світінням.",
        "supports_split": True,
        "defaults": {
            "duration": 1000,
            "easing": "ease-out",
            "split": "whole",
            "stagger": 40,
            "glowRadius": 22,
        },
        "controls": [
            {"key": "glowRadius", "i18n": "obs.ctrl.glow_radius", "label": "Радіус світіння", "type": "range", "min": 6, "max": 45, "step": 2, "unit": "px"},
        ],
    },
]

MOTION_ANIMATIONS: List[Dict[str, Any]] = [
    {
        "id": "none",
        "i18n_key": "ui_346",
        "name": "Без руху (статичний)",
        "category": "Статика",
        "description": "Текст залишається нерухомим після появи.",
        "supports_split": False,
        "defaults": {},
        "controls": [],
    },
    {
        "id": "levitation",
        "i18n_key": "obs.motion.levitation",
        "name": "Левітація (Levitation)",
        "category": "Простір",
        "description": "Плавне невагоме ширяння тексту вгору та вниз.",
        "supports_split": True,
        "defaults": {"duration": 1800, "split": "whole", "stagger": 120, "easing": "ease-in-out", "amplitude": 14, "trigger": "after"},
        "controls": [
            {"key": "amplitude", "i18n": "obs.ctrl.amplitude", "label": "Амплітуда ширяння", "type": "range", "min": 4, "max": 32, "step": 2, "unit": "px"},
        ],
    },
    {
        "id": "rainbow-flow",
        "i18n_key": "obs.motion.rainbow",
        "name": "Веселковий перелив (Rainbow)",
        "category": "Колір",
        "description": "Плавний рухомий веселковий градієнт по тексту.",
        "supports_split": False,
        "defaults": {"duration": 3600, "split": "whole", "stagger": 0, "easing": "linear", "trigger": "simultaneous"},
        "controls": [],
    },
    {
        "id": "shimmer",
        "i18n_key": "obs.motion.shimmer",
        "name": "Світловий відблиск (Shimmer)",
        "category": "Колір",
        "description": "Яскравий промінь світла, що пробігає по літерах.",
        "supports_split": False,
        "defaults": {"duration": 2200, "split": "whole", "stagger": 0, "easing": "linear", "trigger": "simultaneous"},
        "controls": [],
    },
    {
        "id": "pulse-breathe",
        "i18n_key": "obs.motion.pulse",
        "name": "Пульсація (Pulse)",
        "category": "Ритм",
        "description": "Ритмічне плавне збільшення та зменшення масштабу.",
        "supports_split": True,
        "defaults": {"duration": 1100, "split": "whole", "stagger": 70, "easing": "ease-in-out", "scaleMax": 1.08, "trigger": "after"},
        "controls": [
            {"key": "scaleMax", "i18n": "obs.ctrl.scale_max", "label": "Піковий масштаб", "type": "range", "min": 1.02, "max": 1.25, "step": 0.01, "unit": "x"},
        ],
    },
    {
        "id": "wave-loop",
        "i18n_key": "obs.motion.wave_loop",
        "name": "Нескінченна хвиля (Wave Loop)",
        "category": "Ритм",
        "description": "Плавна синусоїдна хвиля, що біжить по літерах або словах.",
        "supports_split": True,
        "defaults": {"duration": 850, "split": "chars", "stagger": 75, "easing": "ease-in-out", "waveHeight": 10, "trigger": "after"},
        "controls": [
            {"key": "waveHeight", "i18n": "obs.ctrl.wave_height", "label": "Висота хвилі", "type": "range", "min": 4, "max": 30, "step": 2, "unit": "px"},
        ],
    },
    {
        "id": "neon-breathe",
        "i18n_key": "obs.motion.neon_breathe",
        "name": "Неонове дихання (Neon Breathe)",
        "category": "Колір",
        "description": "М'яке пульсуюче світіння навколо тексту.",
        "supports_split": True,
        "defaults": {"duration": 1400, "split": "whole", "stagger": 70, "easing": "ease-in-out", "glowMax": 28, "trigger": "after"},
        "controls": [
            {"key": "glowMax", "i18n": "obs.ctrl.glow_max", "label": "Макс. світіння", "type": "range", "min": 10, "max": 50, "step": 2, "unit": "px"},
        ],
    },
    {
        "id": "micro-jitter",
        "i18n_key": "obs.motion.jitter",
        "name": "Енергійна вібрація (Jitter)",
        "category": "Ритм",
        "description": "Легке швидке тремтіння для живих акцентів.",
        "supports_split": True,
        "defaults": {"duration": 900, "split": "chars", "stagger": 35, "jitterPx": 2, "trigger": "after"},
        "controls": [
            {"key": "jitterPx", "i18n": "obs.ctrl.jitter_px", "label": "Сила вібрації", "type": "range", "min": 1, "max": 6, "step": 1, "unit": "px"},
        ],
    },
]

ENTRY_ANIM_MAP: Dict[str, Dict[str, Any]] = {a["id"]: a for a in ENTRY_ANIMATIONS}
MOTION_ANIM_MAP: Dict[str, Dict[str, Any]] = {m["id"]: m for m in MOTION_ANIMATIONS}


def get_entry_spec(anim_id: str) -> Dict[str, Any]:
    return ENTRY_ANIM_MAP.get(anim_id, ENTRY_ANIMATIONS[0])


def get_motion_spec(motion_id: str) -> Dict[str, Any]:
    return MOTION_ANIM_MAP.get(motion_id, MOTION_ANIMATIONS[0])


def get_variant_key(kind: str, spec: Dict[str, Any], params: Dict[str, Any]) -> str:
    """Формує ключ профілю налаштувань для обраного режиму (split + select-варіант)."""
    parts: List[str] = []
    if spec.get("supports_split"):
        parts.append(str(params.get("split") or spec.get("defaults", {}).get("split") or "whole"))
    if kind == "motion" and spec.get("id") != "none":
        parts.append(str(params.get("trigger") or spec.get("defaults", {}).get("trigger") or "after"))
    for ctrl in spec.get("controls", []):
        if ctrl.get("type") == "select":
            k = str(ctrl["key"])
            parts.append(str(params.get(k) or spec.get("defaults", {}).get(k) or ""))
    return "|".join(parts) if parts else "default"


def get_variant_defaults(kind: str, spec: Dict[str, Any], base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Повертає значення режиму з каталогу Test або загальні значення анімації."""
    cur_split = str(base_params.get("split") or spec.get("defaults", {}).get("split") or "whole")
    split_defaults = spec.get("split_defaults", {}).get(cur_split, {})
    defaults = copy.deepcopy(spec.get("defaults", {}))
    defaults.update(copy.deepcopy(split_defaults))
    for k in ("split", "trigger"):
        if k in base_params:
            defaults[k] = base_params[k]
    for ctrl in spec.get("controls", []):
        if ctrl.get("type") == "select" and ctrl["key"] in base_params:
            defaults[ctrl["key"]] = base_params[ctrl["key"]]
    return defaults


def resolve_anim_params(kind: str, anim_id: str, stored_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = get_entry_spec(anim_id) if kind == "entry" else get_motion_spec(anim_id)
    params = copy.deepcopy(spec.get("defaults", {}))
    if not isinstance(stored_map, dict):
        return params
    custom = stored_map.get(spec["id"])
    if not isinstance(custom, dict):
        return params

    # Зчитуємо обрані режими (split, trigger, select-контроли)
    for k in ("split", "trigger"):
        if k in custom:
            params[k] = custom[k]
    for ctrl in spec.get("controls", []):
        if ctrl.get("type") == "select" and ctrl["key"] in custom:
            params[ctrl["key"]] = custom[ctrl["key"]]

    v_key = get_variant_key(kind, spec, params)
    params = get_variant_defaults(kind, spec, params)

    by_variant = custom.get("by_variant")
    if isinstance(by_variant, dict) and isinstance(by_variant.get(v_key), dict):
        for k, v in by_variant[v_key].items():
            if k in params or k in ("duration", "split", "stagger", "trigger"):
                params[k] = v
    else:
        for k, v in custom.items():
            if k != "by_variant" and (k in params or k in ("duration", "split", "stagger", "trigger")):
                params[k] = v
    return params
