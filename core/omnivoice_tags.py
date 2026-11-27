"""
core/omnivoice_tags.py — Універсальний модуль звукових команд і невербальних реакцій (`/`, `\\`, `|`).

Підтримує:
- Вбудовані реакції OmniVoice (`[laughter]`, `[sigh]` тощо), що працюють нативно при увімкненій інтеграції з OmniVoice.
- Власні звукові команди користувача (`\\хаха` + прикріплений `.wav`/`.mp3`/`.ogg`/`.flac`), які працюють
  усередині тексту з БУДЬ-ЯКИМ TTS-рушієм (Edge, Piper, ZONOS3, Qwen3, OmniVoice, ElevenLabs тощо).
- Кастомне відображення (емодзі/символ або автоматично `[команда]`), атомарне видалення Backspace та
  розбиття фрази на сегменти тексту й аудіофайлів.

Навмисно не імпортує важкі бібліотеки (`numpy`, `soundfile`, `win_api`), щоб безпечно використовуватись
у хуку клавіатури та оверлеї без затримки старту.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class OmniVoiceTagSpec:
    tag: str
    emoji: str
    primary_cmds: Dict[str, str]
    descriptions: Dict[str, str]
    aliases: Tuple[str, ...]
    id: str = ""
    audio_file: str = ""
    omnivoice_tag: str = ""
    enabled: bool = True
    custom_cmd: str = ""
    custom_desc: str = ""


# 8 стандартних виразних реакцій (відповідають офіційним тегам OmniVoice)
OMNIVOICE_REACTION_TAGS: Tuple[OmniVoiceTagSpec, ...] = (
    OmniVoiceTagSpec(
        id="rc_laughter",
        tag="[laughter]",
        omnivoice_tag="[laughter]",
        emoji="😂",
        primary_cmds={
            "uk": "сміх",
            "ru": "смех",
            "en": "laugh",
            "de": "lach",
            "es": "risa",
            "fr": "rire",
            "pl": "śmiech",
        },
        descriptions={
            "uk": "Живий сміх",
            "ru": "Живой смех",
            "en": "Natural laughter",
            "de": "Natürliches Lachen",
            "es": "Risa natural",
            "fr": "Rire naturel",
            "pl": "Naturalny śmiech",
        },
        aliases=(
            "сміх", "смех", "ха", "хаха", "лол",
            "lol", "laugh", "laughter", "ha", "haha",
            "lach", "risa", "rire", "śmiech", "smiech",
        ),
    ),
    OmniVoiceTagSpec(
        id="rc_sigh",
        tag="[sigh]",
        omnivoice_tag="[sigh]",
        emoji="😮‍💨",
        primary_cmds={
            "uk": "зітх",
            "ru": "вздох",
            "en": "sigh",
            "de": "seufz",
            "es": "susp",
            "fr": "soupir",
            "pl": "west",
        },
        descriptions={
            "uk": "Втомлене зітхання",
            "ru": "Усталый вздох",
            "en": "Tired sigh",
            "de": "Müder Seufzer",
            "es": "Suspiro cansado",
            "fr": "Soupir fatigué",
            "pl": "Zmęczone westchnienie",
        },
        aliases=(
            "зітх", "зітхання", "вздох", "ех", "эх",
            "sigh", "seufz", "susp", "soupir", "west",
        ),
    ),
    OmniVoiceTagSpec(
        id="rc_surprise_wa",
        tag="[surprise-wa]",
        omnivoice_tag="[surprise-wa]",
        emoji="😮",
        primary_cmds={
            "uk": "вау",
            "ru": "вау",
            "en": "wow",
            "de": "wow",
            "es": "guau",
            "fr": "waouh",
            "pl": "wow",
        },
        descriptions={
            "uk": "Здивування «вау!»",
            "ru": "Удивление «вау!»",
            "en": "Surprised «whoa!»",
            "de": "Überraschtes «Whoa!»",
            "es": "Sorpresa «¡guau!»",
            "fr": "Surprise « waouh ! »",
            "pl": "Zaskoczenie «wow!»",
        },
        aliases=("вау", "ого", "wow", "wa", "waouh", "guau"),
    ),
    OmniVoiceTagSpec(
        id="rc_surprise_oh",
        tag="[surprise-oh]",
        omnivoice_tag="[surprise-oh]",
        emoji="😯",
        primary_cmds={
            "uk": "ох",
            "ru": "ох",
            "en": "oh",
            "de": "oh",
            "es": "oh",
            "fr": "oh",
            "pl": "och",
        },
        descriptions={
            "uk": "М’яке здивування «ох»",
            "ru": "Мягкое удивление «ох»",
            "en": "Soft «oh!»",
            "de": "Sanftes «Oh!»",
            "es": "Suave «¡oh!»",
            "fr": "Doux « oh ! »",
            "pl": "Delikatne «och!»",
        },
        aliases=("ох", "оу", "oh", "och"),
    ),
    OmniVoiceTagSpec(
        id="rc_surprise_ah",
        tag="[surprise-ah]",
        omnivoice_tag="[surprise-ah]",
        emoji="😲",
        primary_cmds={
            "uk": "ах",
            "ru": "ах",
            "en": "ah",
            "de": "ah",
            "es": "ah",
            "fr": "ah",
            "pl": "ach",
        },
        descriptions={
            "uk": "Різкий вигук «ах!»",
            "ru": "Резкий возглас «ах!»",
            "en": "Sudden «ah!»",
            "de": "Plötzliches «Ah!»",
            "es": "Repentino «¡ah!»",
            "fr": "Soudain « ah ! »",
            "pl": "Nagłe «ach!»",
        },
        aliases=("ах", "аха", "ah", "ach"),
    ),
    OmniVoiceTagSpec(
        id="rc_question_en",
        tag="[question-en]",
        omnivoice_tag="[question-en]",
        emoji="🤔",
        primary_cmds={
            "uk": "хм",
            "ru": "хм",
            "en": "hmm",
            "de": "hmm",
            "es": "hmm",
            "fr": "hmm",
            "pl": "hmm",
        },
        descriptions={
            "uk": "Задумливе «хм?»",
            "ru": "Задумчивое «хм?»",
            "en": "Thoughtful «hmm?»",
            "de": "Nachdenkliches «Hmm?»",
            "es": "Pensativo «¿hmm?»",
            "fr": "Pensif « hmm ? »",
            "pl": "Zamyślone «hmm?»",
        },
        aliases=("хм", "мм", "hm", "hmm"),
    ),
    OmniVoiceTagSpec(
        id="rc_confirmation_en",
        tag="[confirmation-en]",
        omnivoice_tag="[confirmation-en]",
        emoji="👍",
        primary_cmds={
            "uk": "угу",
            "ru": "угу",
            "en": "mhm",
            "de": "mhm",
            "es": "aja",
            "fr": "mhm",
            "pl": "mhm",
        },
        descriptions={
            "uk": "Погодження «угу»",
            "ru": "Согласие «угу»",
            "en": "Agreement «mhm»",
            "de": "Zustimmung «Mhm»",
            "es": "Acuerdo «ajá»",
            "fr": "Accord « mhm »",
            "pl": "Potwierdzenie «mhm»",
        },
        aliases=("угу", "мгм", "ага", "mhm", "uhhuh", "aja"),
    ),
    OmniVoiceTagSpec(
        id="rc_dissatisfaction_hnn",
        tag="[dissatisfaction-hnn]",
        omnivoice_tag="[dissatisfaction-hnn]",
        emoji="😤",
        primary_cmds={
            "uk": "хмф",
            "ru": "хмф",
            "en": "hmph",
            "de": "hmph",
            "es": "hmpf",
            "fr": "hmph",
            "pl": "hmf",
        },
        descriptions={
            "uk": "Невдоволене «хмф»",
            "ru": "Недовольное «хмф»",
            "en": "Annoyed «hmph»",
            "de": "Genervtes «Hmph»",
            "es": "Molesto «hmpf»",
            "fr": "Agacé « hmph »",
            "pl": "Niezadowolone «hmf»",
        },
        aliases=("хмф", "фу", "хмпф", "hmph", "hnn", "hmpf", "hmf"),
    ),
)

_BUILTIN_BY_ID: Dict[str, OmniVoiceTagSpec] = {s.id: s for s in OMNIVOICE_REACTION_TAGS}
_BUILTIN_BY_TAG: Dict[str, OmniVoiceTagSpec] = {s.tag.lower(): s for s in OMNIVOICE_REACTION_TAGS}

# Усі 13 офіційних тегів із ServeurpersoCom/omnivoice.cpp/src/prompt-tts.h
OMNIVOICE_ALL_OFFICIAL_TAGS: Tuple[str, ...] = (
    "[laughter]",
    "[sigh]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
)

_TAG_TO_EMOJI: Dict[str, str] = {spec.tag: spec.emoji for spec in OMNIVOICE_REACTION_TAGS}
_TAG_TO_EMOJI.update(
    {
        "[surprise-yo]": "😮",
        "[question-ah]": "🤔",
        "[question-oh]": "🤔",
        "[question-ei]": "🤨",
        "[question-yi]": "🤨",
    }
)

_DEFAULT_ALIAS_TO_TAG: Dict[str, str] = {}
for _spec in OMNIVOICE_REACTION_TAGS:
    for _cmd in _spec.primary_cmds.values():
        _DEFAULT_ALIAS_TO_TAG.setdefault(_cmd.lower(), _spec.tag)
    for _al in _spec.aliases:
        _DEFAULT_ALIAS_TO_TAG.setdefault(_al.lower(), _spec.tag)

# Тригери команд: `/`, `\` або `|` на початку рядка або після пробілу (ігнорує URL, `1/2`, `C:\path`, `a|b`)
_ACTIVE_SLASH_RE = re.compile(
    r"^(?P<prefix>(?:.*\s)?)(?P<trigger>[/\\|])(?P<query>[^\s/\\|\[\]]*)$",
    re.DOTALL,
)

_ALL_OFFICIAL_TAGS_RE = re.compile(
    "|".join(re.escape(t) for t in sorted(OMNIVOICE_ALL_OFFICIAL_TAGS, key=len, reverse=True)),
    re.IGNORECASE,
)

_SLASH_TOKEN_RE = re.compile(r"(?<!\S)[/\\|]([^\s/\\|\[\],.!?;:]+)")


def sanitize_reaction_command_name(raw: str) -> str:
    """Очищає назву команди від початкових `/`, `\\`, `|`, пробілів та дужок."""
    s = str(raw or "").strip()
    while s and s[0] in "/\\|":
        s = s[1:].strip()
    s = re.sub(r"[\s/\\|\[\],.!?;:]+", "", s)
    return s.lower()


def get_default_reaction_commands(ui_lang: str = "uk") -> List[Dict[str, Any]]:
    """Повертає стандартний список із 8 вбудованих реакцій для збереження в config.json."""
    lang = str(ui_lang or "uk").strip().lower()
    if lang == "ua":
        lang = "uk"
    items: List[Dict[str, Any]] = []
    for spec in OMNIVOICE_REACTION_TAGS:
        cmd = spec.primary_cmds.get(lang) or spec.primary_cmds.get("en") or spec.aliases[0]
        desc = spec.descriptions.get(lang) or spec.descriptions.get("en") or ""
        items.append(
            {
                "id": spec.id,
                "command": cmd,
                "aliases": list(spec.aliases),
                "display": spec.emoji,
                "description": desc,
                "audio_file": "",
                "omnivoice_tag": spec.omnivoice_tag,
                "enabled": True,
            }
        )
    return items


def resolve_reaction_audio_path(item: Dict[str, Any], base_dir: Optional[Path] = None) -> str:
    """Повертає абсолютний шлях до прикріпленого аудіофайлу, якщо він існує на диску, інакше ''."""
    raw = str(item.get("audio_file") or "").strip()
    if not raw:
        return ""
    cand = Path(raw)
    if not cand.is_absolute() and base_dir is not None:
        cand = base_dir / cand
    try:
        if cand.is_file():
            return str(cand)
    except OSError:
        pass
    return ""


def is_reaction_item_playable(
    item: Dict[str, Any],
    omnivoice_enabled: bool,
    *,
    base_dir: Optional[Path] = None,
) -> bool:
    """
    Чи може команда працювати прямо зараз:
    - або прикріплений існуючий аудіофайл (працює на всіх TTS-рушіях);
    - або увімкнена інтеграція з OmniVoice і команда має вбудований тег OmniVoice.
    """
    if resolve_reaction_audio_path(item, base_dir=base_dir):
        return True
    return bool(omnivoice_enabled and str(item.get("omnivoice_tag") or "").strip())


def load_reaction_command_items(cfg_data: Optional[Dict[str, Any]], ui_lang: str = "uk") -> List[Dict[str, Any]]:
    """Повертає список записів команд із конфігу (автозаповнює стандартними лише якщо ключ None/відсутній)."""
    if not isinstance(cfg_data, dict):
        return get_default_reaction_commands(ui_lang)
    raw = cfg_data.get("reaction_commands", None)
    if raw is None:
        return get_default_reaction_commands(ui_lang)
    if not isinstance(raw, list):
        return get_default_reaction_commands(ui_lang)
    out: List[Dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, dict):
            out.append(dict(entry))
    return out


def item_to_reaction_spec(
    item: Dict[str, Any],
    *,
    ui_lang: str = "uk",
    base_dir: Optional[Path] = None,
) -> Optional[OmniVoiceTagSpec]:
    """Конвертує dict-запис із config.json у типізований `OmniVoiceTagSpec`."""
    cmd = sanitize_reaction_command_name(str(item.get("command") or ""))
    if not cmd:
        return None
    item_id = str(item.get("id") or f"rc_{cmd}").strip()
    ov_tag = str(item.get("omnivoice_tag") or "").strip()
    builtin = _BUILTIN_BY_ID.get(item_id) or (_BUILTIN_BY_TAG.get(ov_tag.lower()) if ov_tag else None)

    # Токен у буфері: для вбудованих OmniVoice-тегів залишаємо канонічний `[laughter]`, для власних — `[хаха]`
    token = ov_tag if ov_tag else f"[{cmd}]"
    display = str(item.get("display") if item.get("display") is not None else "").strip()
    desc = str(item.get("description") or "").strip()
    audio_path = resolve_reaction_audio_path(item, base_dir=base_dir)
    enabled = bool(item.get("enabled", True))

    raw_aliases = item.get("aliases")
    alias_list: List[str] = [cmd]
    if isinstance(raw_aliases, (list, tuple)):
        for al in raw_aliases:
            clean_al = sanitize_reaction_command_name(str(al))
            if clean_al and clean_al not in alias_list:
                alias_list.append(clean_al)
    elif builtin is not None:
        for al in builtin.aliases:
            if al not in alias_list:
                alias_list.append(al)

    prim_cmds = dict(builtin.primary_cmds) if builtin is not None else {}
    lang_key = str(ui_lang or "uk").strip().lower()
    if lang_key == "ua":
        lang_key = "uk"
    prim_cmds[lang_key] = cmd
    descs = dict(builtin.descriptions) if builtin is not None else {}
    descs[lang_key] = desc

    return OmniVoiceTagSpec(
        id=item_id,
        tag=token,
        emoji=display,
        primary_cmds=prim_cmds,
        descriptions=descs,
        aliases=tuple(alias_list),
        audio_file=audio_path,
        omnivoice_tag=ov_tag,
        enabled=enabled,
        custom_cmd=cmd,
        custom_desc=desc,
    )


def build_active_reaction_specs(
    cfg_data: Optional[Dict[str, Any]] = None,
    *,
    engine_id: Optional[str] = None,
    ui_lang: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> List[OmniVoiceTagSpec]:
    """
    Повертає впорядкований список АКТИВНИХ команд для поточного рушія:
    - Команди з прикріпленим аудіофайлом і увімкненою галочкою працюють на БУДЬ-ЯКОМУ TTS-рушії.
    - Вбудовані команди OmniVoice без файлу працюють, коли увімкнена інтеграція з OmniVoice і активний рушій == 'omnivoice'.
    """
    if cfg_data is None:
        return list(OMNIVOICE_REACTION_TAGS)
    eff_lang = str(ui_lang or cfg_data.get("app_language", "uk") or "uk").strip().lower() or "uk"
    eff_engine = str(engine_id if engine_id is not None else cfg_data.get("active_engine", "") or "").strip()
    ov_enabled = bool(cfg_data.get("reaction_omnivoice_enabled", True))
    items = load_reaction_command_items(cfg_data, eff_lang)

    active: List[OmniVoiceTagSpec] = []
    for item in items:
        if not bool(item.get("enabled", True)):
            continue
        spec = item_to_reaction_spec(item, ui_lang=eff_lang, base_dir=base_dir)
        if spec is None:
            continue
        if spec.audio_file:
            active.append(spec)
        elif ov_enabled and spec.omnivoice_tag and eff_engine == "omnivoice":
            active.append(spec)
    return active


def build_all_configured_reaction_specs(
    cfg_data: Optional[Dict[str, Any]] = None,
    *,
    ui_lang: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> List[OmniVoiceTagSpec]:
    """Повертає всі налаштовані команди (для відображення токенів в оверлеї та прев’ю стилів)."""
    if cfg_data is None:
        return list(OMNIVOICE_REACTION_TAGS)
    eff_lang = str(ui_lang or cfg_data.get("app_language", "uk") or "uk").strip().lower() or "uk"
    items = load_reaction_command_items(cfg_data, eff_lang)
    specs: List[OmniVoiceTagSpec] = []
    for item in items:
        spec = item_to_reaction_spec(item, ui_lang=eff_lang, base_dir=base_dir)
        if spec is not None:
            specs.append(spec)
    return specs


def get_primary_cmd_name(spec: OmniVoiceTagSpec, ui_lang: str = "uk") -> str:
    if spec.custom_cmd:
        return spec.custom_cmd
    lang = str(ui_lang or "uk").strip().lower()
    if lang == "ua":
        lang = "uk"
    return spec.primary_cmds.get(lang) or spec.primary_cmds.get("en") or spec.aliases[0]


def get_primary_slash_cmd(spec: OmniVoiceTagSpec, ui_lang: str = "uk") -> str:
    return f"/{get_primary_cmd_name(spec, ui_lang)}"


def get_tag_description(spec: OmniVoiceTagSpec, ui_lang: str = "uk") -> str:
    if spec.custom_cmd:
        return spec.custom_desc
    lang = str(ui_lang or "uk").strip().lower()
    if lang == "ua":
        lang = "uk"
    return spec.descriptions.get(lang) or spec.descriptions.get("en") or ""


def get_spec_display_label(spec: OmniVoiceTagSpec, ui_lang: str = "uk") -> str:
    """Повертає емодзі/знак команди або `[команда]`, якщо знак не задано."""
    if spec.emoji:
        return spec.emoji
    return f"[{get_primary_cmd_name(spec, ui_lang)}]"


def extract_active_slash_query(text: str) -> Optional[Tuple[str, str]]:
    """
    Якщо текст закінчується активною командою `/...`, `\\...` або `|...`
    на початку рядка або після пробілу, повертає `(prefix_before_trigger, query_without_trigger)`,
    інакше `None`.
    """
    if not text or ("/" not in text and "\\" not in text and "|" not in text):
        return None
    m = _ACTIVE_SLASH_RE.match(text)
    if not m:
        return None
    return m.group("prefix"), m.group("query")


def match_reaction_tags(
    query: str,
    ui_lang: str = "uk",
    *,
    specs: Optional[Sequence[OmniVoiceTagSpec]] = None,
) -> List[OmniVoiceTagSpec]:
    pool: Sequence[OmniVoiceTagSpec] = OMNIVOICE_REACTION_TAGS if specs is None else specs
    q = str(query or "").strip().lower()
    if not q:
        return list(pool)

    scored: List[Tuple[int, int, OmniVoiceTagSpec]] = []
    for idx, spec in enumerate(pool):
        prim = get_primary_cmd_name(spec, ui_lang).lower()
        tag_inner = spec.tag.strip("[]").lower()
        aliases_low = tuple(a.lower() for a in spec.aliases)
        rank: Optional[int] = None
        if prim == q:
            rank = 0
        elif q in aliases_low:
            rank = 1
        elif prim.startswith(q):
            rank = 2
        elif any(al.startswith(q) for al in aliases_low):
            rank = 3
        elif tag_inner.startswith(q) or q in tag_inner:
            rank = 4
        if rank is not None:
            scored.append((rank, idx, spec))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored]


def _build_alias_to_token_map(specs: Optional[Sequence[OmniVoiceTagSpec]], ui_lang: str = "uk") -> Dict[str, str]:
    if specs is None:
        return _DEFAULT_ALIAS_TO_TAG
    mapping: Dict[str, str] = {}
    for spec in specs:
        prim = get_primary_cmd_name(spec, ui_lang).lower()
        if prim:
            mapping.setdefault(prim, spec.tag)
        for al in spec.aliases:
            low = al.lower()
            if low:
                mapping.setdefault(low, spec.tag)
    return mapping


def complete_slash_command(
    text: str,
    ui_lang: str = "uk",
    *,
    selected_index: int = 0,
    exact_only: bool = False,
    specs: Optional[Sequence[OmniVoiceTagSpec]] = None,
) -> Optional[str]:
    """
    Доповнює активну команду `/...`, `\\...` або `|...` в кінці тексту на канонічний токен `[tag] `.
    - `exact_only=False` (Tab): підставляє обраний збіг (`selected_index`).
    - `exact_only=True` (Space): спрацьовує лише при точному збігу з відомою командою/аліасом.
    """
    parsed = extract_active_slash_query(text)
    if parsed is None:
        return None
    prefix, query = parsed
    q = query.strip().lower()
    if exact_only:
        if not q:
            return None
        alias_map = _build_alias_to_token_map(specs, ui_lang)
        tag = alias_map.get(q)
        if not tag:
            return None
        return f"{prefix}{tag} "

    matches = match_reaction_tags(q, ui_lang, specs=specs)
    if not matches:
        return None
    spec = matches[selected_index % len(matches)]
    return f"{prefix}{spec.tag} "


def replace_active_slash_with_tag(text: str, tag: str) -> str:
    """Замінює активну команду `/...`, `\\...` або `|...` на вказаний канонічний `[tag] `."""
    clean_tag = str(tag or "").strip()
    if not clean_tag:
        return text
    parsed = extract_active_slash_query(text)
    if parsed is not None:
        prefix, _query = parsed
        return f"{prefix}{clean_tag} "
    sep = "" if (not text or text.endswith(" ")) else " "
    return f"{text}{sep}{clean_tag} "


def backspace_trailing_omnivoice_tag(
    text: str,
    *,
    specs: Optional[Sequence[OmniVoiceTagSpec]] = None,
) -> Optional[str]:
    """Якщо буфер закінчується повним токеном `[laughter]` або `[хаха]`, видаляє його одним натисканням Backspace."""
    if not text or not text.endswith("]"):
        return None
    low = text.lower()
    candidates: List[str] = []
    if specs is not None:
        for spec in specs:
            if spec.tag:
                candidates.append(spec.tag.lower())
    for tag in OMNIVOICE_ALL_OFFICIAL_TAGS:
        if tag not in candidates:
            candidates.append(tag)
    candidates.sort(key=len, reverse=True)
    for tag in candidates:
        if low.endswith(tag):
            return text[: -len(tag)]
    return None


def expand_omnivoice_tags(
    text: str,
    *,
    specs: Optional[Sequence[OmniVoiceTagSpec]] = None,
    ui_lang: str = "uk",
) -> str:
    """Розгортає окремі `/аліаси`, `\\аліаси` та `|аліаси` у канонічні `[tag]`."""
    if not text or ("/" not in text and "\\" not in text and "|" not in text):
        return text
    alias_map = _build_alias_to_token_map(specs, ui_lang)
    if not alias_map:
        return text

    def _repl(m: re.Match[str]) -> str:
        tok = m.group(1).lower()
        tag = alias_map.get(tok)
        return tag if tag else m.group(0)

    return _SLASH_TOKEN_RE.sub(_repl, text)


def strip_omnivoice_tags(text: str, *, extra_tokens: Optional[Sequence[str]] = None) -> str:
    """Видаляє невербальні теги OmniVoice (та опційно додаткові токени команд) для рушіїв без їх підтримки."""
    if not text or "[" not in text:
        return text
    if extra_tokens:
        all_toks = list(OMNIVOICE_ALL_OFFICIAL_TAGS) + [t for t in extra_tokens if t]
        pat = re.compile("|".join(re.escape(t) for t in sorted(set(all_toks), key=len, reverse=True)), re.IGNORECASE)
        cleaned = pat.sub("", text)
    else:
        cleaned = _ALL_OFFICIAL_TAGS_RE.sub("", text)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def format_tags_for_display(
    text: str,
    *,
    specs: Optional[Sequence[OmniVoiceTagSpec]] = None,
    ui_lang: str = "uk",
) -> str:
    """
    Замінює канонічні токени `[laughter]` / `[хаха]` для відображення в оверлеї:
    - якщо задано емодзі/знак — показує його (`😂`);
    - якщо знак порожній — показує `[команда]` (напр. `[хаха]`).
    """
    if not text or "[" not in text:
        return text
    if specs is None:
        def _repl_def(m: re.Match[str]) -> str:
            key = m.group(0).lower()
            return _TAG_TO_EMOJI.get(key, m.group(0))
        return _ALL_OFFICIAL_TAGS_RE.sub(_repl_def, text)

    disp_map: Dict[str, str] = dict(_TAG_TO_EMOJI)
    for spec in specs:
        disp_map[spec.tag.lower()] = get_spec_display_label(spec, ui_lang)
    pat = re.compile("|".join(re.escape(t) for t in sorted(disp_map.keys(), key=len, reverse=True)), re.IGNORECASE)
    return pat.sub(lambda m: disp_map.get(m.group(0).lower(), m.group(0)), text)


def split_text_for_reaction_playback(
    text: str,
    specs: Sequence[OmniVoiceTagSpec],
    *,
    is_omnivoice_engine: bool,
    ui_lang: str = "uk",
) -> List[Tuple[str, str, str]]:
    """
    Розбиває текст на послідовні сегменти для озвучення:
    - `("text", clean_text, "")` — звичайний текст для TTS (із збереженням `[tag]` для OmniVoice);
    - `("audio", audio_file_path, display_label)` — прикріплений аудіофайл команди.

    Якщо активних команд із прикріпленими аудіофайлами у тексті немає, повертає один `("text", ...)` сегмент.
    """
    if not text:
        return []
    expanded = expand_omnivoice_tags(text, specs=specs, ui_lang=ui_lang)
    audio_by_token: Dict[str, Tuple[str, str]] = {}
    for spec in specs:
        if spec.audio_file:
            audio_by_token[spec.tag.lower()] = (spec.audio_file, get_spec_display_label(spec, ui_lang))

    if not audio_by_token:
        clean = expanded if is_omnivoice_engine else (strip_omnivoice_tags(expanded) or "")
        return [("text", clean, "")] if clean.strip() else []

    pat = re.compile(
        "|".join(re.escape(tok) for tok in sorted(audio_by_token.keys(), key=len, reverse=True)),
        re.IGNORECASE,
    )
    segments: List[Tuple[str, str, str]] = []
    pos = 0
    for m in pat.finditer(expanded):
        before = expanded[pos : m.start()]
        if before:
            seg_txt = before if is_omnivoice_engine else strip_omnivoice_tags(before)
            seg_txt = re.sub(r"\s{2,}", " ", seg_txt).strip()
            # Залишаємо лише якщо є букви, цифри або теги
            if seg_txt and re.search(r"[\w\[]", seg_txt):
                segments.append(("text", seg_txt, ""))
        af, disp = audio_by_token[m.group(0).lower()]
        segments.append(("audio", af, disp))
        pos = m.end()

    tail = expanded[pos:]
    if tail:
        seg_txt = tail if is_omnivoice_engine else strip_omnivoice_tags(tail)
        seg_txt = re.sub(r"\s{2,}", " ", seg_txt).strip()
        if seg_txt and re.search(r"[\w\[]", seg_txt):
            segments.append(("text", seg_txt, ""))

    return segments
