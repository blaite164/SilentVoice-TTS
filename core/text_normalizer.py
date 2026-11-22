"""
core/text_normalizer.py — Перетворення чисел та символів (%, $, €, ₴, °, №, +, -, =) у слова
для всіх підтримуваних мов інтерфейсу (uk, en, pl, de, fr, es, ru) перед озвучкою TTS.
Оригінальний текст для OBS, оверлею та історії не змінюється.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

SUPPORTED_NORM_LANGS = ("uk", "en", "pl", "de", "fr", "es", "ru")

_UK_UNITS_M = ["", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять"]
_UK_UNITS_F = ["", "одна", "дві", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять"]
_UK_TEENS = [
    "десять", "одинадцять", "дванадцять", "тринадцять", "чотирнадцять",
    "п'ятнадцять", "шістнадцять", "сімнадцять", "вісімнадцять", "дев'ятнадцять",
]
_UK_TENS = ["", "", "двадцять", "тридцять", "сорок", "п'ятдесят", "шістдесят", "сімдесят", "вісімдесят", "дев'яносто"]
_UK_HUNDREDS = ["", "сто", "двісті", "триста", "чотириста", "п'ятсот", "шістсот", "сімсот", "вісімсот", "дев'ятсот"]

_RU_UNITS_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_RU_UNITS_F = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_RU_TEENS = [
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
]
_RU_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_RU_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

_PL_UNITS = ["", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć"]
_PL_TEENS = [
    "dziesięć", "jedenaście", "dwanaście", "trzynaście", "czternaście",
    "piętnaście", "szesnaście", "siedemnaście", "osiemnaście", "dziewiętnaście",
]
_PL_TENS = ["", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt", "sześćdziesiąt", "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt"]
_PL_HUNDREDS = ["", "sto", "dwieście", "trzysta", "czterysta", "pięćset", "sześćset", "siedemset", "osiemset", "dziewięćset"]

_EN_UNITS = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_EN_TEENS = [
    "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_DE_UNITS = ["", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
_DE_TEENS = [
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
    "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]

_FR_UNITS = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"]
_FR_TEENS = [
    "dix", "onze", "douze", "treize", "quatorze",
    "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf",
]

_ES_UNITS = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
_ES_TEENS = [
    "diez", "once", "doce", "trece", "catorce",
    "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
]
_ES_TWENTIES = [
    "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
    "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve",
]
_ES_TENS = ["", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"]
_ES_HUNDREDS = ["", "ciento", "doscientos", "trescientos", "cuatrocientos", "quinientos", "seiscientos", "setecientos", "ochocientos", "novecientos"]


def _plural_form(n: int, form1: str, form2: str, form5: str) -> str:
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 19:
        return form5
    last = n_abs % 10
    if last == 1:
        return form1
    if 2 <= last <= 4:
        return form2
    return form5


def _triplet_slavic(
    n: int,
    units_m: List[str],
    units_f: List[str],
    teens: List[str],
    tens: List[str],
    hundreds: List[str],
    feminine: bool = False,
) -> List[str]:
    words: List[str] = []
    h = n // 100
    rem = n % 100
    if h:
        words.append(hundreds[h])
    if 10 <= rem <= 19:
        words.append(teens[rem - 10])
    else:
        t = rem // 10
        u = rem % 10
        if t:
            words.append(tens[t])
        if u:
            words.append((units_f if feminine else units_m)[u])
    return words


def _int_to_words_slavic(n: int, lang: str, feminine: bool = False) -> str:
    if n == 0:
        return {"uk": "нуль", "ru": "ноль", "pl": "zero"}[lang]

    if lang == "uk":
        units_m, units_f, teens, tens, hundreds = _UK_UNITS_M, _UK_UNITS_F, _UK_TEENS, _UK_TENS, _UK_HUNDREDS
        scales = [
            ("", "", "", feminine),
            ("тисяча", "тисячі", "тисяч", True),
            ("мільйон", "мільйони", "мільйонів", False),
            ("мільярд", "мільярди", "мільярдів", False),
        ]
    elif lang == "ru":
        units_m, units_f, teens, tens, hundreds = _RU_UNITS_M, _RU_UNITS_F, _RU_TEENS, _RU_TENS, _RU_HUNDREDS
        scales = [
            ("", "", "", feminine),
            ("тысяча", "тысячи", "тысяч", True),
            ("миллион", "миллиона", "миллионов", False),
            ("миллиард", "миллиарда", "миллиардов", False),
        ]
    else:  # pl
        units_m, units_f, teens, tens, hundreds = _PL_UNITS, _PL_UNITS, _PL_TEENS, _PL_TENS, _PL_HUNDREDS
        scales = [
            ("", "", "", False),
            ("tysiąc", "tysiące", "tysięcy", False),
            ("milion", "miliony", "milionów", False),
            ("miliard", "miliardy", "miliardów", False),
        ]

    parts: List[str] = []
    scale_idx = 0
    cur = abs(n)
    while cur > 0 and scale_idx < len(scales):
        chunk = cur % 1000
        cur //= 1000
        if chunk:
            f1, f2, f5, is_fem = scales[scale_idx]
            if lang == "pl" and scale_idx > 0 and chunk == 1:
                chunk_words = [f1]
            else:
                chunk_words = _triplet_slavic(chunk, units_m, units_f, teens, tens, hundreds, feminine=is_fem)
                if scale_idx > 0:
                    chunk_words.append(_plural_form(chunk, f1, f2, f5))
            parts = chunk_words + parts
        scale_idx += 1
    return " ".join(parts)


def _triplet_en(n: int) -> List[str]:
    words: List[str] = []
    h = n // 100
    rem = n % 100
    if h:
        words.extend([_EN_UNITS[h], "hundred"])
    if 10 <= rem <= 19:
        words.append(_EN_TEENS[rem - 10])
    else:
        t = rem // 10
        u = rem % 10
        if t:
            words.append(_EN_TENS[t])
        if u:
            words.append(_EN_UNITS[u])
    return words


def _int_to_words_en(n: int) -> str:
    if n == 0:
        return "zero"
    scales = ["", "thousand", "million", "billion"]
    parts: List[str] = []
    scale_idx = 0
    cur = abs(n)
    while cur > 0 and scale_idx < len(scales):
        chunk = cur % 1000
        cur //= 1000
        if chunk:
            chunk_words = _triplet_en(chunk)
            if scales[scale_idx]:
                chunk_words.append(scales[scale_idx])
            parts = chunk_words + parts
        scale_idx += 1
    return " ".join(parts)


def _triplet_de(n: int, is_scale: bool = False) -> str:
    parts: List[str] = []
    h = n // 100
    rem = n % 100
    if h:
        parts.append(("ein" if h == 1 else _DE_UNITS[h]) + "hundert")
    if rem:
        if 10 <= rem <= 19:
            parts.append(_DE_TEENS[rem - 10])
        elif rem < 10:
            parts.append("ein" if (rem == 1 and is_scale) else _DE_UNITS[rem])
        else:
            t = rem // 10
            u = rem % 10
            if u == 0:
                parts.append(_DE_TENS[t])
            else:
                u_str = "ein" if u == 1 else _DE_UNITS[u]
                parts.append(f"{u_str}und{_DE_TENS[t]}")
    return "".join(parts)


def _int_to_words_de(n: int) -> str:
    if n == 0:
        return "null"
    parts: List[str] = []
    billions = (n // 1_000_000_000) % 1000
    millions = (n // 1_000_000) % 1000
    thousands = (n // 1000) % 1000
    rest = n % 1000
    if billions:
        parts.append("eine Milliarde" if billions == 1 else f"{_triplet_de(billions)} Milliarden")
    if millions:
        parts.append("eine Million" if millions == 1 else f"{_triplet_de(millions)} Millionen")
    if thousands:
        parts.append("eintausend" if thousands == 1 else f"{_triplet_de(thousands, is_scale=True)}tausend")
    if rest:
        parts.append(_triplet_de(rest))
    return " ".join(parts)


def _sub100_fr(n: int) -> str:
    if n == 0:
        return ""
    if n < 10:
        return _FR_UNITS[n]
    if 10 <= n <= 19:
        return _FR_TEENS[n - 10]
    if n < 70:
        tens_names = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante"]
        t, u = n // 10, n % 10
        if u == 0:
            return tens_names[t]
        if u == 1:
            return f"{tens_names[t]} et un"
        return f"{tens_names[t]}-{_FR_UNITS[u]}"
    if n < 80:
        rem = n - 60
        if rem == 11:
            return "soixante et onze"
        return f"soixante-{_sub100_fr(rem)}"
    if n == 80:
        return "quatre-vingts"
    return f"quatre-vingt-{_sub100_fr(n - 80)}"


def _triplet_fr(n: int) -> str:
    h = n // 100
    rem = n % 100
    parts: List[str] = []
    if h == 1:
        parts.append("cent")
    elif h > 1:
        parts.append(f"{_FR_UNITS[h]} cents" if rem == 0 else f"{_FR_UNITS[h]} cent")
    if rem:
        parts.append(_sub100_fr(rem))
    return " ".join(parts)


def _int_to_words_fr(n: int) -> str:
    if n == 0:
        return "zéro"
    parts: List[str] = []
    billions = (n // 1_000_000_000) % 1000
    millions = (n // 1_000_000) % 1000
    thousands = (n // 1000) % 1000
    rest = n % 1000
    if billions:
        parts.append("un milliard" if billions == 1 else f"{_triplet_fr(billions)} milliards")
    if millions:
        parts.append("un million" if millions == 1 else f"{_triplet_fr(millions)} millions")
    if thousands:
        parts.append("mille" if thousands == 1 else f"{_triplet_fr(thousands)} mille")
    if rest:
        parts.append(_triplet_fr(rest))
    return " ".join(parts)


def _triplet_es(n: int) -> str:
    if n == 100:
        return "cien"
    h = n // 100
    rem = n % 100
    parts: List[str] = []
    if h:
        parts.append(_ES_HUNDREDS[h])
    if rem:
        if rem < 10:
            parts.append(_ES_UNITS[rem])
        elif rem < 20:
            parts.append(_ES_TEENS[rem - 10])
        elif rem < 30:
            parts.append(_ES_TWENTIES[rem - 20])
        else:
            t, u = rem // 10, rem % 10
            if u == 0:
                parts.append(_ES_TENS[t])
            else:
                parts.append(f"{_ES_TENS[t]} y {_ES_UNITS[u]}")
    return " ".join(parts)


def _int_to_words_es(n: int) -> str:
    if n == 0:
        return "cero"
    parts: List[str] = []
    billions = (n // 1_000_000_000) % 1000
    millions = (n // 1_000_000) % 1000
    thousands = (n // 1000) % 1000
    rest = n % 1000
    if billions:
        parts.append("mil millones" if billions == 1 else f"{_triplet_es(billions)} mil millones")
    if millions:
        parts.append("un millón" if millions == 1 else f"{_triplet_es(millions)} millones")
    if thousands:
        parts.append("mil" if thousands == 1 else f"{_triplet_es(thousands)} mil")
    if rest:
        parts.append(_triplet_es(rest))
    return " ".join(parts)


def _int_to_words(n: int, lang: str, feminine: bool = False) -> str:
    if lang in ("uk", "ru", "pl"):
        return _int_to_words_slavic(n, lang, feminine=feminine)
    if lang == "de":
        return _int_to_words_de(n)
    if lang == "fr":
        return _int_to_words_fr(n)
    if lang == "es":
        return _int_to_words_es(n)
    return _int_to_words_en(n)


_SIGN_WORDS: Dict[str, Tuple[str, str]] = {
    "uk": ("плюс ", "мінус "),
    "ru": ("плюс ", "минус "),
    "pl": ("plus ", "minus "),
    "en": ("plus ", "minus "),
    "de": ("plus ", "minus "),
    "fr": ("plus ", "moins "),
    "es": ("más ", "menos "),
}

_DECIMAL_WORDS: Dict[str, str] = {
    "uk": "кома",
    "ru": "запятая",
    "pl": "przecinek",
    "en": "point",
    "de": "Komma",
    "fr": "virgule",
    "es": "coma",
}


def number_to_words(num_str: str, lang: str = "uk", feminine: bool = False) -> str:
    """Перетворює ціле або дробове число (рядком) на слова для обраної мови."""
    norm_lang = lang if lang in SUPPORTED_NORM_LANGS else "uk"
    clean = num_str.replace(" ", "").replace(",", ".")
    plus_w, minus_w = _SIGN_WORDS[norm_lang]
    sign = ""
    if clean.startswith("+"):
        sign = plus_w
        clean = clean[1:]
    elif clean.startswith("-"):
        sign = minus_w
        clean = clean[1:]

    if "." in clean:
        int_part_s, frac_part_s = clean.split(".", 1)
        if not int_part_s.isdigit() or not frac_part_s.isdigit():
            return num_str
        if len(int_part_s) > 12:
            digits = " ".join(_int_to_words(int(d), norm_lang) for d in int_part_s)
            frac_digits = " ".join(_int_to_words(int(d), norm_lang) for d in frac_part_s)
            return f"{sign}{digits} {_DECIMAL_WORDS[norm_lang]} {frac_digits}".strip()

        int_val = int(int_part_s)
        frac_val = int(frac_part_s)

        if norm_lang in ("uk", "ru") and len(frac_part_s) <= 3:
            int_words = _int_to_words_slavic(int_val, norm_lang, feminine=True)
            whole_word = (
                _plural_form(int_val, "ціла", "цілих", "цілих")
                if norm_lang == "uk"
                else _plural_form(int_val, "целая", "целых", "целых")
            )
            if len(frac_part_s) == 1:
                unit = _plural_form(frac_val, "десята", "десятих", "десятих") if norm_lang == "uk" else _plural_form(frac_val, "десятая", "десятых", "десятых")
            elif len(frac_part_s) == 2:
                unit = _plural_form(frac_val, "сота", "сотих", "сотих") if norm_lang == "uk" else _plural_form(frac_val, "сотая", "сотых", "сотых")
            else:
                unit = _plural_form(frac_val, "тисячна", "тисячних", "тисячних") if norm_lang == "uk" else _plural_form(frac_val, "тысячная", "тысячных", "тысячных")
            frac_words = _int_to_words_slavic(frac_val, norm_lang, feminine=True)
            return f"{sign}{int_words} {whole_word} {frac_words} {unit}".strip()

        frac_words = " ".join(_int_to_words(int(d), norm_lang) for d in frac_part_s)
        return f"{sign}{_int_to_words(int_val, norm_lang)} {_DECIMAL_WORDS[norm_lang]} {frac_words}".strip()

    if not clean.isdigit():
        return num_str
    if len(clean) > 12:
        return sign + " ".join(_int_to_words(int(d), norm_lang) for d in clean)
    return sign + _int_to_words(int(clean), norm_lang, feminine=feminine)


def _unit_word(num_str: str, forms_by_lang: Dict[str, Tuple[str, ...]], lang: str) -> str:
    clean = num_str.replace(" ", "").replace(",", ".").lstrip("+-")
    is_frac = "." in clean
    forms = forms_by_lang.get(lang, forms_by_lang["en"])
    if len(forms) == 1:
        return forms[0]
    if len(forms) == 2:
        return forms[0] if (not is_frac and clean == "1") else forms[1]
    if is_frac:
        return forms[1]
    return _plural_form(int(clean), forms[0], forms[1], forms[2])


_CURRENCY_DOLLAR: Dict[str, Tuple[str, ...]] = {
    "uk": ("долар", "долари", "доларів"),
    "ru": ("доллар", "доллара", "долларов"),
    "pl": ("dolar", "dolary", "dolarów"),
    "en": ("dollar", "dollars"),
    "de": ("Dollar",),
    "fr": ("dollar", "dollars"),
    "es": ("dólar", "dólares"),
}

_CURRENCY_EURO: Dict[str, Tuple[str, ...]] = {
    "uk": ("євро",),
    "ru": ("евро",),
    "pl": ("euro",),
    "en": ("euro", "euros"),
    "de": ("Euro",),
    "fr": ("euro", "euros"),
    "es": ("euro", "euros"),
}

_CURRENCY_UAH: Dict[str, Tuple[str, ...]] = {
    "uk": ("гривня", "гривні", "гривень"),
    "ru": ("гривна", "гривны", "гривен"),
    "pl": ("hrywna", "hrywny", "hrywien"),
    "en": ("hryvnia", "hryvnias"),
    "de": ("Griwna",),
    "fr": ("hryvnia", "hryvnias"),
    "es": ("grivna", "grivnas"),
}

_PERCENT_FORMS: Dict[str, Tuple[str, ...]] = {
    "uk": ("відсоток", "відсотки", "відсотків"),
    "ru": ("процент", "процента", "процентов"),
    "pl": ("procent",),
    "en": ("percent",),
    "de": ("Prozent",),
    "fr": ("pour cent",),
    "es": ("por ciento",),
}

_DEGREE_FORMS: Dict[str, Tuple[str, ...]] = {
    "uk": ("градус", "градуси", "градусів"),
    "ru": ("градус", "градуса", "градусов"),
    "pl": ("stopień", "stopnie", "stopni"),
    "en": ("degree", "degrees"),
    "de": ("Grad",),
    "fr": ("degré", "degrés"),
    "es": ("grado", "grados"),
}

_SCALE_NAMES: Dict[str, Tuple[str, str]] = {
    "uk": (" Цельсія", " Фаренгейта"),
    "ru": (" Цельсия", " Фаренгейта"),
    "pl": (" Celsjusza", " Fahrenheita"),
    "en": (" Celsius", " Fahrenheit"),
    "de": (" Celsius", " Fahrenheit"),
    "fr": (" Celsius", " Fahrenheit"),
    "es": (" Celsius", " Fahrenheit"),
}

_NUM_WORD: Dict[str, str] = {
    "uk": "номер",
    "ru": "номер",
    "pl": "numer",
    "en": "number",
    "de": "Nummer",
    "fr": "numéro",
    "es": "número",
}

_SYM_MAP: Dict[str, Dict[str, str]] = {
    "uk": {"%": "відсотків", "$": "доларів", "€": "євро", "₴": "гривень", "№": "номер", "±": "плюс мінус", "=": "дорівнює", "&": "і"},
    "ru": {"%": "процентов", "$": "долларов", "€": "евро", "₴": "гривен", "№": "номер", "±": "плюс минус", "=": "равно", "&": "и"},
    "pl": {"%": "procent", "$": "dolarów", "€": "euro", "₴": "hrywien", "№": "numer", "±": "plus minus", "=": "równa się", "&": "i"},
    "en": {"%": "percent", "$": "dollars", "€": "euros", "₴": "hryvnias", "№": "number", "±": "plus or minus", "=": "equals", "&": "and"},
    "de": {"%": "Prozent", "$": "Dollar", "€": "Euro", "₴": "Griwna", "№": "Nummer", "±": "plus minus", "=": "gleich", "&": "und"},
    "fr": {"%": "pour cent", "$": "dollars", "€": "euros", "₴": "hryvnias", "№": "numéro", "±": "plus ou moins", "=": "égale", "&": "et"},
    "es": {"%": "por ciento", "$": "dólares", "€": "euros", "₴": "grivnas", "№": "número", "±": "más menos", "=": "igual a", "&": "y"},
}


def normalize_tts_text(text: str, lang: str = "uk") -> str:
    """
    Перетворює числа, відсотки, валюти, градуси та базові знаки у слова для природного синтезу TTS
    усіма підтримуваними мовами (uk, en, pl, de, fr, es, ru).
    """
    if not text:
        return ""
    norm_lang = (lang or "uk").strip().lower()
    if norm_lang not in SUPPORTED_NORM_LANGS:
        norm_lang = "uk"

    out = text

    # 1. Валюти перед або після числа ($100, 100$, €50, 50€, ₴200, 200₴)
    def _replace_currency(num_s: str, sym: str) -> str:
        if sym == "$":
            w = _unit_word(num_s, _CURRENCY_DOLLAR, norm_lang)
            return f"{number_to_words(num_s, norm_lang)} {w}"
        if sym == "€":
            w = _unit_word(num_s, _CURRENCY_EURO, norm_lang)
            return f"{number_to_words(num_s, norm_lang)} {w}"
        if sym == "₴":
            w = _unit_word(num_s, _CURRENCY_UAH, norm_lang)
            return f"{number_to_words(num_s, norm_lang, feminine=(norm_lang in ('uk', 'ru')))} {w}"
        return f"{num_s}{sym}"

    out = re.sub(
        r"([\$€₴])\s*([+-]?\d+(?:[.,]\d+)?)",
        lambda m: _replace_currency(m.group(2), m.group(1)),
        out,
    )
    out = re.sub(
        r"([+-]?\d+(?:[.,]\d+)?)\s*([\$€₴])",
        lambda m: _replace_currency(m.group(1), m.group(2)),
        out,
    )

    # 2. Температура / градуси: 25°C, -5°F, 90°
    def _replace_deg(m: re.Match) -> str:
        num_s = m.group(1)
        scale = (m.group(2) or "").upper()
        deg_w = _unit_word(num_s, _DEGREE_FORMS, norm_lang)
        c_name, f_name = _SCALE_NAMES[norm_lang]
        extra = c_name if scale == "C" else (f_name if scale == "F" else "")
        return f"{number_to_words(num_s, norm_lang)} {deg_w}{extra}"

    out = re.sub(r"([+-]?\d+(?:[.,]\d+)?)\s*°\s*([CFcf])?", _replace_deg, out)

    # 3. Відсотки: 50%, 2.5 %, 1%
    def _replace_pct(m: re.Match) -> str:
        num_s = m.group(1)
        pct_w = _unit_word(num_s, _PERCENT_FORMS, norm_lang)
        return f"{number_to_words(num_s, norm_lang)} {pct_w}"

    out = re.sub(r"([+-]?\d+(?:[.,]\d+)?)\s*%", _replace_pct, out)

    # 4. Номер №5
    num_word = _NUM_WORD[norm_lang]
    out = re.sub(r"№\s*(\d+)", lambda m: f"{num_word} {number_to_words(m.group(1), norm_lang)}", out)

    # 5. Час (наприклад 14:30)
    out = re.sub(
        r"\b(\d{1,2}):(\d{2})\b",
        lambda m: f"{number_to_words(m.group(1), norm_lang)} {number_to_words(m.group(2), norm_lang)}",
        out,
    )

    # 6. Окремі знаки: %, $, €, ₴, №, ±, =, &
    sym_map = _SYM_MAP[norm_lang]
    out = re.sub(r"(?<!\S)([%$€₴№±=&])(?!\S)", lambda m: sym_map[m.group(1)], out)
    plus_op = _SIGN_WORDS[norm_lang][0].strip()
    out = re.sub(
        r"(\d)\s*([+=])\s*(\d)",
        lambda m: f"{m.group(1)} {plus_op if m.group(2) == '+' else sym_map['=']} {m.group(3)}",
        out,
    )

    # 7. Решта окремих чисел (цілих і десяткових, включно з +5 / -10)
    out = re.sub(
        r"(?<![A-Za-zА-Яа-яІіЇїЄєҐґÀ-ÿ0-9_])([+-]?\d+(?:[.,]\d+)?)(?![A-Za-zА-Яа-яІіЇїЄєҐґÀ-ÿ0-9_])",
        lambda m: number_to_words(m.group(1), norm_lang),
        out,
    )

    return re.sub(r"\s{2,}", " ", out).strip()


# ──────────────────────────────────────────────────────────────────────
# Фонетична адаптація латиниці / кирилиці для локальних одномовних TTS
# ──────────────────────────────────────────────────────────────────────

_EN_LETTER_NAMES_UK: Dict[str, str] = {
    "a": "ей", "b": "бі", "c": "сі", "d": "ді", "e": "і", "f": "еф",
    "g": "джі", "h": "ейч", "i": "ай", "j": "джей", "k": "кей", "l": "ел",
    "m": "ем", "n": "ен", "o": "оу", "p": "пі", "q": "к'ю", "r": "ар",
    "s": "ес", "t": "ті", "u": "ю", "v": "ві", "w": "дабл ю", "x": "ікс",
    "y": "вай", "z": "зед",
}

_EN_LETTER_NAMES_RU: Dict[str, str] = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф",
    "g": "джи", "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл",
    "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл ю", "x": "икс",
    "y": "вай", "z": "зед",
}

_EN_SPELL_ACRONYMS = {
    "ai", "api", "obs", "fps", "tts", "vpn", "vip", "usb", "cpu", "gpu",
    "ssd", "hdd", "ram", "rgb", "pdf", "url", "dns", "http", "https", "ip",
    "pc", "tv", "hd", "fhd", "uhd", "gg", "wp", "gl", "hf", "brb", "afk",
    "wtf", "omg", "idk", "rtx", "gtx", "cs", "gta", "bmw", "vk", "tg",
}

_EN_WORD_DICT_UK: Dict[str, str] = {
    "hi": "хай", "hello": "хелоу", "hey": "хей", "bye": "бай",
    "ok": "окей", "okay": "окей", "yes": "єс", "no": "ноу",
    "please": "пліз", "pls": "пліз", "plz": "пліз",
    "thanks": "сенкс", "thank": "сенк", "thx": "сенкс", "sorry": "сорі",
    "you": "ю", "your": "йор", "we": "ві", "they": "зей", "he": "хі", "she": "ші", "it": "іт",
    "i'm": "айм", "im": "айм", "don't": "донт", "dont": "донт", "can't": "кант", "cant": "кант",
    "it's": "ітс", "you're": "юр", "we're": "вір", "that's": "зетс",
    "good": "гуд", "bad": "бед", "nice": "найс", "cool": "кул",
    "bro": "бро", "man": "мен", "guys": "гайз", "friend": "френд",
    "love": "лав", "like": "лайк", "hate": "хейт",
    "what": "вот", "why": "вай", "how": "хау", "who": "ху", "where": "вер", "when": "вен",
    "one": "ван", "two": "ту", "three": "срі", "four": "фор", "five": "файв",
    "six": "сікс", "seven": "севен", "eight": "ейт", "nine": "найн", "ten": "тен", "zero": "зіро",
    "all": "ол", "go": "гоу", "stop": "стоп", "start": "старт", "play": "плей",
    "pause": "пауз", "wait": "вейт", "help": "хелп", "ready": "реді", "check": "чек",
    "test": "тест", "time": "тайм", "game": "гейм", "win": "він", "lose": "луз",
    "team": "тім", "solo": "соло", "duo": "дуо", "bot": "бот", "pro": "про",
    "noob": "нуб", "ping": "пінг", "lag": "лаг", "ez": "ізі", "rip": "ріп",
    "lol": "лол", "kek": "кек", "stream": "стрім", "chat": "чат", "sub": "саб",
    "subscribe": "сабскрайб", "donate": "донат", "mute": "м'ют", "unmute": "анм'ют",
    "voice": "войс", "mic": "майк", "sound": "саунд", "audio": "аудіо", "video": "відео",
    "music": "м'юзік", "discord": "діскорд", "youtube": "ютюб", "twitch": "твіч",
    "steam": "стім", "google": "гугл", "windows": "віндовс", "telegram": "телеграм",
    "ctrl": "контрол", "alt": "альт", "shift": "шифт", "enter": "ентер",
    "esc": "ескейп", "tab": "таб", "space": "спейс", "capslock": "капс лок",
    "minecraft": "майнкрафт", "roblox": "роблокс", "valorant": "валорант", "dota": "дота",
    "python": "пайтон", "code": "код", "file": "файл", "link": "лінк",
    "update": "апдейт", "error": "еррор", "fix": "фікс", "bug": "баг",
    "new": "нью", "old": "олд", "live": "лайв", "online": "онлайн", "offline": "офлайн",
    "server": "сервер", "proxy": "проксі", "socks": "сокс", "silent": "сайлент",
    "world": "ворлд", "hello world": "хелоу ворлд", "water": "вотер", "the": "зе",
}

_EN_WORD_DICT_RU_OVERRIDES: Dict[str, str] = {
    "yes": "йес", "please": "плиз", "pls": "плиз", "plz": "плиз",
    "sorry": "сори", "we": "ви", "he": "хи", "she": "ши", "it": "ит",
    "it's": "итс", "we're": "вир", "three": "сри", "six": "сикс",
    "ready": "реди", "team": "тим", "win": "вин", "ping": "пинг",
    "ez": "изи", "rip": "рип", "stream": "стрим", "music": "мьюзик",
    "discord": "дискорд", "twitch": "твич", "steam": "стим", "windows": "виндовс",
    "link": "линк", "fix": "фикс", "proxy": "прокси",
}

_CYR_TO_LATIN_TABLE = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "ґ": "g", "д": "d", "е": "e", "є": "ye",
    "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "yi", "й": "y", "к": "k", "л": "l",
    "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "ё": "yo",
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Ґ": "G", "Д": "D", "Е": "E", "Є": "Ye",
    "Ж": "Zh", "З": "Z", "И": "Y", "І": "I", "Ї": "Yi", "Й": "Y", "К": "K", "Л": "L",
    "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "Kh", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh", "Щ": "Shch", "Ъ": "",
    "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya", "Ё": "Yo",
})


def _uk_cyr_to_ru_cyr(text: str) -> str:
    return (
        text.replace("і", "и")
        .replace("І", "И")
        .replace("ї", "йи")
        .replace("Ї", "Йи")
        .replace("є", "е")
        .replace("Є", "Е")
        .replace("ґ", "г")
        .replace("Ґ", "Г")
    )


def _phonetic_en_word_to_uk(word: str) -> str:
    w = word.lower().replace("'", "")
    if not w:
        return ""
    # Багатосимвольні правила (зліва направо)
    rules: List[Tuple[str, str]] = [
        ("eigh", "ей"), ("igh", "ай"), ("tion", "шн"), ("sion", "жн"),
        ("ough", "о"), ("augh", "о"), ("tch", "ч"), ("dge", "дж"),
        ("sch", "ш"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"),
        ("th", "с"), ("ph", "ф"), ("qu", "кв"), ("ck", "к"), ("ng", "нг"),
        ("ee", "і"), ("ea", "і"), ("oo", "у"), ("ou", "ау"), ("ow", "оу"),
        ("ai", "ей"), ("ay", "ей"), ("ei", "ей"), ("ey", "ей"),
        ("oi", "ой"), ("oy", "ой"), ("au", "о"), ("aw", "о"), ("ew", "ю"),
        ("ie", "і"), ("wh", "в"),
    ]
    out: List[str] = []
    i = 0
    n = len(w)
    vowels = set("aeiouy")
    while i < n:
        if i == 0 and w.startswith("kn"):
            out.append("н")
            i += 2
            continue
        if i == 0 and w.startswith("wr"):
            out.append("р")
            i += 2
            continue
        matched = False
        for pat, rep in rules:
            if w.startswith(pat, i):
                out.append(rep)
                i += len(pat)
                matched = True
                break
        if matched:
            continue

        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        # Подвоєна приголосна читається як один звук
        if ch == nxt and ch not in vowels and ch not in ("c", "g", "x"):
            i += 1
            continue

        # Німа фінальна -e після приголосної (якщо вже є інша голосна)
        if ch == "e" and i == n - 1 and n > 2 and any(c in vowels for c in w[:-1]):
            i += 1
            continue

        if ch == "c":
            out.append("с" if nxt in ("e", "i", "y") else "к")
        elif ch == "g":
            out.append("дж" if nxt in ("e", "i", "y") else "г")
        elif ch == "j":
            out.append("дж")
        elif ch == "x":
            out.append("кс")
        elif ch == "w":
            out.append("в")
        elif ch == "q":
            out.append("к")
        elif ch == "y":
            if i == 0 and nxt in vowels:
                out.append("й")
            elif i == n - 1 and n <= 3 and not any(c in "aeiou" for c in w):
                out.append("ай")
            else:
                out.append("і")
        elif ch == "a":
            # Відкритий склад a + приголосна + голосна (напр. name, game, make)
            if i + 2 < n and w[i + 1] not in vowels and w[i + 2] == "e" and i + 2 == n - 1:
                out.append("ей")
            else:
                out.append("а")
        elif ch == "i":
            if i + 2 < n and w[i + 1] not in vowels and w[i + 2] == "e" and i + 2 == n - 1:
                out.append("ай")
            else:
                out.append("і")
        elif ch == "o":
            if i + 2 < n and w[i + 1] not in vowels and w[i + 2] == "e" and i + 2 == n - 1:
                out.append("оу")
            else:
                out.append("о")
        elif ch == "u":
            out.append("ю" if (i + 2 < n and w[i + 1] not in vowels and w[i + 2] in vowels) else "а")
        elif ch == "e":
            out.append("е")
        else:
            single_map = {
                "b": "б", "d": "д", "f": "ф", "h": "х", "k": "к", "l": "л",
                "m": "м", "n": "н", "p": "п", "r": "р", "s": "с", "t": "т",
                "v": "в", "z": "з",
            }
            out.append(single_map.get(ch, ""))
        i += 1
    return "".join(out)


def transliterate_latin_for_cyrillic_tts(text: str, lang: str = "uk") -> str:
    """
    Перетворює англійські літери, абревіатури й слова (латиницю) у фонетичну кирилицю,
    щоб кириличні локальні TTS-моделі (Silero, HolosTTS, TeraTTS) могли їх коректно озвучити.
    """
    if not text or not re.search(r"[A-Za-z]", text):
        return text

    is_ru = (lang or "").strip().lower() == "ru"
    letter_names = _EN_LETTER_NAMES_RU if is_ru else _EN_LETTER_NAMES_UK

    def _conv_token(m: re.Match) -> str:
        raw = m.group(0)
        low = raw.lower()
        if len(low) == 1:
            return letter_names.get(low, raw)

        if low in _EN_WORD_DICT_RU_OVERRIDES and is_ru:
            return _EN_WORD_DICT_RU_OVERRIDES[low]
        if low in _EN_WORD_DICT_UK:
            res = _EN_WORD_DICT_UK[low]
            return _uk_cyr_to_ru_cyr(res) if is_ru else res

        # Абревіатури або слова без голосних (крім поодиноких приголосних) читаємо по літерах
        clean_alpha = low.replace("'", "")
        if low in _EN_SPELL_ACRONYMS or (
            raw.isupper() and 2 <= len(clean_alpha) <= 5 and not any(c in "AEIOUY" for c in raw)
        ) or (len(clean_alpha) <= 3 and not any(c in "aeiouy" for c in clean_alpha)):
            return " ".join(letter_names.get(ch, ch) for ch in clean_alpha)

        res = _phonetic_en_word_to_uk(raw)
        if not res:
            return " ".join(letter_names.get(ch, ch) for ch in clean_alpha)
        return _uk_cyr_to_ru_cyr(res) if is_ru else res

    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", _conv_token, text)


def transliterate_cyrillic_for_latin_tts(text: str) -> str:
    """Перетворює кирилицю в латинську транслітерацію для іншомовних локальних моделей (EN/DE/FR/ES/PL)."""
    if not text or not re.search(r"[А-Яа-яІіЇїЄєҐґЁё]", text):
        return text
    return text.translate(_CYR_TO_LATIN_TABLE)


def adapt_text_for_local_tts(text: str, target_lang: str, uk_to_ru: bool = False) -> str:
    """
    Адаптує текст під скрипт цільової локальної моделі:
    - для uk/ru моделей: перетворює латинські літери/слова у фонетичну кирилицю (і за потреби укр. літери для RU-моделей);
    - для латинських моделей (en/de/fr/es/pl): транслітерує кирилицю в латиницю.
    """
    if not text:
        return ""
    lang_code = (target_lang or "uk").strip().lower()
    out = text
    if lang_code in ("uk", "ru"):
        out = transliterate_latin_for_cyrillic_tts(out, lang=lang_code)
        if uk_to_ru or lang_code == "ru":
            out = _uk_cyr_to_ru_cyr(out)
    elif lang_code in ("en", "de", "fr", "es", "pl"):
        out = transliterate_cyrillic_for_latin_tts(out)
    return out
