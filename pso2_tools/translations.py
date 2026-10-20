"""Blender-language UI catalogs. Stored names and identifiers stay unchanged."""
import json
import re
from functools import lru_cache
from pathlib import Path

CONTEXT = "PSO2Tools"
LOCALES = ("en_US", "ko_KR", "ja_JP", "zh_HANS", "zh_HANT")
ALIASES = {"zh_CN": "zh_HANS", "zh_TW": "zh_HANT"}
_OWNER = __package__ + ".translations"


@lru_cache(maxsize=None)
def catalog(locale):
    locale = ALIASES.get(locale, locale)
    if locale not in LOCALES:
        locale = "en_US"
    path = Path(__file__).with_name("locales") / (locale + ".json")
    with path.open(encoding="utf8") as stream:
        return json.load(stream)


def iface(message):
    import bpy
    return bpy.app.translations.pgettext_iface(message, CONTEXT)


def report(message):
    import bpy
    return bpy.app.translations.pgettext_rpt(message, CONTEXT)


def status(message):
    """Translate display-time status without localizing stored scene data."""
    match = re.fullmatch(r"FLTD 33 / (\d+) particles / Scene inputs", message)
    if match:
        return iface("FLTD 33 / {count} particles / Scene inputs").format(count=match[1])
    match = re.fullmatch(r"(\d+) facial bones posed \(selected rig only\)", message)
    if match:
        return iface("{count} facial bones posed (selected rig only)").format(count=match[1])
    if message.startswith("Unavailable: "):
        return iface("Unavailable: {reason}").format(reason=message[13:])
    return iface(message)


def register():
    import bpy
    unregister()
    catalog.cache_clear()
    dictionaries = {
        locale: {(CONTEXT, key): value for key, value in catalog(locale).items()}
        for locale in LOCALES
    }
    for alias, locale in ALIASES.items():
        dictionaries[alias] = dictionaries[locale]
    bpy.app.translations.register(_OWNER, dictionaries)


def unregister():
    import bpy
    try:
        bpy.app.translations.unregister(_OWNER)
    except ValueError:
        pass
    catalog.cache_clear()
