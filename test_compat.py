"""
The web interface must run on older hospital PCs.

Baseline: Chrome / Edge 84+, Firefox 78+, Safari 14+ (mid-2020) -- this
includes the last browsers available for Windows 7 and 8.1. These tests fail
if the page starts using something those browsers lack, and check that the
old-browser message itself is plain ES5, so even Internet Explorer can show it.

Feature list: each entry is (pattern, what it is, first supporting versions).
"""
import re
from pathlib import Path

import pytest

HTML = (Path(__file__).parent / "webapp" / "index.html").read_text(encoding="utf-8")
CSS = HTML[HTML.index("<style>"):HTML.index("</style>")]
SCRIPTS = re.findall(r"<script>(.*?)</script>", HTML, flags=re.S)
APP_JS = max(SCRIPTS, key=len)
GUARD_JS = next(s for s in SCRIPTS if "OLD_BROWSER = true" in s)
PROBE_JS = next(s for s in SCRIPTS if "MODERN_OK =" in s)


def _strip_comments(src: str) -> str:
    return re.sub(r"/\*.*?\*/", "", src, flags=re.S)


TOO_NEW_CSS = [
    (r"color-mix\(", "color-mix()", "Chrome 111 / Firefox 113 (2023)"),
    (r"aspect-ratio\s*:", "aspect-ratio", "Chrome 88 / Firefox 89 (2021)"),
    (r"(?<![-\w])inset\s*:", "inset shorthand", "Chrome 87 (2020-11)"),
    (r":has\(", ":has()", "Chrome 105 / Firefox 121"),
    (r"@container", "container queries", "Chrome 105"),
    (r"@layer", "cascade layers", "Chrome 99"),
    (r"\bsubgrid\b", "subgrid", "Chrome 117"),
    (r"text-wrap\s*:", "text-wrap", "Chrome 114"),
    (r"\b(?:dvh|svh|lvh)\b", "dynamic viewport units", "Chrome 108"),
    (r"accent-color", "accent-color", "Chrome 93"),
]

TOO_NEW_JS = [
    (r"\bstructuredClone\(", "structuredClone", "Chrome 98"),
    (r"\.at\(", "Array.prototype.at", "Chrome 92"),
    (r"\.replaceAll\(", "String.replaceAll", "Chrome 85"),
    (r"\.findLast(?:Index)?\(", "findLast", "Chrome 97"),
    (r"Object\.hasOwn\(", "Object.hasOwn", "Chrome 93"),
    (r"Promise\.any\(", "Promise.any", "Chrome 85"),
    (r"(?:\?\?|\|\||&&)=", "logical assignment", "Chrome 85"),
    (r"\.showModal\(", "<dialog>.showModal", "Firefox 98"),
    (r"\.toSorted\(|\.toReversed\(", "change-array-by-copy", "Chrome 110"),
    (r"AbortSignal\.timeout", "AbortSignal.timeout", "Chrome 103"),
    (r"\bimport\s*\(", "dynamic import", "kept out: no modules"),
]


@pytest.mark.parametrize("pattern, name, since", TOO_NEW_CSS)
def test_css_avoids_features_older_browsers_lack(pattern, name, since):
    hits = re.findall(pattern, _strip_comments(CSS))
    assert not hits, f"{name} needs {since}; the baseline is mid-2020 browsers"


@pytest.mark.parametrize("pattern, name, since", TOO_NEW_JS)
def test_js_avoids_features_older_browsers_lack(pattern, name, since):
    hits = re.findall(pattern, APP_JS)
    assert not hits, f"{name} needs {since}; the baseline is mid-2020 browsers"


def test_old_browser_guard_is_plain_es5():
    code = _strip_comments(GUARD_JS)
    for token, name in [("=>", "arrow functions"), ("`", "template literals"),
                        (r"\blet\b", "let"), (r"\bconst\b", "const"),
                        (r"\bclass\b", "classes"), (r"\?\.", "optional chaining")]:
        assert not re.search(token, code), f"guard uses {name}; IE11 cannot run it"


def test_guard_runs_before_the_app_and_blocks_boot():
    # Probe (modern syntax) first, then the ES5 guard, then the app.
    assert HTML.index(PROBE_JS) < HTML.index(GUARD_JS) < HTML.index(APP_JS[:200])
    assert "if (!window.OLD_BROWSER) boot();" in APP_JS
    assert 'id="oldBrowser"' in HTML and "Internet Explorer is not supported" in HTML


def test_no_external_resources():
    # Hospital networks often block the internet, and the CSP forbids it.
    assert not re.search(r"""(?:src|href)\s*=\s*["']https?://""", HTML)
    assert "@import" not in CSS and "fonts.googleapis" not in HTML


def test_text_is_readable_on_old_monitors():
    base = re.search(r"body\{[^}]*font:(\d+)px", CSS)
    assert base and int(base.group(1)) >= 16
    sizes = [float(x) for x in re.findall(r"font-size:(\d+(?:\.\d+)?)px", _strip_comments(CSS))]
    assert min(sizes) >= 11.5, f"smallest text {min(sizes)}px"


def test_guard_does_not_need_eval():
    # The Content-Security-Policy has no 'unsafe-eval': new Function() or eval()
    # in the guard would throw on EVERY browser and lock everyone out.
    for js in (PROBE_JS, GUARD_JS):
        assert "new Function" not in _strip_comments(js)
        assert not re.search(r"eval\(", _strip_comments(js))


def test_probe_uses_the_syntax_the_app_needs():
    for token in ("async function", "await", "?.", "??", "const"):
        assert token in PROBE_JS
