import streamlit as st
import re
import io
import zipfile
import pandas as pd

st.set_page_config(page_title="Indonesian G2P + Syllabification (Corpus Mode)", layout="centered")

st.title("Indonesian → IPA Transcription (Corpus Mode)")
st.caption("⚠️ Heuristic, rule-based transcription. May not be fully accurate. Use with caution.")

# ============================================================
# Default exception lexicon for 'e' (can be overridden by user)
# ============================================================
DEFAULT_LEX_E = {
    "emas": "əmas",
    "merah": "mɛrah",
    "berak": "bɛrak",
    "enak": "enak",
    "empat": "əmpat",
    "enam": "ənam",
}

VOWELS = "aiueoəɛɔ"

# ============================================================
# Utility: detect XML tags or symbol-heavy tokens
# ============================================================
def is_xml_or_symbol(token: str) -> bool:
    if token.startswith("<") and token.endswith(">"):
        return True
    if re.search(r"[<>{}=\\/]", token):
        return True
    return False


# ============================================================
# Phonological handlers
# ============================================================
def handle_o(text):
    # o before consonant -> ɔ
    text = re.sub(r"o(?=[bcdfghjklmnpqrstvwxyz])", "ɔ", text)
    # o before vowel -> o
    text = re.sub(r"o(?=[aiueo])", "o", text)
    # o at end of word -> o
    text = re.sub(r"o\b", "o", text)
    return text


def handle_e_word(word, lex_e):
    # Exception lexicon first
    if word in lex_e:
        return lex_e[word]

    w = word

    # Common prefixes -> schwa
    w = re.sub(r"^(be|me|pe|ke|se|te)(?=[bcdfghjklmnpqrstvwxyz])",
               lambda m: m.group(1)[0] + "ə", w)

    # Heavy/expressive tendency -> ɛ
    w = re.sub(r"e(?=[rktp])", "ɛ", w)

    # Word-initial e + consonant -> e
    w = re.sub(r"^e(?=[bcdfghjklmnpqrstvwxyz])", "e", w)

    # Remaining e -> schwa
    w = re.sub(r"e", "ə", w)

    return w


# ============================================================
# Core G2P rules (applied after e/o handling)
# ============================================================
RULES = [
    # Diphthongs
    (r"ai", "ai̯"),
    (r"au", "au̯"),
    (r"oi", "oi̯"),

    # Digraphs
    (r"ng", "ŋ"),
    (r"ny", "ɲ"),
    (r"sy", "ʃ"),
    (r"kh", "x"),
    (r"dz", "dz"),

    # Letters
    (r"c", "tʃ"),
    (r"j", "dʒ"),
    (r"y", "j"),
    (r"x", "ks"),
    (r"q", "k"),

    # Consonants (mostly identical)
    (r"b", "b"),
    (r"d", "d"),
    (r"f", "f"),
    (r"g", "g"),
    (r"h", "h"),
    (r"k", "k"),
    (r"l", "l"),
    (r"m", "m"),
    (r"n", "n"),
    (r"p", "p"),
    (r"r", "r"),
    (r"s", "s"),
    (r"t", "t"),
    (r"v", "v"),
    (r"w", "w"),
    (r"z", "z"),
]


# ============================================================
# Final /k/ -> ʔ
# ============================================================
def handle_final_k(ipa):
    return re.sub(r"k\b", "ʔ", ipa)


# ============================================================
# Onset-maximising, coda-aware syllabifier
# ============================================================
def syllabify_onset_max(word):
    """
    Onset-maximising heuristic for Indonesian:
    - V.V -> split
    - VCV -> split before C (C goes to onset)
    - VCCV -> split C.C
    """
    chars = list(word)
    syllables = []
    current = ""
    i = 0

    while i < len(chars):
        c = chars[i]
        current += c

        if c in VOWELS:
            if i + 1 < len(chars):
                nxt = chars[i + 1]

                # V V -> boundary
                if nxt in VOWELS:
                    syllables.append(current)
                    current = ""

                # V C V -> boundary before C
                elif nxt not in VOWELS and i + 2 < len(chars) and chars[i + 2] in VOWELS:
                    syllables.append(current)
                    current = ""

                # V C C V -> split C.C
                elif (
                    nxt not in VOWELS
                    and i + 2 < len(chars)
                    and chars[i + 2] not in VOWELS
                    and i + 3 < len(chars)
                    and chars[i + 3] in VOWELS
                ):
                    current += nxt
                    syllables.append(current)
                    current = ""
                    i += 1  # consume extra consonant

        i += 1

    if current:
        syllables.append(current)

    return ".".join(syllables)


# ============================================================
# Token-level pipeline (exception → rule)
# ============================================================
def indo_to_ipa_token(token, user_exceptions, lex_e, syllabify=True):
    w = token.lower()

    # 1. User exception has absolute priority
    if w in user_exceptions:
        ipa = user_exceptions[w]
        return ipa

    # 2. Rule-based pipeline
    w = handle_e_word(w, lex_e)
    w = handle_o(w)

    ipa = w
    for pattern, repl in RULES:
        ipa = re.sub(pattern, repl, ipa)

    ipa = handle_final_k(ipa)

    if syllabify:
        ipa = syllabify_onset_max(ipa)

    return ipa


# ============================================================
# Input processing
# ============================================================
def tokenize_input(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) == 1 and " " in lines[0]:
        return lines[0].split()
    return lines


def process_text(text, user_exceptions, lex_e, syllabify=True):
    tokens = tokenize_input(text)
    output_lines = []

    for tok in tokens:
        if is_xml_or_symbol(tok):
            continue
        ipa = indo_to_ipa_token(tok, user_exceptions, lex_e, syllabify=syllabify)
        output_lines.append(f"{tok}\t{ipa}")

    return output_lines


# ============================================================
# Exception handling (user-defined)
# ============================================================
def parse_exception_lines(text):
    """
    Expect: word<TAB>transcription per line
    """
    exceptions = {}
    lines = text.strip().splitlines()
    for line in lines:
        if "\t" in line:
            word, trans = line.split("\t", 1)
            word = word.strip().lower()
            trans = trans.strip()
            if word and trans:
                exceptions[word] = trans
    return exceptions


def load_exceptions_from_file(file):
    name = file.name.lower()
    if name.endswith(".txt") or name.endswith(".tsv"):
        content = file.read().decode("utf-8", errors="ignore")
        return parse_exception_lines(content)

    elif name.endswith(".xlsx"):
        df = pd.read_excel(file, header=None)
        exceptions = {}
        for _, row in df.iterrows():
            if len(row) >= 2 and pd.notna(row[0]) and pd.notna(row[1]):
                word = str(row[0]).strip().lower()
                trans = str(row[1]).strip()
                if word and trans:
                    exceptions[word] = trans
        return exceptions

    return {}


# ============================================================
# UI: Exception input
# ============================================================
st.subheader("1) User-defined exceptions (priority over rules)")

exception_mode = st.radio("Exception input method:", ["Direct input", "Upload file"], horizontal=True)

user_exceptions = {}

if exception_mode == "Direct input":
    exc_text = st.text_area(
        "Enter exceptions (word<TAB>transcription), one per line:",
        height=150,
        placeholder="bapak\tbapak\nora\tora"
    )
    if exc_text.strip():
        user_exceptions = parse_exception_lines(exc_text)

else:
    exc_file = st.file_uploader(
        "Upload exception file (txt, tsv, xlsx)",
        type=["txt", "tsv", "xlsx"]
    )
    if exc_file:
        user_exceptions = load_exceptions_from_file(exc_file)

st.caption(f"Loaded {len(user_exceptions)} user exception(s). These will be applied first.")


# ============================================================
# UI: Main input
# ============================================================
st.subheader("2) Main input")

input_mode = st.radio("Input method:", ["Direct text input", "Upload file(s)"], horizontal=True)
syllable_mode = st.checkbox("Enable syllabification (.)", value=True)
preview_limit = 10

# Lexicon for e-handling (can be extended later)
LEX_E = DEFAULT_LEX_E.copy()

# ------------------------
# Direct text input
# ------------------------
if input_mode == "Direct text input":
    input_text = st.text_area(
        "Input (one token per line, or plain text):",
        height=200,
        placeholder="merah\nenak\nemas\nbapak\nora"
    )

    if st.button("Transcribe"):
        if input_text.strip():
            output_lines = process_text(input_text, user_exceptions, LEX_E, syllabify=syllable_mode)

            st.subheader("Preview (first 10 tokens):")
            st.code("\n".join(output_lines[:preview_limit]), language="text")

            full_output = "\n".join(output_lines)

            st.download_button(
                label="Download result",
                data=full_output,
                file_name="phon-output.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please enter text first.")


# ------------------------
# File upload mode
# ------------------------
else:
    uploaded_files = st.file_uploader(
        "Upload one or more files (txt, tsv, xlsx)",
        type=["txt", "tsv", "xlsx"],
        accept_multiple_files=True
    )

    if st.button("Process file(s)"):
        if uploaded_files:
            results = {}

            for file in uploaded_files:
                name = file.name.lower()

                if name.endswith(".txt") or name.endswith(".tsv"):
                    content = file.read().decode("utf-8", errors="ignore")
                    output_lines = process_text(content, user_exceptions, LEX_E, syllabify=syllable_mode)

                elif name.endswith(".xlsx"):
                    df = pd.read_excel(file, header=None)
                    tokens = []
                    for col in df.columns:
                        tokens.extend(df[col].dropna().astype(str).tolist())
                    content = "\n".join(tokens)
                    output_lines = process_text(content, user_exceptions, LEX_E, syllabify=syllable_mode)

                else:
                    continue

                results[file.name] = "\n".join(output_lines)

            # Preview from first file
            first_name = list(results.keys())[0]
            first_lines = results[first_name].splitlines()

            st.subheader(f"Preview (first 10 tokens) from {first_name}:")
            st.code("\n".join(first_lines[:preview_limit]), language="text")

            # Single file -> direct download
            if len(results) == 1:
                orig_name = list(results.keys())[0]
                out_name = f"phon-{orig_name}"

                st.download_button(
                    label="Download result",
                    data=results[orig_name],
                    file_name=out_name,
                    mime="text/plain"
                )

            # Multiple files -> zip
            else:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, content in results.items():
                        out_name = f"phon-{fname}"
                        zf.writestr(out_name, content)

                zip_buffer.seek(0)

                st.download_button(
                    label="Download all results (ZIP)",
                    data=zip_buffer,
                    file_name="phon-results.zip",
                    mime="application/zip"
                )
        else:
            st.warning("Please upload at least one file.")


# ============================================================
# Notes
# ============================================================
with st.expander("Technical notes"):
    st.markdown("""
- **User exceptions are applied first** (absolute priority).
- Exception format: **word<TAB>transcription**.
- Works for **dialect research** (e.g. Banyumas *bapak* → *bapak*).
- **Final /k/ → ʔ** is applied in rule mode.
- **Syllabification** uses onset-maximising, coda-aware heuristics.
- **e** handled as **/ə, e, ɛ/** (heuristic + base lexicon).
- **o** handled as **/o ~ ɔ/** (structural heuristic).
- **XML tags and symbol-heavy tokens are skipped**.
- Output format: **word<TAB>transcription**
- File naming: **file.txt → phon-file.txt**
""")
