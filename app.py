import streamlit as st
import os
import csv
import asyncio
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

# =====================================================
# EXTRACTOR CLASS
# =====================================================

class OpenAIGlossaryExtractor:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    async def extract(self, text: str) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — синолог и literary переводчик с китайского на русский.\n"
                    "Выделяй имена и значимые термины из китайской художественной прозы.\n\n"
                    "ПРАВИЛА:\n"
                    "- Не включай общеязыковые слова, глаголы и служебные части речи.\n"
                    "- Имена передавай транскрипцией по Палладию.\n"
                    "- Термины переводи нейтральным или устоявшимся русским эквивалентом.\n"
                    "- Примечания пиши КРАТКО и ТОЛЬКО НА РУССКОМ.\n"
                    "- Тип термина указывай ТОЛЬКО одним из допустимых русских значений.\n"
                    "- Верни СТРОГО JSON, без пояснений и лишнего текста.\n\n"
                    "ДОПУСТИМЫЕ ТИПЫ (СТРОГО):\n"
                    "имя, титул, организация, культивация, предмет, понятие, прочее\n"
                ),
            },
            {
                "role": "user",
                "content": f"""
Извлеки глоссарий из китайского текста.

Формат ответа (строго JSON):
[
  {{
    "chinese": "...",
    "russian": "...",
    "type": "имя|титул|организация|культивация|предмет|понятие|прочее",
    "sex": "мужской|женский",
    "notes": "краткое примечание на русском или пусто"
  }}
]

ТЕКСТ:
{text}
""".strip(),
            },
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or "[]"
            data = json_safe_loads(raw)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

def json_safe_loads(raw):
    import json
    # Clean markdown formatting if model accidentally wraps response in code blocks
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())

# =====================================================
# HELPERS
# =====================================================

def split_text(text: str, max_chars: int = 6000) -> List[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def load_existing(glossary_path: Path) -> Dict[str, bool]:
    existing = {}
    if glossary_path.exists():
        with glossary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                zh = (row.get("Chinese") or "").strip()
                if zh:
                    existing[zh] = True
    return existing

def append_rows(glossary_path: Path, rows: List[Dict[str, str]]) -> None:
    glossary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not glossary_path.exists() or glossary_path.stat().st_size == 0

    with glossary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Chinese", "Russian", "Notes", "Type"])
        for r in rows:
            writer.writerow([
                r["Chinese"],
                r["Russian"],
                r["Notes"],
                r["Type"],
            ])

async def process_file(
    path: Path,
    extractor: OpenAIGlossaryExtractor,
    existing: Dict[str, bool],
) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = split_text(text)

    new_rows: List[Dict[str, str]] = []
    for chunk in chunks:
        items = await extractor.extract(chunk)
        for it in items:
            zh = (it.get("chinese") or "").strip()
            ru = (it.get("russian") or "").strip()

            if not zh or zh in existing:
                continue

            new_rows.append({
                "Chinese": zh,
                "Russian": ru,
                "Notes": (it.get("notes") or "").strip(),
                "Type": it.get("type", "прочее"),
            })
            existing[zh] = True

    return new_rows

async def run_glossary_pipeline(input_dir: Path, glossary_path: Path, api_key: str, model: str, status_container):
    extractor = OpenAIGlossaryExtractor(api_key=api_key, model=model)
    existing = load_existing(glossary_path)

    files = sorted(p for p in input_dir.glob("**/*.txt") if p.is_file())
    if not files:
        status_container.error("Не найдено текстовых файлов для сканирования.")
        return False

    all_new: List[Dict[str, str]] = []
    total_files = len(files)

    for idx, fp in enumerate(files, 1):
        status_container.text(f"[{idx}/{total_files}] Сканирование файла: {fp.name}...")
        rows = await process_file(fp, extractor, existing)
        all_new.extend(rows)

    if all_new:
        append_rows(glossary_path, all_new)
        status_container.success(f"Готово! Добавлено {len(all_new)} новых записей в глоссарий.")
    else:
        status_container.info("Сканирование завершено. Новых терминов не найдено.")
    
    return True

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("📖 AI Chinese-Russian Glossary Builder")
st.write("Upload Chinese text files to automatically extract characters, titles, organizations, and key terms into a structured CSV glossary via OpenAI.")

api_key_input = st.text_input("OpenAI API Key", type="password", value="")
model_input = st.text_input("OpenAI Model Name", value="gpt-4o-mini")

uploaded_files = st.file_uploader("Upload chapter .txt files", accept_multiple_files=True, type=["txt"])

if uploaded_files:
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    # Cleanup older runs
    for old_file in input_dir.glob("*.txt"):
        old_file.unlink()

    for uploaded_file in uploaded_files:
        file_path = input_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
    st.success(f"Успешно загружено файлов: {len(uploaded_files)}")

if st.button("Сгенерировать глоссарий"):
    if not api_key_input:
        st.error("Пожалуйста, введите ваш OpenAI API Key!")
    else:
        glossary_file = Path("glossary.csv")
        status_box = st.empty()
        
        # Run async function in Streamlit
        success = asyncio.run(run_glossary_pipeline(Path("input"), glossary_file, api_key_input, model_input, status_box))
        
        if success and glossary_file.exists():
            with open(glossary_file, "rb") as fp:
                st.download_button(
                    label="📦 Скачать готовый глоссарий (glossary.csv)",
                    data=fp,
                    file_name="glossary.csv",
                    mime="text/csv"
                )