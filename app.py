#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
from typing import List, Dict, Any
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, Response, JSONResponse
import requests
from playwright.sync_api import sync_playwright

# ========= Config =========
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

app = FastAPI(title="Gemini Share -> Anki TSV")

# ========= Filters =========
DELETE_LINE_PATTERNS = [
    # header/meta/share info
    r"^About Gemini$",
    r"^Gemini App$",
    r"^Subscriptions$",
    r"^For Business$",
    r"^Opens in a new window$",
    r"^Laptop On Desk$",
    r"^https://gemini\.google\.com/share/.*$",
    r"^Created with.*$",
    r"^Published.*$",

    # cookie / ui
    r"^Sign in$",
    r"^Before you continue to Google$",
    r"^We use cookies and data.*$",
    r"^Deliver and maintain Google services$",
    r"^Track outages.*$",
    r"^Measure audience engagement.*$",
    r"^If you choose to.*$",
    r"^Non-personalized content.*$",
    r"^Personalized content.*$",
    r"^Select .*privacy.*$",
    r"^Reject all$",
    r"^Accept all$",
    r"^More options$",
    r"^Privacy Policy$",
    r"^Terms of Service$",
]
DELETE_RE = [re.compile(p, re.IGNORECASE) for p in DELETE_LINE_PATTERNS]


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def should_delete_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return any(r.match(line) for r in DELETE_RE)


def split_into_paragraphs(lines: List[str]) -> List[str]:
    paras, buf = [], []
    for line in lines:
        if line.strip() == "":
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        paras.append("\n".join(buf).strip())
    return [p for p in paras if p]


def handle_cookie(page):
    for name in ["Accept all", "Reject all", "全部接受", "全部拒绝"]:
        try:
            btn = page.get_by_role("button", name=name)
            if btn.count() > 0:
                btn.first.click(timeout=5000)
                time.sleep(1.5)
                return
        except Exception:
            pass


def fetch_dialogue_from_share(url: str) -> List[str]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(locale="en-US")
        page.goto(url, wait_until="domcontentloaded")
        handle_cookie(page)
        time.sleep(2.5)

        # scroll to load more
        for _ in range(7):
            page.mouse.wheel(0, 1400)
            time.sleep(0.4)

        raw = ""
        for sel in ["main", "article", "div[role='main']"]:
            try:
                t = page.locator(sel).first.inner_text(timeout=5000)
                if t and len(t.strip()) > 50:
                    raw = t
                    break
            except Exception:
                pass

        if not raw:
            raw = page.evaluate("() => document.body.innerText") or ""

        browser.close()

    raw = clean_text(raw)
    kept_lines: List[str] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            kept_lines.append("")
            continue
        if should_delete_line(line):
            continue
        kept_lines.append(line)

    paragraphs = split_into_paragraphs(kept_lines)

    # exact dedup (keep order)
    final_dialogue = []
    seen = set()
    for p in paragraphs:
        if p in seen:
            continue
        seen.add(p)
        final_dialogue.append(p)
    return final_dialogue


def extract_json_from_text(text: str) -> Dict[str, Any]:
    t = text.strip()
    t = re.sub(r"^\s*```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*```\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Model did not return JSON. First 600 chars:\n" + t[:600])

    json_text = t[start:end + 1]
    return json.loads(json_text)


def call_deepseek_for_anki(dialogue: List[str]) -> Dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY on server environment.")

    system_prompt = """
You are an English learning assistant.
Create high-quality Anki flashcards from the given conversation text.

Tasks:
1) Remove duplicate or near-duplicate content (semantic deduplication).
2) Select ONLY English sentences or short paragraphs worth learning.
3) Rewrite slightly if needed to be natural and grammatical English.
4) Keep each card independent (does not rely heavily on context).
5) Prefer practical, reusable expressions.

Output JSON ONLY with:
{
  "anki_cards": [
    {"english": "...", "note_cn": "..."},
    ...
  ]
}

Rules:
- Do NOT include navigation/header/cookie/UI text.
- Do NOT include very short fillers like "Yes", "Okay", "Great!" unless they carry learning value.
- Avoid overly long paragraphs (keep <= ~2-4 sentences).
""".strip()

    user_content = "Conversation:\n\n" + "\n\n".join(dialogue)

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API error HTTP {resp.status_code}: {resp.text[:800]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return extract_json_from_text(content)


def to_tsv(cards: List[Dict[str, str]]) -> str:
    lines = []
    seen = set()
    for c in cards:
        eng = (c.get("english") or "").strip().replace("\t", " ")
        note = (c.get("note_cn") or "").strip().replace("\t", " ")
        if not eng:
            continue
        if eng in seen:
            continue
        seen.add(eng)
        lines.append(f"{eng}\t{note}")
    return "\n".join(lines)


# ========= Web UI =========
HOME_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Gemini Share → Anki TSV</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto; max-width: 720px; margin: 24px auto; padding: 0 14px; }
    input { width: 100%; padding: 12px; font-size: 16px; }
    button { padding: 12px 14px; font-size: 16px; margin-top: 10px; width: 100%; }
    .hint { color: #444; font-size: 14px; margin-top: 10px; line-height: 1.5; }
    .warn { color: #b00; font-size: 14px; margin-top: 10px; }
  </style>
</head>
<body>
  <h2>Gemini Share → Anki TSV</h2>
  <form method="post" action="/download_tsv">
    <label>Paste Gemini share link:</label><br/>
    <input name="url" placeholder="https://gemini.google.com/share/..." required />
    <button type="submit">Generate & Download TSV</button>
  </form>
  <p class="hint">
    输出 TSV 可直接导入 Anki（Tab 分隔）：第一列=English，第二列=中文备注。<br/>
    服务器会自动：抓取分享页 → 清洗无关内容 → 去重 → DeepSeek 精选句子 → 下载 TSV。
  </p>
  <p class="warn">
    注意：链接必须是公开可访问的 Gemini share 页面。
  </p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HOME_HTML


@app.post("/download_tsv")
def download_tsv(url: str = Form(...)):
    try:
        dialogue = fetch_dialogue_from_share(url)
        result = call_deepseek_for_anki(dialogue)
        cards = result.get("anki_cards", [])
        if not isinstance(cards, list):
            raise RuntimeError("DeepSeek returned invalid format: anki_cards is not a list")

        tsv = to_tsv(cards)
        filename = "anki_cards.tsv"
        return Response(
            content=tsv.encode("utf-8"),
            media_type="text/tab-separated-values; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        # 返回可读的错误（手机端也能直接看到）
        return HTMLResponse(
            f"<h3>Error</h3><pre>{str(e)}</pre><p><a href='/'>Back</a></p>",
            status_code=500
        )


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "model": DEEPSEEK_MODEL, "has_key": bool(DEEPSEEK_API_KEY)})
