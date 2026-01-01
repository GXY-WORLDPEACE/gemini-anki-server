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

app = FastAPI(title="Gemini Share/Text -> Anki TSV")

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
    """
    Robustly extract JSON object from model output.
    Handles cases like:
    - leading explanation lines
    - ```json ... ```
    - extra trailing text
    """
    t = (text or "").strip()
    t = re.sub(r"^\s*```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*```\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Model did not return JSON. First 800 chars:\n" + t[:800])

    json_text = t[start:end + 1]
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from model output: {e}\n"
            f"Extracted JSON (first 800 chars):\n{json_text[:800]}"
        )


def call_deepseek_for_anki_from_conversation(dialogue: List[str]) -> Dict[str, Any]:
    """From conversation paragraphs -> anki_cards JSON"""
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


def call_deepseek_for_anki_from_text(raw_text: str) -> Dict[str, Any]:
    """From pasted raw text -> anki_cards JSON"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY on server environment.")

    raw_text = (raw_text or "").strip()
    if len(raw_text) < 20:
        raise RuntimeError("Text too short. Please paste more content.")

    system_prompt = """
You are an English learning assistant.
Create high-quality Anki flashcards from the given text.

Tasks:
1) Remove duplicate or near-duplicate content (semantic deduplication).
2) Select ONLY English sentences or short paragraphs worth learning.
3) Rewrite slightly if needed to be natural and grammatical English.
4) Keep each card independent.
5) Prefer practical, reusable expressions.

Output JSON ONLY with:
{
  "anki_cards": [
    {"english": "...", "note_cn": "..."},
    ...
  ]
}

Rules:
- Ignore UI noise if any exists.
- Do NOT include very short fillers unless they carry learning value.
- Avoid overly long paragraphs (keep <= ~2-4 sentences).
""".strip()

    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
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
        if not isinstance(c, dict):
            continue
        eng = (c.get("english") or "").strip().replace("\t", " ")
        note = (c.get("note_cn") or "").strip().replace("\t", " ")
        if not eng:
            continue
        if eng in seen:
            continue
        seen.add(eng)
        lines.append(f"{eng}\t{note}")
    return "\n".join(lines)


def tsv_response(tsv_text: str, filename: str) -> Response:
    # mobile-friendly download headers
    return Response(
        content=tsv_text.encode("utf-8"),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ========= Web UI =========
HOME_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
  <title>Gemini → Anki TSV</title>
  <style>
    :root{
      --bg:#0b0c10; --card:#12141b; --text:#e8eaf0; --muted:#a8afc3;
      --border:#2a2f3a; --accent:#4f8cff; --danger:#ff5a6a;
      --shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    @media (prefers-color-scheme: light){
      :root{ --bg:#f6f7fb; --card:#ffffff; --text:#14161f; --muted:#5b6173;
        --border:#e6e8f0; --accent:#2f6bff; --danger:#d92b3a; --shadow: 0 10px 30px rgba(20,22,31,.08);}
    }
    *{box-sizing:border-box}
    body{
      margin:0; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Noto Sans","PingFang SC","Microsoft YaHei",sans-serif;
      background:var(--bg); color:var(--text);
      padding: env(safe-area-inset-top) 14px env(safe-area-inset-bottom);
    }
    .wrap{max-width:720px; margin:22px auto 28px;}
    .title{font-size:22px; font-weight:800; letter-spacing:.2px; margin:6px 0 10px;}
    .subtitle{color:var(--muted); font-size:14px; line-height:1.55; margin:0 0 16px;}
    .card{
      background:var(--card); border:1px solid var(--border); border-radius:18px;
      box-shadow:var(--shadow); padding:16px;
      margin-bottom:14px;
    }
    label{display:block; font-weight:700; font-size:14px; margin:0 0 8px;}
    input, textarea{
      width:100%;
      font-size:16px; /* iOS 16px+ prevents zoom */
      padding:14px 14px;
      border-radius:14px;
      border:1px solid var(--border);
      background:transparent; color:var(--text);
      outline:none;
    }
    textarea{resize:vertical; min-height:150px;}
    input:focus, textarea:focus{border-color:rgba(79,140,255,.8); box-shadow:0 0 0 4px rgba(79,140,255,.18);}
    .btn{
      width:100%; margin-top:12px;
      border:0; border-radius:14px;
      padding:14px 14px;
      font-size:16px; font-weight:800;
      background:var(--accent); color:white;
      cursor:pointer;
      display:flex; align-items:center; justify-content:center; gap:10px;
    }
    .btn:disabled{opacity:.65; cursor:not-allowed;}
    .hint{
      margin-top:12px; color:var(--muted); font-size:13px; line-height:1.55;
    }
    .steps{
      margin:12px 0 0; padding:0 0 0 18px; color:var(--muted); font-size:13px; line-height:1.6;
    }
    .pill{
      display:inline-flex; align-items:center; gap:8px;
      margin-top:10px;
      padding:10px 12px; border:1px dashed var(--border); border-radius:14px;
      color:var(--muted); font-size:13px;
    }
    .spinner{
      width:16px; height:16px; border-radius:999px;
      border:2px solid rgba(255,255,255,.45);
      border-top-color:white;
      animation:spin .8s linear infinite;
    }
    @keyframes spin{to{transform:rotate(360deg)}}
    .footer{margin-top:10px; color:var(--muted); font-size:12px; line-height:1.55;}
    code{background:rgba(127,127,127,.15); padding:2px 6px; border-radius:8px;}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">生成 Anki TSV</div>
    <p class="subtitle">
      两种方式：① Gemini 分享链接 ② 直接粘贴文本。生成后会下载 <code>.tsv</code>（Tab 分隔，可导入 Anki）。
    </p>

    <div class="card">
      <form id="f_link" method="post" action="/download_tsv">
        <label for="url">方式 A：Gemini 分享链接</label>
        <input id="url" name="url" inputmode="url" autocomplete="off"
               placeholder="https://gemini.google.com/share/..." required>
        <button id="btn_link" class="btn" type="submit">
          <span id="btnText_link">用链接生成并下载 TSV</span>
        </button>

        <div id="status_link" class="pill" style="display:none;">
          <div class="spinner" aria-hidden="true"></div>
          <div>生成中…（抓取页面 + 总结句子），可能需要 10–60 秒</div>
        </div>

        <div class="hint">
          建议：分享链接必须是公开可访问的页面（无需登录）。
          <ol class="steps">
            <li>Gemini 对话 → 分享 → 复制链接</li>
            <li>粘贴到这里 → 点击生成</li>
          </ol>
        </div>
      </form>
    </div>

    <div class="card">
      <form id="f_text" method="post" action="/download_tsv_text">
        <label for="text">方式 B：直接粘贴一段文字</label>
        <textarea id="text" name="text"
                  placeholder="在这里粘贴你的英文/对话/文章片段…（建议 200–4000 字）"
                  required></textarea>

        <button id="btn_text" class="btn" type="submit">
          <span id="btnText_text">用文本生成并下载 TSV</span>
        </button>

        <div id="status_text" class="pill" style="display:none;">
          <div class="spinner" aria-hidden="true"></div>
          <div>生成中…（筛选可学句子 + 去重），可能需要 5–40 秒</div>
        </div>

        <div class="footer">
          Anki 导入：分隔符选 Tab；第一列=English，第二列=中文备注。
        </div>
      </form>
    </div>
  </div>

  <script>
    // Auto focus for mobile convenience
    window.addEventListener('load', () => {
      const u = document.getElementById('url');
      if (u) u.focus();
    });

    // Show loading state on submit (link)
    document.getElementById('f_link').addEventListener('submit', () => {
      const btn = document.getElementById('btn_link');
      const status = document.getElementById('status_link');
      const btnText = document.getElementById('btnText_link');
      btn.disabled = true;
      btnText.textContent = '生成中…';
      status.style.display = 'inline-flex';
    });

    // Show loading state on submit (text)
    document.getElementById('f_text').addEventListener('submit', () => {
      const btn = document.getElementById('btn_text');
      const status = document.getElementById('status_text');
      const btnText = document.getElementById('btnText_text');
      btn.disabled = true;
      btnText.textContent = '生成中…';
      status.style.display = 'inline-flex';
    });
  </script>
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
        result = call_deepseek_for_anki_from_conversation(dialogue)
        cards = result.get("anki_cards", [])
        if not isinstance(cards, list):
            raise RuntimeError("DeepSeek returned invalid format: anki_cards is not a list")

        tsv = to_tsv(cards)
        return tsv_response(tsv, filename="anki_cards.tsv")
    except Exception as e:
        return HTMLResponse(
            f"<h3>Error</h3><pre>{str(e)}</pre><p><a href='/'>Back</a></p>",
            status_code=500
        )


@app.post("/download_tsv_text")
def download_tsv_text(text: str = Form(...)):
    try:
        result = call_deepseek_for_anki_from_text(text)
        cards = result.get("anki_cards", [])
        if not isinstance(cards, list):
            raise RuntimeError("DeepSeek returned invalid format: anki_cards is not a list")

        tsv = to_tsv(cards)
        return tsv_response(tsv, filename="anki_cards_from_text.tsv")
    except Exception as e:
        return HTMLResponse(
            f"<h3>Error</h3><pre>{str(e)}</pre><p><a href='/'>Back</a></p>",
            status_code=500
        )


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "model": DEEPSEEK_MODEL, "has_key": bool(DEEPSEEK_API_KEY)})
