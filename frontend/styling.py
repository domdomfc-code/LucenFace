"""CSS và tiện ích HTML nhỏ cho Streamlit."""
from __future__ import annotations

import streamlit as st

from frontend.config import BG, BLUE, TEXT


def inject_app_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --blue: {BLUE};
            --bg: {BG};
            --card: #ffffff;
            --text: {TEXT};
            --muted: rgba(84, 97, 108, 0.78);
            --border: rgba(2, 6, 23, 0.10);
            --shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
          }}

          .stApp {{
            color-scheme: light;
            background:
              radial-gradient(1000px 500px at 20% 0%, rgba(0, 91, 196, 0.14), rgba(0,0,0,0) 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(56, 189, 248, 0.14), rgba(0,0,0,0) 60%),
              var(--bg);
          }}

          section.main > div {{
            padding-top: 1.25rem;
          }}
          section.main .stMarkdown p,
          section.main .stMarkdown li {{
            color: var(--text);
          }}

          /* Không ép width sidebar bằng !important: Streamlit khi thu gọn cần co width ~0;
             nếu không, vùng trái vẫn “chiếm chỗ” và cột chính bị ép sang phải / hẹp. */

          /* Không ẩn cả header — Streamlit cần vùng này để mở lại sidebar khi thu gọn */
          header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, 0.55) !important;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
          }}
          .stDeployButton {{
            display: none !important;
          }}
          footer {{
            visibility: hidden;
            height: 0px;
          }}

          .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: rgba(255,255,255,0.75);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
          }}
          .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 900;
            color: var(--text);
            letter-spacing: -0.02em;
          }}
          .brand-badge {{
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--blue), #38bdf8);
            box-shadow: 0 10px 22px rgba(0,91,196,0.20);
          }}
          .brand-title {{
            font-size: 1.05rem;
            line-height: 1.1;
          }}
          .brand-sub {{
            font-size: 0.8rem;
            color: var(--muted);
            font-weight: 700;
            margin-top: 1px;
          }}
          .top-actions {{
            display: flex;
            gap: 8px;
            align-items: center;
            color: var(--muted);
            font-weight: 700;
          }}
          .pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: rgba(255,255,255,0.9);
          }}

          .app-title {{
            font-size: 1.8rem;
            font-weight: 900;
            color: var(--text);
            margin: 0.35rem 0 0.2rem 0;
            letter-spacing: -0.03em;
          }}
          .app-subtitle-wrap {{
            max-width: 42rem;
            margin-bottom: 1rem;
          }}
          .app-subtitle {{
            color: var(--muted);
            margin: 0;
            font-weight: 600;
            line-height: 1.55;
            font-size: 0.98rem;
          }}
          .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 14px 10px 14px;
            box-shadow: var(--shadow);
          }}
          .card-soft {{
            background: rgba(255,255,255,0.85);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
          }}
          .badge-ok {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            background: rgba(22,163,74,0.12);
            color: #166534;
            font-weight: 700;
            font-size: 0.85rem;
          }}
          .badge-fail {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            background: rgba(220,38,38,0.12);
            color: #991b1b;
            font-weight: 700;
            font-size: 0.85rem;
          }}
          .small-note {{
            font-size: 0.9rem;
            opacity: 0.85;
          }}
          .checklist {{
            margin-top: 6px;
            padding-left: 0px;
          }}
          .check-item {{
            display: flex;
            gap: 8px;
            margin: 4px 0px;
            align-items: baseline;
          }}
          .check-name {{
            font-weight: 700;
            color: var(--text);
            min-width: 160px;
          }}
          .check-msg {{
            color: var(--muted);
            opacity: 0.95;
          }}
          .muted {{
            color: var(--muted);
          }}

          /* Một vùng upload chính — giao diện hero tối, căn giữa (theo mẫu “Chọn ảnh”). */
          section.main [data-testid="stFileUploader"] {{
            margin-top: 0 !important;
          }}
          section.main [data-testid="stFileUploader"] > div {{
            background: #1e1e24 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-top: 1px dashed rgba(255, 255, 255, 0.2) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 0 0 16px 16px !important;
            padding: 1.1rem 1.25rem 1.2rem !important;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.14) !important;
          }}
          section.main [data-testid="stFileUploader"] label {{
            color: rgba(255, 255, 255, 0.88) !important;
            font-weight: 700 !important;
          }}
          section.main [data-testid="stFileUploader"] span[class*="uploadedFile"] {{
            color: rgba(255, 255, 255, 0.9) !important;
          }}
          section.main [data-testid="stFileUploader"] small {{
            color: rgba(255, 255, 255, 0.5) !important;
            font-weight: 600 !important;
          }}
          section.main [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
          section.main [data-testid="stFileUploader"] [data-testid="stCaption"] {{
            color: rgba(255, 255, 255, 0.62) !important;
          }}
          section.main [data-testid="stFileUploader"] button {{
            background: var(--blue) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 9999px !important;
            font-weight: 800 !important;
            padding: 0.55rem 1.35rem !important;
            box-shadow: 0 8px 22px rgba(0, 91, 196, 0.35) !important;
          }}
          section.main [data-testid="stFileUploader"] button::before {{
            content: "+ ";
            font-weight: 900;
            margin-right: 0.15rem;
          }}
          section.main [data-testid="stFileUploader"] button:hover {{
            filter: brightness(1.08);
          }}

          section.main [data-testid="column"]:has(.p2c-upload-hero-top) {{
            margin-bottom: 0.35rem;
          }}
          .p2c-upload-hero-top {{
            text-align: center;
            background: #1e1e24;
            color: rgba(255, 255, 255, 0.92);
            border-radius: 16px 16px 0 0;
            padding: 2rem 1.5rem 1.15rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-bottom: none;
            margin-bottom: 0;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.1);
          }}
          .p2c-upload-hero-top .p2c-upload-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 64px;
            height: 64px;
            border-radius: 16px;
            background: #374151;
            margin-bottom: 1rem;
          }}
          .p2c-upload-tagline-scroll {{
            overflow-x: auto;
            overflow-y: hidden;
            -webkit-overflow-scrolling: touch;
            text-align: center;
            margin: 0 -0.35rem;
            padding: 0 0.35rem 0.15rem;
            scrollbar-width: thin;
          }}
          .p2c-upload-hero-top .p2c-upload-tagline {{
            display: inline-block;
            margin: 0;
            font-size: clamp(0.78rem, 1.1vw + 0.65rem, 0.95rem);
            line-height: 1.45;
            color: rgba(255, 255, 255, 0.72);
            font-weight: 600;
            white-space: nowrap;
            max-width: none;
          }}
          section.main [data-testid="column"]:has(.p2c-upload-hero-top) iframe {{
            border-radius: 0 0 16px 16px !important;
          }}
          section.main [data-testid="column"]:has(.p2c-upload-hero-top) [data-testid="stIFrame"] {{
            margin-bottom: 0.25rem;
          }}
          .p2c-upload-empty {{
            text-align: center;
            padding: 0.85rem 1rem;
            margin-top: 0.75rem;
            border-radius: 12px;
            background: rgba(0, 91, 196, 0.12);
            color: var(--text);
            font-weight: 700;
            font-size: 0.92rem;
            border: 1px solid rgba(0, 91, 196, 0.2);
          }}

          /* Hàng ảnh mẫu kiểu remove.bg: chữ phẳng + thumbnail */
          div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker) {{
            align-items: center !important;
          }}
          .p2c-try-light {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 88px;
            padding: 0.2rem 0.75rem 0.2rem 0;
          }}
          .p2c-try-light-title {{
            margin: 0 0 0.25rem 0;
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -0.02em;
            line-height: 1.15;
          }}
          .p2c-try-light-sub {{
            margin: 0;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--muted);
            line-height: 1.4;
            max-width: 22rem;
          }}
          /* Ảnh mẫu: bo góc rõ — overflow trên wrapper để iframe bị clip đúng hình */
          section.main [data-testid="column"]:has(.p2c-sample-thumb-marker) [data-testid="stIFrame"] {{
            width: 88px !important;
            max-width: 88px !important;
            min-width: 88px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.12),
              0 0 0 1px rgba(15, 23, 42, 0.06) !important;
          }}
          section.main [data-testid="column"]:has(.p2c-sample-thumb-marker) iframe {{
            border-radius: 14px !important;
            display: block !important;
          }}

          /* Chỉ mobile: lưới 2 hàng (chữ full / 4 ảnh) + ảnh nhỏ hơn — PC không đụng */
          @media (max-width: 768px) {{
            /* Ảnh mẫu nhỏ hơn trên màn hẹp */
            section.main [data-testid="column"]:has(.p2c-sample-thumb-marker) [data-testid="stIFrame"] {{
              width: 60px !important;
              max-width: 60px !important;
              min-width: 60px !important;
              height: 60px !important;
            }}
            section.main [data-testid="column"]:has(.p2c-sample-thumb-marker) iframe {{
              max-width: 60px !important;
              max-height: 60px !important;
            }}
            /* Trường A: HorizontalBlock > Column (trực tiếp) */
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker):has(> [data-testid="column"]) {{
              display: grid !important;
              grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
              width: 100% !important;
              max-width: min(calc(100vw - 1.25rem), 320px) !important;
              margin-left: auto !important;
              margin-right: auto !important;
              gap: 0.35rem 0.2rem !important;
              justify-items: center !important;
            }}
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker):has(> [data-testid="column"]) > [data-testid="column"]:has(.p2c-try-light) {{
              grid-column: 1 / -1 !important;
              justify-self: stretch !important;
              width: 100% !important;
              min-width: 0 !important;
            }}
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker):has(> [data-testid="column"]) > [data-testid="column"]:has(.p2c-sample-thumb-marker) {{
              min-width: 0 !important;
              width: 100% !important;
              max-width: 72px !important;
            }}
            /* Trường B: HorizontalBlock > div bọc > Column */
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker) > div:has([data-testid="column"]) {{
              display: grid !important;
              grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
              width: 100% !important;
              max-width: min(calc(100vw - 1.25rem), 320px) !important;
              margin-left: auto !important;
              margin-right: auto !important;
              gap: 0.35rem 0.2rem !important;
              justify-items: center !important;
            }}
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker) > div:has([data-testid="column"]) > [data-testid="column"]:has(.p2c-try-light) {{
              grid-column: 1 / -1 !important;
              justify-self: stretch !important;
              width: 100% !important;
              min-width: 0 !important;
            }}
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker) > div:has([data-testid="column"]) > [data-testid="column"]:has(.p2c-sample-thumb-marker) {{
              min-width: 0 !important;
              width: 100% !important;
              max-width: 72px !important;
            }}
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker) .p2c-try-light {{
              align-items: center;
              text-align: center;
              padding: 0.35rem 0.5rem 0.45rem;
              min-height: auto;
              max-width: 100%;
              margin-left: auto;
              margin-right: auto;
            }}
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker) .p2c-try-light-sub {{
              max-width: 100%;
            }}
          }}
          @media (max-width: 400px) {{
            section.main [data-testid="column"]:has(.p2c-sample-thumb-marker) [data-testid="stIFrame"] {{
              width: 52px !important;
              max-width: 52px !important;
              min-width: 52px !important;
              height: 52px !important;
            }}
            section.main [data-testid="column"]:has(.p2c-sample-thumb-marker) iframe {{
              max-width: 52px !important;
              max-height: 52px !important;
            }}
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker):has(> [data-testid="column"]),
            section.main div[data-testid="stHorizontalBlock"]:has(.p2c-try-light):has(.p2c-sample-thumb-marker) > div:has([data-testid="column"]) {{
              max-width: min(calc(100vw - 1rem), 280px) !important;
            }}
          }}

          .p2c-try-disclaimer {{
            font-size: 0.78rem;
            color: #71717a;
            line-height: 1.45;
            margin-top: 0.65rem;
            max-width: 52rem;
          }}
          .p2c-try-disclaimer a {{
            color: #a5b4fc;
            text-decoration: underline;
          }}

          .stButton > button {{
            border-radius: 12px;
            font-weight: 800;
            border: 1px solid var(--border);
          }}
          .stDownloadButton > button {{
            border-radius: 12px;
            font-weight: 800;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_reopen_button() -> None:
    """Mở sidebar khi vừa tải (Streamlit có thể nhớ trạng thái thu gọn trong localStorage)."""
    html = """
<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:transparent;">
<div style="position:fixed;top:52px;left:8px;z-index:999999;">
<button type="button" title="Mở cài đặt (sidebar)"
  onclick="(() => {
    try {
      const d = window.parent.document;
      const q = (s) => d.querySelector(s);
      (q('[data-testid="collapsedControl"]')
        || q('[data-testid="stSidebarCollapsedControl"]'))?.click();
    } catch (e) {}
  })()"
  style="font-size:1.05rem;line-height:1;padding:0.45rem 0.55rem;border-radius:10px;
         border:1px solid rgba(15,23,42,0.12);background:rgba(255,255,255,0.96);
         cursor:pointer;box-shadow:0 4px 14px rgba(15,23,42,0.12);color:__P2C_TEXT__;">
  ☰
</button>
</div>
<script>
(function () {
  const w = window.parent;
  if (w.__p2cSidebarExpandOnLoadScheduled) return;
  w.__p2cSidebarExpandOnLoadScheduled = true;
  const expandIfCollapsed = () => {
    try {
      const d = w.document;
      const side = d.querySelector('section[data-testid="stSidebar"]');
      if (!side) return;
      if (side.getBoundingClientRect().width >= 64) return;
      const q = (s) => d.querySelector(s);
      (q('[data-testid="collapsedControl"]')
        || q('[data-testid="stSidebarCollapsedControl"]'))?.click();
    } catch (e) {}
  };
  [0, 350, 900, 1800].forEach((ms) => setTimeout(expandIfCollapsed, ms));
})();
</script>
</body></html>
""".replace("__P2C_TEXT__", TEXT)
    iframe = getattr(st, "iframe", None)
    if iframe is not None:
        iframe(html, height=52)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=52)
