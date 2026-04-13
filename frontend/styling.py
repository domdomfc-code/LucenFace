"""CSS và tiện ích HTML nhỏ cho Streamlit."""
from __future__ import annotations

import streamlit as st

from frontend.config import BG, BLUE


def inject_app_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --blue: {BLUE};
            --bg: {BG};
            --card: #ffffff;
            --text: #0f172a;
            --muted: #64748b;
            --border: rgba(2, 6, 23, 0.10);
            --shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
          }}

          .stApp {{
            background:
              radial-gradient(1000px 500px at 20% 0%, rgba(0, 91, 196, 0.14), rgba(0,0,0,0) 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(56, 189, 248, 0.14), rgba(0,0,0,0) 60%),
              var(--bg);
          }}

          section.main > div {{
            padding-top: 1.25rem;
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
          .app-subtitle {{
            color: var(--muted);
            margin-bottom: 1.0rem;
            font-weight: 600;
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
            color: #111827;
            min-width: 160px;
          }}
          .check-msg {{
            color: #374151;
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
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-top: 1px dashed rgba(255, 255, 255, 0.18) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 0 !important;
            padding: 1.1rem 1.25rem 1.15rem !important;
            box-shadow: none !important;
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

          .p2c-upload-hero-top {{
            text-align: center;
            background: #1e1e24;
            color: rgba(255, 255, 255, 0.92);
            border-radius: 16px 16px 0 0;
            padding: 2rem 1.5rem 1.15rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-bottom: none;
            margin-bottom: 0;
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
            color: #0f172a;
            font-weight: 700;
            font-size: 0.92rem;
            border: 1px solid rgba(0, 91, 196, 0.2);
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
         cursor:pointer;box-shadow:0 4px 14px rgba(15,23,42,0.12);color:#0f172a;">
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
"""
    iframe = getattr(st, "iframe", None)
    if iframe is not None:
        iframe(html, height=52)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=52)
