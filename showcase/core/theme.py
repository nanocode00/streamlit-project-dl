"""두 미니프로젝트용 showcase 디자인 헬퍼.

파일 구조는 유지하면서 Home·분류·회귀 페이지와 같은 테마 값을 사용합니다.
Streamlit이 관리하는 Light/Dark/System 배경색과 글자색은 덮어쓰지 않습니다.
"""

import html

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --sc-coral: #e76f51;
          --sc-mint: #2a9d8f;
          --sc-hero-radius: 18px;
          --sc-surface-radius: 16px;
          --sc-control-radius: 12px;
        }

        .stApp {
          background-image:
            radial-gradient(
              circle at 8% 0%,
              rgba(231, 111, 81, .075) 0,
              transparent 30rem
            ),
            radial-gradient(
              circle at 92% 12%,
              rgba(42, 157, 143, .065) 0,
              transparent 32rem
            );
        }

        [data-testid="stSidebar"] {
          border-right: 1px solid rgba(127, 127, 127, .20);
        }

        .sc-hero {
          position: relative;
          overflow: hidden;
          color: inherit;
          padding: 1.6rem 1.8rem;
          margin-bottom: 1.2rem;
          border: 1px solid rgba(127, 127, 127, .20);
          border-radius: var(--sc-hero-radius);
          background: rgba(127, 127, 127, .04);
          box-shadow: 0 12px 30px rgba(0, 0, 0, .10);
        }

        .sc-hero::after {
          content: "";
          position: absolute;
          width: 240px;
          height: 240px;
          right: -90px;
          top: -100px;
          border-radius: 50%;
          background-image:
            radial-gradient(
              circle,
              rgba(231, 111, 81, .22),
              transparent 70%
            );
          pointer-events: none;
        }

        .sc-kicker {
          color: var(--sc-coral);
          font-weight: 800;
          letter-spacing: .08em;
          font-size: .78rem;
        }

        .sc-title {
          color: inherit;
          font-size: clamp(1.8rem, 4vw, 3rem);
          line-height: 1.08;
          margin: .35rem 0;
        }

        .sc-sub {
          color: inherit;
          font-size: 1.05rem;
          max-width: 780px;
          margin: 0;
          opacity: .74;
        }

        .sc-author {
          color: inherit;
          margin: .8rem 0 0;
          opacity: .64;
        }

        .sc-chip {
          display: inline-block;
          color: inherit;
          margin: .35rem .35rem .35rem 0;
          padding: .35rem .65rem;
          border: 1px solid rgba(127, 127, 127, .20);
          border-radius: 999px;
          background: rgba(127, 127, 127, .04);
          font-size: .82rem;
          opacity: .86;
        }

        .sc-note {
          color: inherit;
          padding: .7rem .9rem;
          border-left: 4px solid var(--sc-mint);
          border-radius: 0 12px 12px 0;
          background: rgba(127, 127, 127, .035);
          opacity: .86;
        }

        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"] {
          border-radius: var(--sc-surface-radius);
        }

        .stButton > button,
        [data-testid="stPageLink"] a {
          border-radius: var(--sc-control-radius);
        }

        @supports (background: color-mix(in srgb, currentColor 4%, transparent)) {
          [data-testid="stSidebar"] {
            border-right-color: color-mix(in srgb, currentColor 20%, transparent);
          }

          .sc-hero {
            background: color-mix(in srgb, currentColor 4%, transparent);
            border-color: color-mix(in srgb, currentColor 20%, transparent);
            box-shadow:
              0 12px 30px
              color-mix(in srgb, currentColor 10%, transparent);
          }

          .sc-chip {
            background: color-mix(in srgb, currentColor 4%, transparent);
            border-color: color-mix(in srgb, currentColor 20%, transparent);
          }

          .sc-note {
            background: color-mix(in srgb, currentColor 3.5%, transparent);
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, name: str) -> None:
    st.markdown(
        f"""
        <section class="sc-hero">
          <div class="sc-kicker">DEEP LEARNING · TWO MODEL STORIES</div>
          <h1 class="sc-title">{html.escape(title)}</h1>
          <p class="sc-sub">{html.escape(subtitle)}</p>
          <div style="margin-top:1rem"><span class="sc-chip">Classification</span><span class="sc-chip">Regression</span><span class="sc-chip">PyTorch</span><span class="sc-chip">Streamlit</span></div>
          <p class="sc-author">Built by {html.escape(name)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )