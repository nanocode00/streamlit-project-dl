import sys
from html import escape
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from showcase.profile import PROFILE


def apply_page_style() -> None:
    """
    Streamlit이 관리하는 Light/Dark/System 색상은 덮어쓰지 않습니다.

    이 CSS는 레이아웃, 간격, 둥근 모서리, 그림자, 투명 장식과
    제한적인 브랜드 강조색만 담당합니다. 커스텀 영역의 표면색과
    테두리는 현재 테마의 글자색(currentColor)을 기준으로 계산됩니다.
    """
    st.markdown(
        """
        <style>
        :root {
          --sc-coral: #e76f51;
          --sc-mint: #2a9d8f;
        }

        /*
         * Streamlit의 기본 배경색은 그대로 유지합니다.
         * 투명한 background-image만 추가하므로 테마 전환을 방해하지 않습니다.
         */
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

        /* 커스텀 히어로 카드: color-mix 미지원 브라우저용 기본값 */
        .sc-hero {
          color: inherit;
          padding: 1.6rem 1.8rem;
          margin-bottom: 1.2rem;
          border: 1px solid rgba(127, 127, 127, .20);
          border-radius: 18px;
          background: rgba(127, 127, 127, .04);
          box-shadow: 0 12px 30px rgba(0, 0, 0, .10);
        }

        .sc-kicker {
          color: var(--sc-coral);
          font-size: .78rem;
          font-weight: 800;
          letter-spacing: .08em;
        }

        .sc-title {
          color: inherit;
          font-size: clamp(1.8rem, 4vw, 3rem);
          line-height: 1.08;
          margin: .35rem 0;
        }

        .sc-subtitle {
          color: inherit;
          max-width: 780px;
          margin: 0;
          opacity: .74;
        }

        .sc-author {
          color: inherit;
          margin: .8rem 0 0;
          font-size: .9rem;
          opacity: .64;
        }

        /* 커스텀 안내문: 브랜드색은 왼쪽 강조선에만 사용합니다. */
        .sc-note {
          color: inherit;
          padding: .7rem .9rem;
          border-left: 4px solid var(--sc-mint);
          border-radius: 0 12px 12px 0;
          background: rgba(127, 127, 127, .035);
          opacity: .86;
        }

        /* Streamlit 기본 컴포넌트의 색은 건드리지 않고 모양만 유지합니다. */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"] {
          border-radius: 16px;
        }

        .stButton > button,
        [data-testid="stPageLink"] a {
          border-radius: 12px;
        }

        @supports (background: color-mix(in srgb, currentColor 4%, transparent)) {
          .sc-hero {
            background: color-mix(in srgb, currentColor 4%, transparent);
            border-color: color-mix(in srgb, currentColor 20%, transparent);
            box-shadow:
              0 12px 30px
              color-mix(in srgb, currentColor 10%, transparent);
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
    """현재 Streamlit 테마의 글자색을 상속하는 소개 영역을 표시합니다."""
    st.markdown(
        f"""
        <section class="sc-hero">
          <div class="sc-kicker">DEEP LEARNING MODEL LAB</div>
          <h1 class="sc-title">{escape(str(title))}</h1>
          <p class="sc-subtitle">{escape(str(subtitle))}</p>
          <p class="sc-author">제작 · {escape(str(name))}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="나의 딥러닝 모델 랩", page_icon="🧪", layout="wide")
apply_page_style()
hero(PROFILE["title"], PROFILE["subtitle"], PROFILE["name"])

classification_model = PROJECT_ROOT / "과제_Streamlit_앱_분류/mnist_cnn.pt"
regression_model = PROJECT_ROOT / "과제_Streamlit_앱_회귀/bike_reg.pt"
ready_count = sum(path.exists() for path in [classification_model, regression_model])

st.markdown("### 프로젝트 현황")
m1, m2, m3 = st.columns(3)
m1.metric("완성한 모델", f"{ready_count}/2")
m2.metric("문제 유형", "분류 + 회귀")
m3.metric("최종 데모", "1 showcase")

left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True):
        st.markdown("### ✍️ 이미지 분류")
        st.caption("MNIST · CNN · Accuracy")
        st.write("사진을 28×28 tensor로 변환하고, 학습한 CNN이 숫자 0~9의 확률을 출력합니다.")
        st.success("체크포인트 준비 완료" if classification_model.exists() else "노트북 [Step 7] 완료 후 체크포인트 생성")
        st.page_link("pages/1_분류_MNIST.py", label="분류 모델 열기 →", use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown("### 🚲 수요 회귀")
        st.caption("Seoul Bike · MLP · MAE")
        st.write("시간과 날씨를 train 통계로 표준화하고, 미래 시점의 시간당 대여량을 예측합니다.")
        st.success("체크포인트 준비 완료" if regression_model.exists() else "체크포인트 저장 셀 완료 후 생성")
        st.page_link("pages/2_회귀_자전거.py", label="회귀 모델 열기 →", use_container_width=True)

st.markdown("### 내가 설명할 수 있어야 하는 것")
q1, q2, q3 = st.columns(3)
with q1:
    with st.container(border=True):
        st.markdown("**01 · 분류의 한계**")
        st.write(PROFILE["classification_insight"])

with q2:
    with st.container(border=True):
        st.markdown("**02 · 회귀의 오차**")
        st.write(PROFILE["regression_insight"])

with q3:
    with st.container(border=True):
        st.markdown("**03 · 다음 실험**")
        st.write(PROFILE["next_step"])

with st.expander("🧩 두 프로젝트가 하나의 showcase가 되는 구조"):
    st.code(
        """노트북 1 ──학습/저장──> mnist_cnn.pt ──> 분류 앱 ─┐
                                                        ├─> showcase/Home.py
노트북 2 ──학습/저장──> bike_reg.pt  ──> 회귀 앱 ────┘""",
        language="text",
    )
    st.markdown(
        '<div class="sc-note">Part I showcase와 같은 단방향 구조입니다. showcase는 두 앱을 재사용하지만, 각 앱은 showcase 없이도 독립 실행됩니다.</div>',
        unsafe_allow_html=True,
    )