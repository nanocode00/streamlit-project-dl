# 서울 자전거 수요 예측기 — 학생용 핵심 스캐폴딩
# 실행: python3.11 -m streamlit run app_bike.py
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bike_reg.pt"
MY_NAME = "김재훈"
REQUIRED_FEATURES = ["시간", "기온", "습도", "풍속", "가시거리", "이슬점", "일사량", "강우량", "적설량"]

SWEEP_SPECS = {
    "시간": {"min": 0, "max": 23, "step": 1, "label": "시간(시)", "digits": 0},
    "기온": {"min": -18.0, "max": 40.0, "step": 0.5, "label": "기온(°C)", "digits": 1},
    "습도": {"min": 0, "max": 100, "step": 1, "label": "습도(%)", "digits": 0},
    "풍속": {"min": 0.0, "max": 8.0, "step": 0.1, "label": "풍속(m/s)", "digits": 1},
    "가시거리": {"min": 0, "max": 2000, "step": 50, "label": "가시거리(10m)", "digits": 0},
    "이슬점": {"min": -30.0, "max": 28.0, "step": 0.5, "label": "이슬점(°C)", "digits": 1},
    "일사량": {"min": 0.0, "max": 3.6, "step": 0.1, "label": "일사량(MJ/m²)", "digits": 1},
    "강우량": {"min": 0.0, "max": 35.0, "step": 0.5, "label": "강우량(mm)", "digits": 1},
    "적설량": {"min": 0.0, "max": 9.0, "step": 0.1, "label": "적설량(cm)", "digits": 1},
}


class ScaffoldIncomplete(RuntimeError):
    """학생이 아직 완성하지 않은 핵심 구간을 화면에 친절하게 알립니다."""


def build_model(config: dict) -> nn.Module:
    """체크포인트 설정과 같은 회귀 신경망을 만듭니다."""
    # 노트북 run_reg()와 동일한 2개 은닉층 구조입니다.
    # 마지막 층에는 ReLU를 두지 않아 회귀값 전체 범위를 학습할 수 있게 합니다.
    return nn.Sequential(
        nn.Linear(config["input_dim"], config["hidden"]),
        nn.ReLU(),
        nn.Linear(config["hidden"], config["hidden"]),
        nn.ReLU(),
        nn.Linear(config["hidden"], 1),
    )


@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])

    # 학습된 파라미터를 복원하고 Dropout/BatchNorm 등이 추론 방식으로 동작하게 합니다.
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def prepare_features(values: dict[str, float], checkpoint: dict) -> torch.Tensor:
    """화면 입력을 학습 때와 같은 특성 순서·스케일로 변환합니다."""
    feature_names = list(checkpoint["feature_names"])

    # 화면에서 제공하지 않는 특성은 train 평균으로 두어 표준화 후 0이 되게 합니다.
    mean = checkpoint["mean"].detach().clone().to(dtype=torch.float32)
    std = checkpoint["std"].detach().clone().to(dtype=torch.float32)
    raw = mean.clone()

    for name, value in values.items():
        if name in feature_names:
            index = feature_names.index(name)
            raw[index] = float(value)

    # 혹시 표준편차가 0인 특성이 있어도 0으로 나누지 않게 방어합니다.
    safe_std = torch.where(std == 0, torch.ones_like(std), std)
    return (raw - mean) / safe_std


def predict_count(model: nn.Module, normalized: torch.Tensor) -> float:
    """표준화된 한 행으로 시간당 대여량을 예측합니다."""
    # 한 행을 (특성 수,)에서 모델 입력 형태인 (1, 특성 수)로 바꿉니다.
    batch = normalized.unsqueeze(0)

    with torch.inference_mode():
        prediction = model(batch)

    # 출력 shape (1, 1)의 첫 번째 행·첫 번째 값을 Python float으로 변환합니다.
    return float(prediction[0, 0].item())


def make_sweep_values(feature_name: str) -> np.ndarray:
    """선택한 특성의 슬라이더 범위를 sweep 배열로 만듭니다."""
    spec = SWEEP_SPECS[feature_name]
    values = np.arange(
        spec["min"],
        spec["max"] + spec["step"] * 0.5,
        spec["step"],
        dtype=np.float32,
    )

    if spec["digits"] == 0:
        values = values.astype(np.int32)

    return values


def sweep_feature(
    model: nn.Module,
    checkpoint: dict,
    values: dict[str, float],
    feature_name: str,
    sweep_values: np.ndarray,
) -> np.ndarray:
    """
    다른 입력은 현재 값으로 고정하고 선택 특성만 바꿔 한 번에 예측합니다.

    각 행이 sweep 지점 하나가 되도록 입력 배치를 만든 뒤,
    모델을 한 번만 호출해 전체 예측 곡선을 계산합니다.
    """
    feature_names = list(checkpoint["feature_names"])
    mean = torch.as_tensor(checkpoint["mean"], dtype=torch.float32)
    std = torch.as_tensor(checkpoint["std"], dtype=torch.float32)
    safe_std = torch.where(std == 0, torch.ones_like(std), std)

    raw_batch = mean.unsqueeze(0).repeat(len(sweep_values), 1)

    for name, value in values.items():
        if name in feature_names:
            raw_batch[:, feature_names.index(name)] = float(value)

    sweep_index = feature_names.index(feature_name)
    raw_batch[:, sweep_index] = torch.as_tensor(sweep_values, dtype=torch.float32)
    normalized_batch = (raw_batch - mean) / safe_std

    with torch.inference_mode():
        predictions = model(normalized_batch).squeeze(1).clamp_min(0)

    return predictions.cpu().numpy()


def format_sweep_value(feature_name: str, value: float) -> str:
    """sweep 요약에 표시할 변수 값을 단위에 맞춰 포맷합니다."""
    spec = SWEEP_SPECS[feature_name]
    return f"{value:.{spec['digits']}f}"



def apply_page_style() -> None:
    """
    Streamlit이 관리하는 배경색·본문색·사이드바색은 덮어쓰지 않습니다.
    사용자 스타일은 currentColor를 기준으로 만들어 테마 변경 즉시 함께 전환됩니다.
    """
    st.markdown(
        """
        <style>
        /*
         * 핵심 원칙
         * 1. .stApp의 background-color와 color를 강제로 지정하지 않음
         * 2. sidebar의 배경색도 강제로 지정하지 않음
         * 3. 카드·테두리는 currentColor를 낮은 투명도로 섞어 생성
         *
         * 따라서 Light/Dark/System 전환을 Python에서 감지할 필요가 없습니다.
         */

        .stApp {
          /* Streamlit의 원래 배경색은 유지하고 아주 옅은 장식만 추가합니다. */
          background-image:
            linear-gradient(
              135deg,
              rgba(42, 157, 143, .025) 0%,
              rgba(231, 111, 81, .018) 100%
            );
        }

        [data-testid="stHeader"] {
          background: transparent;
        }

        .mp-hero {
          padding: 1.6rem 1.8rem;
          margin-bottom: 1.2rem;
          border-radius: 18px;

          /* color-mix 미지원 브라우저용 기본값 */
          border: 1px solid rgba(127, 127, 127, .28);
          background: rgba(127, 127, 127, .045);
          box-shadow: 0 12px 30px rgba(0, 0, 0, .08);

          /* 현재 테마의 상속된 글자색을 기준으로 즉시 계산됩니다. */
          border-color:
            color-mix(in srgb, currentColor 20%, transparent);
          background:
            color-mix(in srgb, currentColor 4%, transparent);
          box-shadow:
            0 12px 30px
            color-mix(in srgb, currentColor 10%, transparent);
        }

        .mp-kicker {
          color: #2a9d8f;
          font-weight: 800;
          letter-spacing: .08em;
          font-size: .78rem;
        }

        .mp-title {
          /* 고정 색이나 별도 테마 변수를 사용하지 않고 부모 색을 상속합니다. */
          color: inherit;
          font-size: clamp(1.8rem, 4vw, 3rem);
          line-height: 1.08;
          margin: .35rem 0;
        }

        .mp-sub {
          color: inherit;
          opacity: .72;
          margin: 0;
          max-width: 780px;
        }

        .mp-step {
          border-left: 4px solid #e76f51;
          padding: .35rem .8rem;
          color: inherit;
          opacity: .88;
        }

        /*
         * Streamlit이 expander와 border container의 실제 배경·글자색을
         * 관리하게 두고 모양만 조정합니다.
         */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"] {
          border-radius: 16px;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
          color: inherit;
        }

        .stButton > button {
          border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="mp-hero">
          <div class="mp-kicker">MODEL LAB · REGRESSION</div>
          <h1 class="mp-title">날씨와 시간이<br>대여량 하나가 되기까지</h1>
          <p class="mp-sub">학습 때 저장한 특성 순서와 train 통계를 그대로 복원해 미래 시점의 수요를 예측합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="자전거 회귀 모델 랩", page_icon="🚲", layout="wide")
    apply_page_style()
    render_header()

    with st.expander("🧭 이 앱에서 내가 직접 완성할 핵심 4단계", expanded=False):
        st.markdown(
            """
            1. `build_model()` — 체크포인트 설정으로 회귀 구조 재구성
            2. `load_model()` — `state_dict` 복원과 추론 모드 전환
            3. `prepare_features()` — 특성 순서 복원과 train 통계 표준화
            4. `predict_count()` — 한 행을 배치로 바꿔 예측값 반환

            슬라이더와 결과 패널은 제공됩니다. 네 함수가 연결되어야 실제 예측이 시작됩니다.
            """
        )

    if not MODEL_PATH.exists():
        st.error("`bike_reg.pt`가 없습니다. 과제 노트북의 체크포인트 저장 셀을 실행하세요.")
        st.stop()

    try:
        model, checkpoint = load_model()
    except ScaffoldIncomplete as exc:
        st.warning(str(exc))
        st.info("`app_bike.py`의 TODO 1→2 순서로 완성한 뒤 파일을 저장하면 화면이 자동으로 다시 실행됩니다.")
        st.stop()
    except (KeyError, RuntimeError, TypeError) as exc:
        st.error("학습 때의 모델 구조와 앱의 구조가 일치하지 않습니다. TODO 1·2와 checkpoint key를 확인하세요.")
        st.code(str(exc))
        st.stop()

    feature_names = list(checkpoint["feature_names"])
    missing = [name for name in REQUIRED_FEATURES if name not in feature_names]
    if missing:
        st.error(f"필수 특성이 체크포인트에 없습니다: {missing}")
        st.stop()

    metrics = checkpoint["metrics"]
    train_config = checkpoint["training_config"]
    model_config = checkpoint["model_config"]
    st.sidebar.header("MODEL PASSPORT")
    st.sidebar.metric("Validation MAE", f"{metrics['val_mae']:,.1f}대")
    st.sidebar.metric("최종 Test MAE", f"{metrics['test_mae']:,.1f}대")
    st.sidebar.caption(f"hidden {model_config['hidden']} · epochs {train_config['epochs']}")
    st.sidebar.caption(f"lr {train_config['lr']} · batch {train_config['batch']}")
    st.sidebar.caption(f"제작: {MY_NAME}")

    input_col, result_col = st.columns([1.35, 0.65], gap="large")
    with input_col:
        st.subheader("01 · 예측 조건")
        row1 = st.columns(3)
        hour = row1[0].slider("시간", 0, 23, 8)
        temperature = row1[1].slider("기온(°C)", -18.0, 40.0, 20.0, 0.5)
        humidity = row1[2].slider("습도(%)", 0, 100, 55)
        row2 = st.columns(3)
        wind = row2[0].slider("풍속(m/s)", 0.0, 8.0, 1.5, 0.1)
        visibility = row2[1].slider("가시거리(10m)", 0, 2000, 1500, 50)
        dew_point = row2[2].slider("이슬점(°C)", -30.0, 28.0, 10.0, 0.5)
        with st.expander("강수·일사 조건", expanded=True):
            row3 = st.columns(3)
            solar = row3[0].slider("일사량(MJ/m²)", 0.0, 3.6, 0.5, 0.1)
            rain = row3[1].slider("강우량(mm)", 0.0, 35.0, 0.0, 0.1)
            snow = row3[2].slider("적설량(cm)", 0.0, 9.0, 0.0, 0.1)

    values = {
        "시간": hour, "기온": temperature, "습도": humidity, "풍속": wind,
        "가시거리": visibility, "이슬점": dew_point, "일사량": solar,
        "강우량": rain, "적설량": snow,
    }
    with result_col:
        st.subheader("02 · 예측 결과")
        try:
            normalized = prepare_features(values, checkpoint)
            prediction = predict_count(model, normalized)
        except ScaffoldIncomplete as exc:
            st.warning(str(exc))
            st.stop()
        with st.container(border=True):
            st.metric("예상 대여량", f"{max(0, prediction):,.0f}대/시간")
            st.caption(f"Validation MAE 기준, 평균적으로 약 {metrics['val_mae']:,.0f}대 차이가 날 수 있습니다.")
        st.markdown('<p class="mp-step">슬라이더 하나만 바꿔 예측이 어떻게 움직이는지 관찰하고, 데이터에서 배운 관계인지 설명해 보세요.</p>', unsafe_allow_html=True)
        st.caption("2017-12~2018-11 운영일 자료로 학습한 교육용 모델입니다. 실제 운영에는 최신 데이터와 불확실성 검증이 필요합니다.")

    st.divider()
    st.subheader("03 · 변수 민감도")

    sweep_col, guide_col = st.columns([0.42, 0.58], gap="large")
    with sweep_col:
        sweep_name = st.selectbox(
            "변화시킬 변수",
            options=REQUIRED_FEATURES,
            index=REQUIRED_FEATURES.index("기온"),
            help="다른 변수는 현재 슬라이더 값으로 고정하고 선택한 변수만 전체 범위에서 바꿉니다.",
        )
    with guide_col:
        st.markdown(
            f"""
            **현재 조건에서 `{sweep_name}`만 변화시킵니다.**  
            이 그래프는 모델이 학습한 예측 패턴을 보여주며,
            `{sweep_name}`이 수요 변화의 직접적인 원인이라는 뜻은 아닙니다.
            """
        )

    sweep_values = make_sweep_values(sweep_name)
    sweep_predictions = sweep_feature(
        model=model,
        checkpoint=checkpoint,
        values=values,
        feature_name=sweep_name,
        sweep_values=sweep_values,
    )

    spec = SWEEP_SPECS[sweep_name]
    sweep_df = pd.DataFrame({
        spec["label"]: sweep_values,
        "예상 대여량(대/시간)": sweep_predictions,
    }).set_index(spec["label"])

    st.line_chart(sweep_df, height=330)

    min_index = int(np.argmin(sweep_predictions))
    max_index = int(np.argmax(sweep_predictions))
    min_prediction = float(sweep_predictions[min_index])
    max_prediction = float(sweep_predictions[max_index])
    prediction_range = max_prediction - min_prediction

    summary1, summary2, summary3 = st.columns(3)
    summary1.metric(
        "최저 예측",
        f"{min_prediction:,.0f}대/시간",
        help=f"{spec['label']} {format_sweep_value(sweep_name, sweep_values[min_index])} 부근",
    )
    summary2.metric(
        "최고 예측",
        f"{max_prediction:,.0f}대/시간",
        help=f"{spec['label']} {format_sweep_value(sweep_name, sweep_values[max_index])} 부근",
    )
    summary3.metric("전체 변화 폭", f"{prediction_range:,.0f}대/시간")

    st.caption(
        f"최저점: {spec['label']} {format_sweep_value(sweep_name, sweep_values[min_index])} · "
        f"최고점: {spec['label']} {format_sweep_value(sweep_name, sweep_values[max_index])} · "
        f"현재 입력: {format_sweep_value(sweep_name, values[sweep_name])}"
    )

    with st.expander("sweep 예측 데이터 보기", expanded=False):
        display_df = sweep_df.reset_index().copy()
        display_df["예상 대여량(대/시간)"] = display_df["예상 대여량(대/시간)"].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()