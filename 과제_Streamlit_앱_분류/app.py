# THEME FIX V2 + TRUE CENTERED CANVAS + STABLE DRAWING V4 + PNG DOWNLOAD V5 + MOBILE CAMERA V2
# MNIST 손글씨 분류기
# 실행: python3.11 -m streamlit run app.py
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import re

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "mnist_cnn.pt"
MY_NAME = "김재훈"
CANVAS_SIZE = 280


def is_mobile_device() -> bool:
    """요청 헤더를 이용해 모바일·태블릿 브라우저인지 판별합니다."""
    headers = {str(key).lower(): str(value) for key, value in st.context.headers.items()}
    user_agent = headers.get("user-agent", "").lower()
    client_hint_mobile = headers.get("sec-ch-ua-mobile", "").strip()

    if client_hint_mobile == "?1":
        return True

    mobile_pattern = re.compile(
        r"android|iphone|ipad|ipod|mobile|tablet|windows phone|opera mini|opera mobi",
        re.IGNORECASE,
    )
    return bool(mobile_pattern.search(user_agent))


CANVAS_DRAWING_KEY = "mnist_canvas_drawing"
CANVAS_REDO_KEY = "mnist_canvas_redo"
CANVAS_REVISION_KEY = "mnist_canvas_revision"
CANVAS_PENDING_DRAWING_KEY = "mnist_canvas_pending_drawing"


def ensure_canvas_state() -> None:
    """외부 캔버스 제어 버튼에서 사용할 그리기 상태를 초기화합니다."""
    if CANVAS_DRAWING_KEY not in st.session_state:
        st.session_state[CANVAS_DRAWING_KEY] = None
    if CANVAS_REDO_KEY not in st.session_state:
        st.session_state[CANVAS_REDO_KEY] = []
    if CANVAS_REVISION_KEY not in st.session_state:
        st.session_state[CANVAS_REVISION_KEY] = 0
    if CANVAS_PENDING_DRAWING_KEY not in st.session_state:
        st.session_state[CANVAS_PENDING_DRAWING_KEY] = None


def canvas_objects(drawing: dict | None = None) -> list[dict]:
    """저장된 Fabric.js drawing에서 그려진 객체 목록을 반환합니다."""
    if drawing is None:
        drawing = st.session_state.get(CANVAS_DRAWING_KEY)
    if not isinstance(drawing, dict):
        return []
    objects = drawing.get("objects", [])
    return objects if isinstance(objects, list) else []


def canvas_signature(drawing: dict | None) -> str:
    """화면 상태와 세션 상태를 비교하기 위한 안정적인 객체 서명을 만듭니다."""
    return json.dumps(canvas_objects(drawing), sort_keys=True, separators=(",", ":"), default=str)


def sync_canvas_drawing(drawing: dict | None) -> None:
    """사용자가 새로 그린 결과를 세션 상태에 저장하고 redo 기록을 비웁니다."""
    if not isinstance(drawing, dict):
        return

    saved_drawing = st.session_state.get(CANVAS_DRAWING_KEY)
    if canvas_signature(drawing) != canvas_signature(saved_drawing):
        st.session_state[CANVAS_DRAWING_KEY] = deepcopy(drawing)
        st.session_state[CANVAS_REDO_KEY] = []


def remount_canvas(drawing: dict | None) -> None:
    """외부 제어 결과를 다음 캔버스 마운트에 한 번만 주입합니다."""
    st.session_state[CANVAS_PENDING_DRAWING_KEY] = deepcopy(drawing)
    st.session_state[CANVAS_REVISION_KEY] += 1


def undo_canvas() -> None:
    """마지막에 그린 획 하나를 제거합니다."""
    ensure_canvas_state()
    drawing = deepcopy(st.session_state[CANVAS_DRAWING_KEY])
    objects = canvas_objects(drawing)
    if not objects:
        return

    removed_object = objects.pop()
    drawing["objects"] = objects
    st.session_state[CANVAS_DRAWING_KEY] = drawing
    st.session_state[CANVAS_REDO_KEY].append(removed_object)
    remount_canvas(drawing)


def redo_canvas() -> None:
    """가장 최근에 실행 취소한 획 하나를 복원합니다."""
    ensure_canvas_state()
    redo_stack = st.session_state[CANVAS_REDO_KEY]
    if not redo_stack:
        return

    drawing = deepcopy(st.session_state[CANVAS_DRAWING_KEY])
    if not isinstance(drawing, dict):
        drawing = {"version": "4.4.0", "objects": []}

    objects = canvas_objects(drawing)
    objects.append(redo_stack.pop())
    drawing["objects"] = objects
    st.session_state[CANVAS_DRAWING_KEY] = drawing
    remount_canvas(drawing)


def reset_canvas() -> None:
    """캔버스의 모든 획과 실행 취소 기록을 제거합니다."""
    ensure_canvas_state()
    empty_drawing = {"version": "4.4.0", "objects": []}
    st.session_state[CANVAS_DRAWING_KEY] = empty_drawing
    st.session_state[CANVAS_REDO_KEY] = []
    remount_canvas(empty_drawing)


class ScaffoldIncomplete(RuntimeError):
    """학생이 아직 완성하지 않은 핵심 구간을 화면에 친절하게 알립니다."""


class CNN(nn.Module):
    def __init__(self, conv1=32, conv2=64, hidden=128, dropout=0.3):
        super().__init__()

        # 노트북 [Step 3]과 같은 특징 추출부입니다.
        # (N, 1, 28, 28) → (N, conv1, 14, 14) → (N, conv2, 7, 7)
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 체크포인트의 conv2·hidden 설정을 그대로 사용할 수 있게 구성합니다.
        # CrossEntropyLoss로 학습했으므로 마지막에는 Softmax를 넣지 않고 logit을 출력합니다.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2 * 7 * 7, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    """체크포인트로 학습 때와 같은 CNN을 복원합니다."""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    model = CNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def make_mnist_canvas(image: Image.Image) -> np.ndarray:
    """다양한 입력 이미지를 MNIST와 비슷한 28×28 캔버스로 정렬합니다."""
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    gray = ImageOps.grayscale(Image.alpha_composite(white, rgba).convert("RGB"))
    arr = np.array(gray, dtype=np.uint8)

    if float(arr.mean()) > 127:
        arr = 255 - arr
    arr[arr < 30] = 0

    mask = arr > 0
    if mask.any():
        ys, xs = np.where(mask)
        digit = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        digit = arr

    height, width = digit.shape
    side = max(height, width, 1)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = digit

    resized = Image.fromarray(square).resize((20, 20), Image.Resampling.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = np.asarray(resized, dtype=np.uint8)
    return canvas


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Pillow 이미지 → 모델 입력 tensor와 화면용 28×28 배열."""
    canvas = make_mnist_canvas(image)

    x = torch.from_numpy(canvas).to(dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0) / 255.0
    return x, canvas


def canvas_data_to_image(image_data: np.ndarray) -> Image.Image | None:
    """Drawable Canvas의 RGBA 배열을 Pillow 이미지로 변환합니다."""
    rgba = np.asarray(image_data, dtype=np.uint8)
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        return None

    rgb = rgba[:, :, :3]
    drawn_pixels = np.count_nonzero(rgb.max(axis=2) > 20)
    if drawn_pixels < 20:
        return None

    # 알파 채널의 투명 배경 영향을 제거하고 검은 배경·흰 글씨 RGB로 전달합니다.
    return Image.fromarray(rgb, mode="RGB")


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Pillow 이미지를 PNG 다운로드에 사용할 바이트로 변환합니다."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def predict_probabilities(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """모델 logit을 숫자 0~9의 확률 배열로 변환합니다."""
    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.squeeze(0).cpu().numpy()


def apply_page_style() -> None:
    """Streamlit 테마는 유지하고 showcase 전체와 같은 디자인 값을 적용합니다."""
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

        .mp-hero {
          color: inherit;
          padding: 1.6rem 1.8rem;
          margin-bottom: 1.2rem;
          border: 1px solid rgba(127, 127, 127, .20);
          border-radius: var(--sc-hero-radius);
          background: rgba(127, 127, 127, .04);
          box-shadow: 0 12px 30px rgba(0, 0, 0, .10);
        }

        .mp-kicker {
          color: var(--sc-coral);
          font-weight: 800;
          letter-spacing: .08em;
          font-size: .78rem;
        }

        .mp-title {
          color: inherit;
          font-size: clamp(1.8rem, 4vw, 3rem);
          line-height: 1.08;
          margin: .35rem 0;
        }

        .mp-sub {
          color: inherit;
          opacity: .74;
          margin: 0;
          max-width: 780px;
        }

        .mp-step {
          color: inherit;
          opacity: .86;
          border-left: 4px solid var(--sc-mint);
          padding: .35rem .8rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"] {
          border-radius: var(--sc-surface-radius);
        }

        .stButton > button {
          border-radius: var(--sc-control-radius);
        }

        /*
         * 캔버스 관련 규칙은 기능적 레이아웃이므로 그대로 유지합니다.
         */
        .st-key-mnist_canvas_shell {
          margin-inline: auto;
        }

        .st-key-mnist_canvas_shell [data-testid="stCustomComponentV1"],
        .st-key-mnist_canvas_shell iframe {
          width: 100% !important;
          max-width: 100% !important;
        }

        @supports (background: color-mix(in srgb, currentColor 4%, transparent)) {
          [data-testid="stSidebar"] {
            border-right-color: color-mix(in srgb, currentColor 20%, transparent);
          }

          .mp-hero {
            background: color-mix(in srgb, currentColor 4%, transparent);
            border-color: color-mix(in srgb, currentColor 20%, transparent);
            box-shadow:
              0 12px 30px
              color-mix(in srgb, currentColor 10%, transparent);
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="mp-hero">
          <div class="mp-kicker">MODEL LAB · CLASSIFICATION</div>
          <h1 class="mp-title">손글씨 한 장이<br>열 개의 확률이 되기까지</h1>
          <p class="mp-sub">캔버스에 직접 그리거나 이미지를 올리고, 모바일에서는 카메라로 촬영해 CNN의 판단을 확인합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_canvas_input() -> tuple[Image.Image | None, torch.Tensor | None, np.ndarray | None]:
    """가운데 정렬된 숫자 캔버스와 테마 대응 외부 제어 버튼을 표시합니다."""
    if st_canvas is None:
        st.error("캔버스 기능에 필요한 패키지가 없습니다.")
        st.code("pip install streamlit-drawable-canvas-fix==0.9.8")
        return None, None, None

    ensure_canvas_state()
    stroke_width = st.slider("펜 굵기", min_value=12, max_value=36, value=22, step=2)
    st.caption("검은 영역에 흰색으로 숫자 하나를 크게 그리세요. 아래 버튼으로 실행 취소·다시 실행·초기화할 수 있습니다.")

    # 일반적인 획 추가 rerun에서는 initial_drawing을 다시 전달하지 않습니다.
    # 같은 key의 프런트엔드 상태를 유지해 캔버스 재초기화와 깜빡임을 막습니다.
    # 실행 취소·다시 실행·초기화 직후에만 수정된 drawing을 한 번 주입합니다.
    canvas_key = f"mnist_digit_canvas_{st.session_state[CANVAS_REVISION_KEY]}"
    initial_drawing = deepcopy(st.session_state[CANVAS_PENDING_DRAWING_KEY])
    st.session_state[CANVAS_PENDING_DRAWING_KEY] = None

    # custom component는 부모에서 stretch 요소로 취급되므로,
    # 먼저 280px 고정 너비 shell을 만든 뒤 그 shell 자체를 가운데에 배치합니다.
    with st.container(horizontal=True, horizontal_alignment="center"):
        with st.container(width=CANVAS_SIZE, key="mnist_canvas_shell"):
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                update_streamlit=True,
                height=CANVAS_SIZE,
                width=CANVAS_SIZE,
                drawing_mode="freedraw",
                initial_drawing=initial_drawing,
                display_toolbar=False,
                key=canvas_key,
            )

    sync_canvas_drawing(canvas_result.json_data)

    image = None
    if canvas_result.image_data is not None:
        image = canvas_data_to_image(canvas_result.image_data)

    png_data = image_to_png_bytes(image) if image is not None else b""
    undo_disabled = not canvas_objects()
    redo_disabled = not st.session_state[CANVAS_REDO_KEY]
    reset_disabled = undo_disabled and redo_disabled
    download_disabled = image is None

    with st.container(horizontal=True, horizontal_alignment="center", gap="small"):
        st.button(
            "실행 취소",
            icon=":material/undo:",
            disabled=undo_disabled,
            on_click=undo_canvas,
            width="content",
        )
        st.button(
            "다시 실행",
            icon=":material/redo:",
            disabled=redo_disabled,
            on_click=redo_canvas,
            width="content",
        )
        st.button(
            "초기화",
            icon=":material/delete:",
            disabled=reset_disabled,
            on_click=reset_canvas,
            width="content",
        )
        st.download_button(
            "PNG 저장",
            data=png_data,
            file_name="mnist_canvas.png",
            mime="image/png",
            icon=":material/download:",
            disabled=download_disabled,
            on_click="ignore",
            width="content",
        )

    if image is None:
        return None, None, None

    x, preview = preprocess_image(image)
    return image, x, preview


def render_upload_input() -> tuple[Image.Image | None, torch.Tensor | None, np.ndarray | None]:
    """업로드된 이미지를 읽고 모델 입력으로 변환합니다."""
    source = st.file_uploader("PNG 또는 JPG", type=["png", "jpg", "jpeg"])
    if source is None:
        return None, None, None

    try:
        image = Image.open(source)
        x, preview = preprocess_image(image)
        return image, x, preview
    except Exception:
        st.error("이미지를 열 수 없습니다. 손상되지 않은 PNG/JPG 파일인지 확인하세요.")
        return None, None, None


def render_camera_input() -> tuple[Image.Image | None, torch.Tensor | None, np.ndarray | None]:
    """모바일 카메라로 촬영한 이미지를 모델 입력으로 변환합니다."""
    source = st.camera_input("종이의 숫자를 크게 촬영하세요")
    if source is None:
        return None, None, None

    try:
        image = Image.open(source)
        x, preview = preprocess_image(image)
        return image, x, preview
    except Exception:
        st.error("촬영 이미지를 열 수 없습니다. 다시 촬영해 주세요.")
        return None, None, None


def main():
    st.set_page_config(page_title="MNIST 분류 모델 랩", page_icon="✍️", layout="wide")
    apply_page_style()
    render_header()

    with st.expander("🧭 이 앱에서 연결된 핵심 4단계", expanded=False):
        st.markdown(
            """
            1. `CNN` — 학습 때와 같은 모델 구조 재구성
            2. `load_model()` — 구조와 `state_dict` 복원, 추론 모드 전환
            3. `preprocess_image()` — 캔버스·업로드·촬영 이미지를 `(1,1,28,28)` 입력으로 변환
            4. `predict_probabilities()` — logit을 10개 확률로 변환
            """
        )

    if not MODEL_PATH.exists():
        st.error("`mnist_cnn.pt`가 없습니다. 과제 노트북 [Step 7]을 실행해 이 폴더에 생성하세요.")
        st.stop()

    try:
        model, checkpoint = load_model()
    except ScaffoldIncomplete as exc:
        st.warning(str(exc))
        st.info("모델 구조와 체크포인트 저장 형식을 다시 확인하세요.")
        st.stop()
    except (KeyError, RuntimeError, TypeError) as exc:
        st.error("학습 때의 모델 구조와 앱의 구조가 일치하지 않습니다. checkpoint key와 각 층의 크기를 확인하세요.")
        st.code(str(exc))
        st.stop()

    metrics = checkpoint["metrics"]
    train_config = checkpoint["training_config"]
    model_config = checkpoint["model_config"]
    st.sidebar.header("MODEL PASSPORT")
    st.sidebar.metric("Validation 정확도", f"{metrics['val_acc']:.4f}")
    st.sidebar.metric("최종 Test 정확도", f"{metrics['test_acc']:.4f}")
    st.sidebar.metric("파라미터", f"{checkpoint['n_params']:,}")
    st.sidebar.caption(f"epochs {train_config['epochs']} · lr {train_config['lr']}")
    st.sidebar.caption(f"conv {model_config['conv1']}/{model_config['conv2']} · hidden {model_config['hidden']}")
    st.sidebar.caption(f"제작: {MY_NAME}")

    input_col, result_col = st.columns([1, 1], gap="large")
    image = x = preview = None

    with input_col:
        st.subheader("01 · 숫자 입력")
        mobile_device = is_mobile_device()
        input_methods = ["캔버스에 그리기", "이미지 업로드"]
        if mobile_device:
            input_methods.insert(1, "카메라 촬영")

        method = st.radio("입력 방식", input_methods, horizontal=True)

        if method == "캔버스에 그리기":
            image, x, preview = render_canvas_input()
            if image is None:
                st.info("캔버스에 숫자를 그리면 모델 입력과 예측 결과가 자동으로 갱신됩니다.")
                st.markdown(
                    '<p class="mp-step">관찰 포인트: 획의 굵기와 숫자의 위치가 예측 확률에 어떤 영향을 줄까요?</p>',
                    unsafe_allow_html=True,
                )
        elif method == "카메라 촬영":
            image, x, preview = render_camera_input()
            if image is None:
                st.info("밝은 곳에서 종이의 숫자 하나가 크게 보이도록 촬영하세요.")
                st.markdown(
                    '<p class="mp-step">관찰 포인트: 조명과 촬영 각도가 예측 확률에 어떤 영향을 줄까요?</p>',
                    unsafe_allow_html=True,
                )
        else:
            image, x, preview = render_upload_input()
            if image is None:
                st.info("숫자 한 개가 크게 보이는 이미지를 준비하세요. 흰 배경·검은 글씨도 자동으로 반전합니다.")
                st.markdown(
                    '<p class="mp-step">관찰 포인트: 학습 데이터와 업로드 이미지의 분포는 어떻게 다를까요?</p>',
                    unsafe_allow_html=True,
                )

        if image is not None and preview is not None:
            before, after = st.columns(2)
            before.image(image, caption="입력 원본", width="stretch")
            after.image(preview, caption="모델 입력 28×28", clamp=True, width="stretch")

    with result_col:
        st.subheader("02 · 모델의 판단")
        if x is None:
            with st.container(border=True):
                st.markdown("### 결과 대기 중")
                st.caption("왼쪽에서 숫자를 그리거나 이미지를 선택·촬영하면 예측 숫자와 클래스 확률이 표시됩니다.")
            return

        probabilities = predict_probabilities(model, x)
        prediction = int(probabilities.argmax())

        with st.container(border=True):
            st.metric("예측 숫자", prediction)
            st.caption(f"가장 높은 softmax 점수 · {probabilities[prediction] * 100:.1f}%")
            st.bar_chart({"클래스 확률": probabilities})
            top3 = probabilities.argsort()[-3:][::-1]
            st.caption("Top 3 · " + " · ".join(f"{int(i)}: {probabilities[i]:.1%}" for i in top3))

        st.caption("Test 정확도는 정제된 데이터의 성적입니다. 직접 그리거나 업로드·촬영한 이미지는 획·크기·중심이 달라 같은 성능을 보장하지 않습니다.")


if __name__ == "__main__":
    main()