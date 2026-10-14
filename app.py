"""Streamlit web application for online purchase prediction."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "best_model.joblib"
METADATA_FILE = ARTIFACTS_DIR / "best_model_metadata.json"
COMPARISON_FILE = ARTIFACTS_DIR / "model_comparison.csv"


st.set_page_config(
    page_title="Online Purchase Prediction",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล: {MODEL_FILE}. "
            "กรุณารัน C05_online_purchase_classification.py ก่อน"
        )
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metadata():
    if not METADATA_FILE.exists():
        return {}
    with METADATA_FILE.open(encoding="utf-8-sig") as file:
        return json.load(file)


@st.cache_data
def load_comparison():
    if not COMPARISON_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_FILE)


def create_prediction_input(
    pages_viewed,
    session_minutes,
    items_viewed,
    discount_seen,
    returning_visitor,
    device,
):
    return pd.DataFrame(
        [
            {
                "pages_viewed": pages_viewed,
                "session_minutes": session_minutes,
                "items_viewed": items_viewed,
                "discount_seen": discount_seen,
                "returning_visitor": returning_visitor,
                "device": device,
            }
        ]
    )


def show_prediction(model, input_data):
    prediction = int(model.predict(input_data)[0])
    if prediction == 1:
        st.success("ผลการทำนาย: มีแนวโน้มซื้อสินค้า (purchased = 1)")
    else:
        st.warning("ผลการทำนาย: ไม่มีแนวโน้มซื้อสินค้า (purchased = 0)")

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(input_data)[0])
        st.caption(
            "Decision score ของโมเดล: "
            f"{score:.4f} (ไม่ใช่ probability โดยตรง)"
        )


try:
    model = load_model()
    metadata = load_metadata()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()


st.title("🛒 Online Purchase Prediction")
st.write(
    "กรอกข้อมูลพฤติกรรมของผู้เข้าชมเว็บไซต์ "
    "เพื่อทำนายว่าผู้เข้าชมมีแนวโน้มซื้อสินค้าหรือไม่"
)

best_algorithm = metadata.get("best_algorithm", "ไม่ทราบ")
test_metrics = metadata.get("test_metrics", {})

metric_columns = st.columns(5)
metric_columns[0].metric("Best Algorithm", best_algorithm)
metric_columns[1].metric(
    "CV F1",
    f"{metadata.get('best_cv_f1', 0):.3f}",
)
metric_columns[2].metric(
    "Test Accuracy",
    f"{test_metrics.get('test_accuracy', 0):.3f}",
)
metric_columns[3].metric(
    "Test Precision",
    f"{test_metrics.get('test_precision', 0):.3f}",
)
metric_columns[4].metric(
    "Test Recall",
    f"{test_metrics.get('test_recall', 0):.3f}",
)

st.divider()

with st.form("purchase_prediction_form"):
    st.subheader("กรอกข้อมูลผู้เข้าชม")
    left_column, right_column = st.columns(2)

    with left_column:
        pages_viewed = st.number_input(
            "จำนวนหน้าที่ดู (pages_viewed)",
            min_value=0.0,
            max_value=100.0,
            value=9.0,
            step=1.0,
        )
        session_minutes = st.number_input(
            "เวลา Session เป็นนาที (session_minutes)",
            min_value=0.0,
            max_value=240.0,
            value=8.0,
            step=0.1,
        )
        items_viewed = st.number_input(
            "จำนวนสินค้าที่ดู (items_viewed)",
            min_value=0.0,
            max_value=200.0,
            value=13.0,
            step=1.0,
        )

    with right_column:
        discount_seen = st.selectbox(
            "เห็นส่วนลดหรือไม่ (discount_seen)",
            options=["Yes", "No"],
        )
        returning_visitor = st.selectbox(
            "เป็นผู้เข้าชมเดิมหรือไม่ (returning_visitor)",
            options=["Yes", "No"],
        )
        device = st.selectbox(
            "อุปกรณ์ที่ใช้ (device)",
            options=["Desktop", "Mobile", "Tablet"],
        )

    submitted = st.form_submit_button(
        "ทำนายผลการซื้อสินค้า",
        type="primary",
        use_container_width=True,
    )

if submitted:
    prediction_input = create_prediction_input(
        pages_viewed,
        session_minutes,
        items_viewed,
        discount_seen,
        returning_visitor,
        device,
    )
    st.subheader("ผลการทำนาย")
    show_prediction(model, prediction_input)
    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(prediction_input, use_container_width=True)

st.divider()

with st.expander("ผลการเปรียบเทียบโมเดล"):
    comparison = load_comparison()
    if comparison.empty:
        st.info("ไม่พบไฟล์ model_comparison.csv")
    else:
        display_columns = [
            "Algorithm",
            "Best CV F1",
            "Test Accuracy",
            "Test Precision",
            "Test Recall",
            "Test F1",
        ]
        st.dataframe(
            comparison[display_columns].style.format(
                {
                    "Best CV F1": "{:.4f}",
                    "Test Accuracy": "{:.4f}",
                    "Test Precision": "{:.4f}",
                    "Test Recall": "{:.4f}",
                    "Test F1": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

st.caption(
    "โมเดลถูกฝึกจาก C05_online_purchase_classification.xlsx "
    "และเลือกจาก Mean CV F1 บน Training Set"
)
