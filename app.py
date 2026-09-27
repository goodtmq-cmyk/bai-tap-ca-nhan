import streamlit as st
import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import requests


# ============================================================
# 0. TÌM URL ĐÍCH SAU REDIRECT
# ============================================================
def get_real_url(url):
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=5
        )
        return response.url
    except requests.RequestException:
        return url


# ============================================================
# 1. TRÍCH XUẤT FEATURE
# ============================================================
def extract_features(url):

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower()
    path = parsed_url.path.lower()
    full_url = url.lower()

    # ----------------------------
    # Feature cơ bản
    # ----------------------------
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)

    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_underscore = url.count("_")
    num_slash = url.count("/")
    num_question = url.count("?")
    num_equal = url.count("=")
    num_at = url.count("@")
    num_and = url.count("&")

    num_digits = sum(c.isdigit() for c in url)

    # ----------------------------
    # IP thay vì domain
    # ----------------------------
    has_ip = 1 if re.fullmatch(
        r"\d{1,3}(\.\d{1,3}){3}",
        domain.split(":")[0]
    ) else 0

    # ----------------------------
    # File đáng chú ý
    # ----------------------------
    has_php = int(".php" in path)

    dangerous_extensions = (
        ".exe", ".scr", ".bat", ".cmd",
        ".msi", ".jar", ".ps1"
    )

    has_exe = int(
        any(ext in path for ext in dangerous_extensions)
    )

    # ----------------------------
    # HTTPS
    # ----------------------------
    has_https = int(parsed_url.scheme == "https")

    # ----------------------------
    # Số lượng subdomain gần đúng
    # ----------------------------
    domain_without_port = domain.split(":")[0]

    domain_parts = [
        p for p in domain_without_port.split(".")
        if p
    ]

    num_subdomains = max(0, len(domain_parts) - 2)

    # ----------------------------
    # Từ khóa thường xuất hiện
    # trong URL phishing
    # ----------------------------
    suspicious_words = [
        "login",
        "verify",
        "verification",
        "secure",
        "security",
        "account",
        "update",
        "signin",
        "password",
        "credential",
        "confirm",
        "banking",
        "wallet",
        "recover"
    ]

    suspicious_word_count = sum(
        1 for word in suspicious_words
        if word in full_url
    )

    # ----------------------------
    # Dấu hiệu URL encoding
    # ----------------------------
    num_percent = url.count("%")

    # ----------------------------
    # Domain chứa nhiều số
    # ----------------------------
    domain_digits = sum(
        c.isdigit() for c in domain
    )

    # ----------------------------
    # www xuất hiện bất thường
    # ----------------------------
    num_www = full_url.count("www")

    return [
        url_length,
        domain_length,
        path_length,
        num_dots,
        num_hyphens,
        num_underscore,
        num_slash,
        num_question,
        num_equal,
        num_at,
        num_and,
        num_digits,
        has_ip,
        has_php,
        has_exe,
        has_https,
        num_subdomains,
        suspicious_word_count,
        num_percent,
        domain_digits,
        num_www
    ]


# ============================================================
# 2. TÊN CÁC FEATURE
# ============================================================
FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscore",
    "num_slash",
    "num_question",
    "num_equal",
    "num_at",
    "num_and",
    "num_digits",
    "has_ip",
    "has_php",
    "has_exe",
    "has_https",
    "num_subdomains",
    "suspicious_word_count",
    "num_percent",
    "domain_digits",
    "num_www"
]


# ============================================================
# 3. PHÂN TÍCH LÝ DO
# ============================================================
def analyze_reasons(features):

    reasons = []

    feature = dict(zip(FEATURE_COLUMNS, features))

    if feature["has_ip"]:
        reasons.append(
            "🚨 URL sử dụng trực tiếp địa chỉ IP thay vì tên miền."
        )

    if feature["has_exe"]:
        reasons.append(
            "🚨 URL trỏ tới file thực thi có khả năng gây rủi ro."
        )

    if feature["num_at"] > 0:
        reasons.append(
            "🚩 URL chứa ký tự `@`, cần kiểm tra kỹ tên miền thực."
        )

    if feature["url_length"] > 100:
        reasons.append(
            "🚩 URL có độ dài bất thường."
        )

    if feature["num_subdomains"] >= 4:
        reasons.append(
            f"🚩 URL có nhiều tên miền phụ "
            f"({feature['num_subdomains']} subdomain)."
        )

    if feature["suspicious_word_count"] >= 3:
        reasons.append(
            "🚩 URL chứa nhiều từ liên quan đến "
            "đăng nhập/xác minh/tài khoản."
        )

    if feature["num_hyphens"] >= 4:
        reasons.append(
            "🚩 Tên URL chứa nhiều dấu gạch ngang."
        )

    if feature["num_percent"] >= 3:
        reasons.append(
            "🚩 URL chứa nhiều ký tự được mã hóa bằng `%`."
        )

    if not reasons:
        reasons.append(
            "ℹ️ Không phát hiện dấu hiệu cấu trúc URL nổi bật."
        )

    return reasons


# ============================================================
# 4. TRAIN MODEL
# ============================================================
@st.cache_resource
def train_model():

    try:
        files = [
            "malicious_phish_part1.csv",
            "malicious_phish_part2.csv",
            "malicious_phish_part3.csv"
        ]

        df_list = [
            pd.read_csv(f)
            for f in files
        ]

        df = pd.concat(
            df_list,
            ignore_index=True
        )

    except FileNotFoundError:
        return None, None, None, None

    # Chỉ giữ dữ liệu cần thiết
    df = df[["url", "type"]].dropna()

    # Xóa URL trùng
    df = df.drop_duplicates(
        subset=["url"]
    )

    # Nếu dataset quá lớn thì lấy mẫu
    if len(df) > 100000:
        df = df.sample(
            n=100000,
            random_state=42
        )

    # ----------------------------
    # Extract feature
    # ----------------------------
    feature_data = df["url"].apply(
        lambda x: extract_features(str(x))
    )

    X = pd.DataFrame(
        feature_data.tolist(),
        columns=FEATURE_COLUMNS
    )

    # ----------------------------
    # Encode label
    # ----------------------------
    le = LabelEncoder()

    y = le.fit_transform(
        df["type"].astype(str)
    )

    # ----------------------------
    # Tách TRAIN / TEST
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ----------------------------
    # Random Forest
    # ----------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # ----------------------------
    # Đánh giá trên TEST
    # ----------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "report": report
    }

    return (
        model,
        le,
        FEATURE_COLUMNS,
        metrics
    )


# ============================================================
# 5. GIAO DIỆN
# ============================================================
st.set_page_config(
    page_title="Phát Hiện URL Mã Độc",
    page_icon="🛡️",
    layout="centered"
)

st.title(
    "🛡️ Web App Kiểm Tra URL Mã Độc"
)

st.markdown(
    """
    Hệ thống phân tích cấu trúc URL bằng Machine Learning
    và đưa ra các dấu hiệu cần chú ý.
    """
)


# ============================================================
# 6. LOAD MODEL
# ============================================================
model, label_encoder, feature_columns, metrics = train_model()


if model is None:

    st.error(
        "❌ Không tìm thấy các file dataset."
    )

else:

    # Có thể bỏ phần này nếu không muốn hiện accuracy
    with st.expander("📊 Thông tin mô hình"):

        st.write(
            f"Accuracy trên tập test: "
            f"**{metrics['accuracy'] * 100:.2f}%**"
        )

        report_df = pd.DataFrame(
            metrics["report"]
        ).transpose()

        st.dataframe(report_df)


    # ========================================================
    # INPUT URL
    # ========================================================
    user_url = st.text_input(
        "Nhập URL cần quét:"
    )


    if st.button(
        "🔍 Quét URL",
        type="primary"
    ):

        if not user_url.strip():

            st.warning(
                "Vui lòng nhập URL."
            )

        else:

            with st.spinner(
                "Hệ thống đang phân tích..."
            ):

                # --------------------------------------------
                # Resolve redirect
                # --------------------------------------------
                real_url = get_real_url(
                    user_url.strip()
                )

                if real_url != user_url:

                    st.info(
                        f"""
                        🔗 URL sau redirect:

                        `{real_url}`
                        """
                    )


                # --------------------------------------------
                # Extract feature
                # --------------------------------------------
                features_list = extract_features(
                    real_url
                )

                features_df = pd.DataFrame(
                    [features_list],
                    columns=feature_columns
                )


                # --------------------------------------------
                # Predict probability
                # --------------------------------------------
                probabilities = model.predict_proba(
                    features_df
                )[0]

                best_index = np.argmax(
                    probabilities
                )

                prediction_text = (
                    label_encoder.classes_[best_index]
                )

                max_prob = (
                    probabilities[best_index] * 100
                )


                # --------------------------------------------
                # HIỂN THỊ XÁC SUẤT TỪNG CLASS
                # --------------------------------------------
                probability_dict = {}

                for cls, prob in zip(
                    label_encoder.classes_,
                    probabilities
                ):
                    probability_dict[cls] = prob * 100


                # --------------------------------------------
                # Rule analysis
                # --------------------------------------------
                reasons = analyze_reasons(
                    features_list
                )

                feature_dict = dict(
                    zip(
                        FEATURE_COLUMNS,
                        features_list
                    )
                )

                strong_warning = (
                    feature_dict["has_ip"] == 1
                    or
                    feature_dict["has_exe"] == 1
                    or
                    feature_dict["num_at"] > 0
                    or
                    feature_dict["suspicious_word_count"] >= 3
                    or
                    feature_dict["num_subdomains"] >= 4
                )


                st.divider()


                # ====================================================
                # 7. QUYẾT ĐỊNH KẾT QUẢ
                # ====================================================

                if max_prob < 60:

                    st.warning(
                        f"""
                        ⚠️ **KẾT QUẢ KHÔNG CHẮC CHẮN**

                        Model nghiêng về **{prediction_text.upper()}**
                        với xác suất **{max_prob:.2f}%**.
                        """
                    )

                elif prediction_text == "benign":

                    if strong_warning:

                        st.warning(
                            f"""
                            ⚠️ **CÓ DẤU HIỆU ĐÁNG NGỜ**

                            Model dự đoán Benign
                            ({max_prob:.2f}%),
                            nhưng URL chứa một số đặc điểm
                            cần kiểm tra thêm.
                            """
                        )

                    else:

                        st.success(
                            f"""
                            ✅ **KHÔNG PHÁT HIỆN NGUY CƠ RÕ RÀNG**

                            Model dự đoán Benign:
                            **{max_prob:.2f}%**
                            """
                        )

                elif prediction_text == "phishing":

                    st.error(
                        f"""
                        🎣 **NGHI NGỜ PHISHING**

                        Xác suất mô hình:
                        **{max_prob:.2f}%**
                        """
                    )

                elif prediction_text == "malware":

                    st.error(
                        f"""
                        🦠 **NGHI NGỜ MALWARE**

                        Xác suất mô hình:
                        **{max_prob:.2f}%**
                        """
                    )

                elif prediction_text == "defacement":

                    st.warning(
                        f"""
                        ⚠️ **NGHI NGỜ DEFACEMENT**

                        Xác suất mô hình:
                        **{max_prob:.2f}%**
                        """
                    )

                else:

                    st.warning(
                        f"""
                        ⚠️ Model dự đoán:
                        **{prediction_text}**

                        Xác suất:
                        **{max_prob:.2f}%**
                        """
                    )


                # ====================================================
                # 8. XÁC SUẤT CHI TIẾT
                # ====================================================

                st.markdown(
                    "### 📊 Xác suất dự đoán"
                )

                prob_df = pd.DataFrame({
                    "Loại": list(
                        probability_dict.keys()
                    ),
                    "Xác suất (%)": list(
                        probability_dict.values()
                    )
                })

                prob_df = prob_df.sort_values(
                    "Xác suất (%)",
                    ascending=False
                )

                st.dataframe(
                    prob_df,
                    hide_index=True,
                    use_container_width=True
                )


                # ====================================================
                # 9. GIẢI THÍCH
                # ====================================================

                st.markdown(
                    "### 🔎 Phân tích URL"
                )

                for reason in reasons:

                    st.markdown(
                        f"- {reason}"
                    )