import streamlit as st
import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests

# --- 0. HÀM TÌM LINK GỐC TỪ LINK RÚT GỌN ---
def get_real_url(url):
    if not url.startswith('http'):
        url = 'http://' + url
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

# --- 1. HÀM TẠO THÔNG SỐ TỪ URL TRƠN ---
def extract_features(url):
    if not url.startswith('http'):
        url = 'http://' + url
        
    parsed_url = urlparse(url)
    
    url_length = len(url)
    domain_length = len(parsed_url.netloc)
    path_length = len(parsed_url.path)
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_underscore = url.count('_')
    num_slash = url.count('/')
    num_question = url.count('?')
    num_equal = url.count('=')
    num_at = url.count('@')
    num_and = url.count('&')
    num_digits = sum(c.isdigit() for c in url)
    has_ip = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc) else 0
    has_php = 1 if '.php' in parsed_url.path.lower() else 0
    has_exe = 1 if ('.exe' in parsed_url.path.lower() or '.rar' in parsed_url.path.lower() or '.zip' in parsed_url.path.lower()) else 0
    
    return [url_length, domain_length, path_length, num_dots, num_hyphens, num_underscore, 
            num_slash, num_question, num_equal, num_at, num_and, num_digits, has_ip, has_php, has_exe]

# --- 2. HÀM PHÂN TÍCH LÝ DO TỪ CÁC THÔNG SỐ ---
def analyze_reasons(features, is_malicious):
    reasons = []
    
    # Đối chiếu vị trí các thông số trong mảng features trả về từ extract_features
    if features[12] == 1: # has_ip
        reasons.append("🚨 **Dùng trực tiếp địa chỉ IP** thay vì tên miền bình thường (Kỹ thuật hacker thường dùng để ẩn danh).")
    if features[14] == 1: # has_exe
        reasons.append("🚨 **Tải file trực tiếp** (.exe, .zip, .rar), nguy cơ cực kỳ cao chứa virus/trojan.")
    if features[9] > 0: # num_at
        reasons.append("🚩 **Chứa ký tự '@'** (Kỹ thuật đánh lừa trình duyệt bỏ qua phần tên miền hợp pháp phía trước).")
    if features[0] > 75: # url_length
        reasons.append("🚩 **Đường link quá dài** (Thường dùng để nhồi nhét mã độc hoặc che giấu tên miền giả mạo).")
    if features[3] > 3: # num_dots
        reasons.append("🚩 **Quá nhiều dấu chấm** (Tạo ra các tên miền phụ để lừa mắt người dùng, ví dụ: apple.com.login.verify.xxx).")
    if features[4] > 3: # num_hyphens
        reasons.append("🚩 **Nhiều dấu gạch ngang** trong tên miền (Cách phổ biến để bắt chước các thương hiệu lớn).")
        
    # Nếu không có dấu hiệu nổi bật nào
    if not reasons and not is_malicious:
        reasons.append("✅ Cấu trúc link cơ bản, độ dài hợp lý, dùng tên miền chuẩn và không chứa tệp tin đáng ngờ.")
    elif not reasons and is_malicious:
        reasons.append("⚠️ AI nhận diện mẫu cấu trúc này giống với cơ sở dữ liệu mã độc đã học, dù không có lỗi lộ liễu bên ngoài.")
        
    return reasons

# --- 3. HÀM HUẤN LUYỆN AI ---
@st.cache_resource 
def train_model():
    try:
        # Đọc và gộp cả 3 file phần lại
        files = ['malicious_phish_part1.csv', 'malicious_phish_part2.csv', 'malicious_phish_part3.csv']
        df_list = [pd.read_csv(f) for f in files]
        df = pd.concat(df_list, ignore_index=True).sample(n=30000, random_state=42) # Lấy ngẫu nhiên 30k dòng từ tổng thể
    except FileNotFoundError:
        return None, None, None
        
    features = df['url'].apply(lambda x: extract_features(str(x)))
    X = pd.DataFrame(features.tolist(), columns=[
        'url_length', 'domain_length', 'path_length', 'num_dots', 'num_hyphens', 'num_underscore', 
        'num_slash', 'num_question', 'num_equal', 'num_at', 'num_and', 'num_digits', 'has_ip', 'has_php', 'has_exe'
    ])
    
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, le, X.columns.tolist()

# --- 4. GIAO DIỆN WEB ---
st.set_page_config(page_title="Phát Hiện URL Mã Độc", page_icon="🛡️", layout="centered")

st.title("🛡️ Web App Kiểm Tra URL Mã Độc")
st.markdown("Hệ thống tự động giải mã link rút gọn, quét mã độc và **giải thích chi tiết lý do** phát hiện.")

model, label_encoder, feature_columns = train_model()

if model is None:
    st.error("❌ Không tìm thấy file `malicious_phish.csv`. Vui lòng để nó cùng thư mục với file `app.py`.")
else:
    user_url = st.text_input("Nhập URL cần quét:")
    
    if st.button("🔍 Quét URL", type="primary"):
        if user_url.strip() == "":
            st.warning("Vui lòng nhập một đường link hợp lệ!")
        else:
            with st.spinner("Hệ thống đang phân tích chi tiết..."):
                
                real_url = get_real_url(user_url)
                if real_url != user_url and real_url != ('http://' + user_url):
                    st.info(f"🔗 **Phát hiện Link Rút Gọn!** \nHệ thống đã truy xuất được link đích thực sự là: \n`{real_url}`")
                
                features_list = extract_features(real_url)
                features_df = pd.DataFrame([features_list], columns=feature_columns)
                
                pred_encoded = model.predict(features_df)[0]
                prediction_text = label_encoder.inverse_transform([pred_encoded])[0] 
                is_malicious = (prediction_text != 'benign')
                
                probabilities = model.predict_proba(features_df)[0]
                max_prob = np.max(probabilities) * 100
                
                st.divider()
                
                # In ra kết quả tổng quan
                if not is_malicious:
                    st.success(f"✅ **AN TOÀN (Benign)** - Độ tin cậy: **{max_prob:.2f}%**")
                elif prediction_text == 'phishing':
                    st.error(f"🎣 **CẢNH BÁO LỪA ĐẢO (Phishing)** - Độ tin cậy: **{max_prob:.2f}%**")
                elif prediction_text == 'malware':
                    st.error(f"🦠 **CẢNH BÁO MÃ ĐỘC (Malware)** - Độ tin cậy: **{max_prob:.2f}%**")
                elif prediction_text == 'defacement':
                    st.warning(f"⚠️ **CẢNH BÁO DEFACEMENT** - Độ tin cậy: **{max_prob:.2f}%**")
                
                # In ra phần Lý Do
                st.markdown("### 🔎 Phân tích của hệ thống:")
                reasons = analyze_reasons(features_list, is_malicious)
                for r in reasons:
                    st.markdown(f"- {r}")