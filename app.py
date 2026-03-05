import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Time Bank",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:bold;
color:#00f2ff;
}

.card{
padding:20px;
border-radius:15px;
background-color:#111827;
box-shadow:0px 0px 15px rgba(0,255,255,0.3);
}

.big-font{
font-size:22px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.markdown('<p class="main-title">🤖 AI Smart Time Bank System</p>', unsafe_allow_html=True)
st.write("ระบบจับคู่การช่วยเหลือผู้สูงอายุด้วย **Artificial Intelligence Matching Algorithm**")

# -------------------------
# Load Excel Data
# -------------------------
df = pd.read_csv("ai_timebank_volunteers_updated.csv")

# เปลี่ยนชื่อ column
df = df.rename(columns={
    "ระยะทาง_km":"ระยะทาง",
    "คะแนนรีวิว":"รีวิว",
    "ความสำเร็จงานที่ผ่านมา":"งานสำเร็จ"
})

# -------------------------
# Service Columns
# -------------------------
service_cols = [
    "พาไปโรงพยาบาล",
    "ซ่อมของ",
    "ทำความสะอาดบ้าน",
    "ซื้อของให้",
    "พาไปทำธุระ",
    "ดูแลผู้สูงอายุ"
]

# -------------------------
# AI MODEL
# -------------------------
X = df[service_cols + ["ระยะทาง","รีวิว","งานสำเร็จ"]]

y = df["รีวิว"]*5 + df["งานสำเร็จ"]*0.2 - df["ระยะทาง"]

model = RandomForestRegressor()
model.fit(X,y)

# -------------------------
# Layout
# -------------------------
col1,col2 = st.columns([1,2])

# -------------------------
# Request Panel
# -------------------------
with col1:

    st.markdown("### 📋 Request Service")

    service = st.selectbox(
        "ประเภทการช่วยเหลือ",
        service_cols
    )

    distance = st.slider("ระยะทางสูงสุด",1,10,5)

    search = st.button("🔍 ค้นหาอาสาสมัคร")

# -------------------------
# Result Panel
# -------------------------
with col2:

    st.markdown("### 🤖 AI Matching Result")

if search:

    df2 = df[df["ระยะทาง"] <= distance].copy()

    # กรองประเภทบริการ
    df2 = df2[df2[service] == 1]

    if len(df2) == 0:
        st.error("ไม่พบอาสาสมัครในระยะที่กำหนด")

    else:

        # AI Prediction
        X_pred = df2[service_cols + ["ระยะทาง","รีวิว","งานสำเร็จ"]]

        df2["AI Score"] = model.predict(X_pred)

        # Final Score (เน้นรีวิวมากกว่า AI)
        df2["Final Score"] = (df2["รีวิว"] * 5) + df2["AI Score"]

        df2 = df2.sort_values("Final Score", ascending=False)

        best = df2.iloc[0]

        # -------------------------
        # Best Volunteer Card
        # -------------------------
        st.markdown(f"""
        <div class="card">
        <p class="big-font">🏆 อาสาสมัครที่เหมาะสมที่สุด</p>
        👤 {best['ชื่อ']} <br>
        ⭐ รีวิว {best['รีวิว']} <br>
        📍 ระยะทาง {best['ระยะทาง']} km <br>
        🤖 AI Score {round(best['AI Score'],2)} <br>
        🏆 Final Score {round(best['Final Score'],2)}
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # จำนวนอาสาสมัคร
        # -------------------------
        st.metric(
            label="👥 จำนวนอาสาสมัครที่พบ",
            value=len(df2)
)
        # -------------------------
        # Table
        # -------------------------
        st.markdown("### 📊 Volunteer Data")
        st.dataframe(df2)
