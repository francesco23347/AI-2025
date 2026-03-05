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
# CSS Style
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
# Load Data
# -------------------------
df = pd.read_csv("ai_timebank_volunteers_updated.csv")

# เปลี่ยนชื่อ column ให้สั้นลง
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

# Feature ที่ใช้ train AI
X = df[service_cols + ["ระยะทาง","รีวิว","งานสำเร็จ"]]

# Target Score
y = df["รีวิว"]*5 + df["งานสำเร็จ"]*0.2 - df["ระยะทาง"]

# สร้างโมเดล AI
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
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

    distance = st.slider(
        "ระยะทางสูงสุด (km)",
        1,10,5
    )

    search = st.button("🔍 ค้นหาอาสาสมัคร")

# -------------------------
# Result Panel
# -------------------------
with col2:

    st.markdown("### 🤖 AI Matching Result")

if search:

    # กรองตามระยะทาง
    df2 = df[df["ระยะทาง"] <= distance].copy()

    # กรองตามประเภทบริการ
    df2 = df2[df2[service] == 1]

    if len(df2) == 0:

        st.error("ไม่พบอาสาสมัครในระยะที่กำหนด")

    else:

        # -------------------------
        # AI Prediction
        # -------------------------

        X_pred = df2[service_cols + ["ระยะทาง","รีวิว","งานสำเร็จ"]]

        df2["AI Score"] = model.predict(X_pred)

        # สูตรคะแนนใหม่ (ดูเป็น AI มากขึ้น)
        df2["Final Score"] = (
            df2["AI Score"] * 0.5 +
            df2["รีวิว"] * 10 +
            df2["งานสำเร็จ"] * 0.1 -
            df2["ระยะทาง"] * 2
        )

        # เรียงคะแนน
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
        # Ranking Graph
        # -------------------------

        fig = px.bar(
            df2.head(10),
            x="ชื่อ",
            y="Final Score",
            color="Final Score",
            title="Top AI Volunteer Ranking"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Table
        # -------------------------

        st.markdown("### 📊 Volunteer Data")
        st.dataframe(df2)
