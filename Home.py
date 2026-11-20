from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.header("โปรเจคการจำแนกข้อมูลโรคหัวใจ")

col1, col2 = st.columns(2)

with col1:
   st.header("คนเป็นโรคหัวใจ")
   st.image("./img/h1.png")

with col2:
   st.header("คนไม่เป็นโรคหัวใจ")
   st.image("./img/Pawaret005.jpg")

html_7 = """
<div style="background-color:#EC7063;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>สถิติข้อมูลข้อมูลโรคหัวใจ</h5></center>
</div>
"""

st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Cholesterol'].mean()
dt2 = dt['RestingECG'].mean()
dt3 = dt['HeartDisease'].mean()
dt4 = dt['MaxHR'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""

st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("Age", min_value=1, max_value=100, value=20)
pt_wd = st.slider("Sex", min_value=0, max_value=1)

chest = st.selectbox("ChestPainType", [0, 1, 2, 3])
bp = st.number_input("RestingBP", value=120)

sp_len = st.number_input("Cholesterol")
fast = st.selectbox("FastingBS", [0, 1])
ecg = st.selectbox("RestingECG", [0, 1, 2])

sp_wd = st.number_input("MaxHR")
angina = st.selectbox("ExerciseAngina", [0, 1])
oldpeak = st.number_input("Oldpeak", value=0.0)
slope = st.selectbox("ST_Slope", [0, 1, 2])

if st.button("ทำนายผล"):
   X = dt.drop('HeartDisease', axis=1)
   y = dt['HeartDisease']

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)

   x_input = np.array([[
       pt_len,
       pt_wd,
       chest,
       bp,
       sp_len,
       fast,
       ecg,
       sp_wd,
       angina,
       oldpeak,
       slope
   ]])

   st.write(Knn_model.predict(x_input))
   out = Knn_model.predict(x_input)

   if out[0] == 1:
      st.image("./img/h1.png")
   else:
      st.image("./img/h3.png")
else:
    st.write("ไม่ทำนาย")