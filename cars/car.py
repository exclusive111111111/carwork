import streamlit as st
import os
from fastai.vision.all import *
import pathlib
import sys

# 根据不同的操作系统设置正确的pathlib.Path
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

# 获取当前文件所在的文件夹路径
path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(path,"cars.pkl")

# 加载模型
learn_inf = load_learner(model_path)

# 恢复pathlib.Path的原始值
if sys.platform == "win32":
    pathlib.PosixPath = temp
else:
    pathlib.WindowsPath = temp

st.title("汽车图像分类")
st.write("上传一张图片，应用将预测对应的标签。")

# 允许用户上传图片
uploaded_file = st.file_uploader("选择一张图片",type=["jpg","jpeg","png"])

# 如果用户已上传图片
if uploaded_file is not None:
    # 显示上传的图片
    image = PILImage.create(uploaded_file)
    st.image(image,caption="上传的图片",use_column_width=True)
    
    # 获取预测的标签
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Load the trained model
model_path = 'treemodel.pkl'
clf = joblib.load(model_path)

# Function to predict whether a car is suitable for purchase
def predict_suitability(brand, model,price,oprice,capacity,trunk, year, mileage, repairs, accidents, fuel_consumption, ncap, satisfaction):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        '品牌': [brand],
        '型号': [model],
        '先出售价(万)':[price],
        '原价(万)':[oprice],
        '容纳人数':[capacity],
        '后备箱容量(升)':[trunk],
        '生产年份': [year],
        '里程数(公里)': [mileage],
        '维修次数': [repairs],
        '事故次数': [accidents],
        '油耗(l/100km)': [fuel_consumption],
        'ncap碰撞测试评分': [ncap],
        '用户满意度评分': [satisfaction]
    })
    
    # Convert non-numeric features to numeric using LabelEncoder
    for column in input_data.columns:
        if input_data[column].dtype == type(object):
            le = LabelEncoder()
            input_data[column] = le.fit_transform(input_data[column])
    
    # Predict whether the car is suitable for purchase
    prediction = clf.predict(input_data)
    
    return prediction

# Streamlit app
def main():
    # Title
    st.title("车辆收购适宜性预测")
    
    # Input fields for car features
    brand = st.text_input("品牌")
    model = st.text_input("型号")
    price=st.number_input("售价(万)", value=0)
    oprice=st.number_input("原价(万)", value=0)
    options=['5','6','7']
    capacity=st.selectbox('请选择可容纳人数：',options)
    trunk=st.number_input("后备箱容量(升)", value=0)
    year = st.slider("生产年份", 2000, 2025, 2020)
    mileage = st.number_input("里程数(公里)", value=0)
    repairs = st.number_input("维修次数", value=0)
    accidents = st.number_input("事故次数", value=0)
    fuel_consumption = st.number_input("油耗(l/100km)", value=0.0)
    ncap = st.slider("NCAP碰撞测试评分", 1, 5, 3)
    satisfaction = st.slider("用户满意度评分", 1.0, 5.0, 3.0)
    
    # Prediction button
    if st.button("预测"):
        prediction = predict_suitability(brand, model,price,oprice,capacity,trunk, year, mileage, repairs, accidents, fuel_consumption, ncap, satisfaction)
        if prediction == 1:
            st.success("该车辆适合收购")
        else:
            st.error("该车辆不适合收购")

if __name__ == "__main__":
    main()