# %%writefile streamlit/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import phik

st.set_page_config(page_title="Car price prediction", page_icon="🚘", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pickle"
DATA_PATH = BASE_DIR / "data.pickle"
DATA_plus_PATH = BASE_DIR / "data_plus.pickle"

cat_feature_names = ['year', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
float_feature_names = ['km_driven','mileage','engine','max_power','torque','max_torque_rpm']
new_cat_feature_names = ['brand', 'model_name', 'drive_mode', 'three_tokens']

@st.cache_resource
def load_data():
    """Загружаем модель через pickle"""

    with open(MODEL_PATH, 'rb') as f:
        OHE, stand_scaler, model = pickle.load(f)
    with open(DATA_PATH, 'rb') as f:
        df_train = pickle.load(f)
    with open(DATA_plus_PATH, 'rb') as f:
        new_cats = pickle.load(f)
    return OHE, stand_scaler, model, df_train, new_cats

def find_token(text, tokens):
    text = text.lower().split()
    for tok in tokens:
        tok_low = tok.lower()
        if tok_low in text:
            return tok
    return 'idk'

def prepare_features(df, new_cats):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()

    df_proc['mileage'] = df_proc['mileage']
    df_proc['engine'] = df_proc['engine']
    df_proc['max_power'] = df_proc['max_power']
    df_proc['max_torque_rpm'] = df_proc['torque']
    df_proc['torque'] = df_proc['torque']
    df_proc['year'] = df_proc['year'].astype(int)
    df_proc['seats'] = df_proc['seats'].astype(int)

    df_proc['brand'] = df_proc['name'].apply(lambda s: find_token(s, new_cats[0]))
    df_proc['model_name'] = df_proc['name'].apply(lambda s: find_token(s, new_cats[1]))
    df_proc['drive_mode'] = df_proc['name'].apply(lambda s: find_token(s, new_cats[2]))
    df_proc['three_tokens'] = df_proc['name'].apply(lambda s: find_token(s, new_cats[3]))

    return df_proc


def scale_features(df_proc, OHE, stand_scaler):
    scaled_part = stand_scaler.transform(df_proc.drop(columns=['name']+cat_feature_names+new_cat_feature_names))
    cat_part = OHE.transform(df_proc.drop(columns=['name']+float_feature_names))
    all_part = np.concat((cat_part.toarray(), scaled_part), axis=1)
    return all_part


# Загружаем модель
try:
    OHE, stand_scaler, MODEL, df_train, new_cats = load_data()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()


# --- Основной интерфейс ---
st.title("Предсказание цены автомобиля")

try:
    proc_df_train = prepare_features(df_train, new_cats)
    features = scale_features(proc_df_train.drop(columns='selling_price'), OHE, stand_scaler)
    predictions = MODEL.predict(features)
    true = df_train['selling_price'].values
    
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


# --- Метрики ---
st.subheader("📊 Анализ")
col1, col2 = st.columns([0.35,0.65])
with col1:
    fig1 = px.imshow(proc_df_train.corr(numeric_only='True'))
    st.plotly_chart(fig1)
with col2:
    st.table(proc_df_train.describe())


# --- Визуализации ---
st.subheader("📈 Визуализации")
st.markdown('**Ошибка предсказания в процентах**')
column2 = st.selectbox(
    "Выберите столбец для анализа:",
    cat_feature_names,   # или любой список: numeric_columns, categorical_columns
)
fig2 = px.histogram((predictions-true)/true*100, color=df_train[column2].values, labels={"value": "Ошибка в процентах", 'color':column2})
st.plotly_chart(fig2)

# --- Форма для предсказания ---
st.subheader("🔮 Сделать предсказание для нового автомобиля")

with st.form("prediction_form"):
    input_data = {}

    input_data['name'] = st.text_input('name', key=f"cat_name", value='Maruti Swift Dzire VDI')
    cols = st.columns(5)
    with cols[0]:
        st.markdown('**Распознаны столбцы:**')
    for idx,name in enumerate(new_cat_feature_names):
        with cols[idx+1]:
            st.markdown(f'**{name}**: {find_token(input_data["name"], new_cats[idx])}')
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("**Категориальные:**")
        for col in  cat_feature_names:
            unique_vals = sorted(df_train[col].astype(str).unique().tolist())
            input_data[col] = st.selectbox(col, unique_vals, key=f"cat_{col}")
    
    with col_right:
        st.write("**Числовые:**")
        for col in float_feature_names:
            val = float(df_train[col].median())
            input_data[col] = st.number_input(col, value=val, key=f"num_{col}")

    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        st.table(input_df)
        prepared_input = prepare_features(input_df, new_cats)
        pred = MODEL.predict(scale_features(prepared_input, OHE, stand_scaler))

        st.success(f"**Результат:** {pred[0]}")
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e}")