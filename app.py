import streamlit as st
import pandas as pd
import pickle

cat_features = None
models = None
final_model = None
features_list = None
features_list_final = None


def load_model():
    f = open('models.pkl','rb')
    saved = pickle.load(f)
    all_models = saved['models']
    global features_list
    global features_list_final
    global models
    global final_model
    features_list = saved['features']
    features_list_final = saved['features_final']
    cat_features = saved['cat_features']

    models = all_models[:-1]
    final_model = all_models[-1]

def apply_models(df,models):
    features = df[features_list]
    for index,m in enumerate(models):
        df[f'model_{index}'] = m.predict(features)
    return df

load_model()

st.header('🚘💵 Стоимость поддержанного автомобиля')


def process_data(year,date,make,model,body,state,condition,odometer,color,interior,seller,trim):
    df = pd.DataFrame({
                       'year':[year],
                       'date':[pd.to_datetime(date)],
                       'model':[model],
                       'make':[make],
                       'body':[body],
                       'state':[state],
                       'condition':[condition],
                       'odometer':[odometer],
                       'color':[color],
                       'interior':[interior],
                       'seller':[seller],
                       'trim':[trim]
                       })
    df['make'] = df['make'].str.lower()
    df['model'] = df['model'].str.lower()
    df['body'] = df['body'].str.lower()
    df['seller'] = df['seller'].str.lower()
    df['color'] = df['color'].str.lower()
    df['interior'] = df['interior'].str.lower()
    df['trim'] = df['trim'].str.lower()

    df['make'] = df['make'].fillna('uknown')
    df['model'] = df['model'].fillna('uknown')
    df['body'] = df['body'].fillna('uknown')
    df['odometer'] = df['odometer'].fillna(0)
    df['color'] = df['color'].fillna('uknown')
    df['interior'] = df['interior'].fillna('uknown')
    df['trim'] = df['trim'].fillna('uknown')

    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dow'] = df['date'].dt.day_of_week
    df['sale_year']=df['date'].dt.year

    df['age'] = df['sale_year']-df['year']
    df = df.drop(['sale_year','year','date'],axis=1)
    return df

def predict(df):
    df = apply_models(df,models)
    features = df[features_list_final]
    return final_model.predict(features)
    


with st.form("my_form"):
   st.write("🤖 Модель")
   year = st.number_input('Год выпуска',min_value=1980,max_value=2023)
   saledate = st.date_input('Дата продажи')
   make = st.text_input('Марка')
   model = st.text_input('Издание автомобиля определенной марки')
   body = st.text_input('Тип кузова')
   trim = st.text_input('Уровень отделки салона')
   state = st.text_input('Штат регистрации')
   condition = st.number_input('Состояние на момент аукциона',min_value=0.0,max_value=5.0)
   odometer = st.number_input('Расстояние, пройденное с момента выпуска')
   color = st.text_input('Цвет кузова')
   interior = st.text_input('Цвет салона')
   seller = st.text_input('Продавец автомобиля, автосалоны')


   # Every form must have a submit button.
   submitted = st.form_submit_button("Рассчитать стоимость")
   if submitted:
       df = process_data(year,saledate,make,model,body,state,condition,odometer,color,interior,seller,trim)
       res = predict(df)
       st.write(f"Цена: {res[0]}")