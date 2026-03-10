# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 10:13:47 2026

@author: Lab
"""
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load models
loan_model = pickle.load(open('loan_model.sav', 'rb'))
ridingmower_model = pickle.load(open('RidingMowers_model.sav', 'rb'))
bmi_model = pickle.load(open('bmi_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu(
        'Classification',
        ['Loan', 'RidingMowers', 'BMI'])


# ─── BMI ───────────────────────────────────────────────────────────────────────
if selected == 'BMI':
    st.title('BMI Classification Predict')

    Gender = st.selectbox('Gender', options=['Male', 'Female'])
    Height = st.text_input('Height (cm)')
    Weight = st.text_input('Weight (kg)')

    bmi_label_map = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity',
        5: 'Extreme Obesity'
    }

    bmi_result = ''

    if st.button('Predict BMI'):
        try:
            gender_val = 1 if Gender == 'Male' else 0
            bmi_prediction = bmi_model.predict([[
                float(gender_val),
                float(Height),
                float(Weight)
            ]])
            bmi_result = bmi_label_map.get(bmi_prediction[0], str(bmi_prediction[0]))
        except Exception as e:
            bmi_result = f'Error: {e}'

    st.success(bmi_result)


# ─── RidingMowers ──────────────────────────────────────────────────────────────
if selected == 'RidingMowers':
    st.title('RidingMowers Predict')

    Income = st.text_input('Income')
    LotSize = st.text_input('LotSize')
    ridingmowers_predict = ''

    if st.button('ridingmowers_predict'):
        try:
            ridingmowers_prediction = ridingmower_model.predict([[
                float(Income),
                float(LotSize)
            ]])
            if ridingmowers_prediction[0] == 0:
                ridingmowers_predict = 'Not Owner'
            else:
                ridingmowers_predict = 'Owner'
        except Exception as e:
            ridingmowers_predict = f'Error: {e}'

    st.success(ridingmowers_predict)


# ─── Loan ──────────────────────────────────────────────────────────────────────
if selected == 'Loan':
    st.title('Loan Predict')

    person_age = st.text_input('person_age')
    person_gender = st.text_input('person_gender')
    person_education = st.text_input('person_education')
    person_income = st.text_input('person_income')
    person_emp_exp = st.text_input('person_emp_exp')
    person_home_ownership = st.text_input('person_home_ownership')
    loan_amnt = st.text_input('loan_amnt')
    loan_intent = st.text_input('loan_intent')
    loan_int_rate = st.text_input('loan_int_rate')
    loan_percent_income = st.text_input('loan_percent_income')
    cb_person_cred_hist_length = st.text_input('cb_person_cred_hist_length')
    credit_score = st.text_input('credit_score')
    previous_loan_defaults_on_file = st.text_input('previous_loan_defaults_on_file')

    loan_predict = ''

    if st.button('Predict'):
        try:
            loan_prediction = loan_model.predict([[
                float(person_age),
                float(person_gender),
                float(person_education),
                float(person_income),
                float(person_emp_exp),
                float(person_home_ownership),
                float(loan_amnt),
                float(loan_intent),
                float(loan_int_rate),
                float(loan_percent_income),
                float(cb_person_cred_hist_length),
                float(credit_score),
                float(previous_loan_defaults_on_file)
            ]])
            if loan_prediction[0] == 0:
                loan_predict = 'Not Accept'
            else:
                loan_predict = 'Accept'
        except Exception as e:
            loan_predict = f'Error: {e}'

    st.success(loan_predict)
