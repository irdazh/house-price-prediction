import streamlit as st

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import joblib


# Load final model
model = joblib.load('models/final_model.joblib')

# Get expected features
cols = model.feature_names_in_

# ------- Sidebar & user inputs
st.sidebar.header('Input House Details')

# Some options
exter_qual_options = {
    'Excellent': 'Ex',
    'Good': 'Gd',
    'Average/Typical': 'TA',
    'Fair': 'Fa',
    'Poor': 'Po'
}

garage_type_options = {
    'More than one type of garage': '2Types',
    'Attached to home': 'Attchd',
    'Detached from home': 'Detchd',
    'Built-In (Garage part of house - typically has room above garage)': 'BuiltIn',
    'Basement Garage': 'Basment',
    'Car Port': 'CarPort',
    'No Garage': np.nan
}

# Function for options
def categorical_input(label, options_dict):
    choice = st.sidebar.selectbox(label, list(options_dict.keys()))
    return options_dict[choice]

# Make data frame
# OverallQual, GarageCars, ExterQual, GrLivArea, and GarageType
user_inputs = {
    'OverallQual': st.sidebar.slider('Overall Quality (1-10)', min_value=1, max_value=10, value=5),
    'YearBuilt': st.sidebar.slider('Year Built', 1900, 2010, value=1945),
    'GrLivArea': st.sidebar.number_input('Above Ground Living Area (sq ft)', 500, 5000, value=1500),
    'TotalBsmtSF': st.sidebar.number_input('Basement Area (sq ft)', 0, 3000, value=800),
    'ExterQual': categorical_input('Exterior Quality', exter_qual_options),
    'GarageCars': st.sidebar.selectbox('Garage Capacity (cars)', options=[0, 1, 2, 3, 4]),
    'GarageType': categorical_input('GarageType', garage_type_options),

}

# Input all other cols as NA
for col in cols:
    if col not in user_inputs:
        user_inputs[col] = None

# Create a DataFrame from user inputs
input_df = pd.DataFrame([user_inputs], columns=cols)

# Predict the house price
pred= model.predict(input_df)[0]



# ----- Main Area
st.title('House Price Prediction App')
st.markdown('**This app predicts house prices based on user input using an XGB Regressor model trained on Ames Housing Dataset.**')
with st.expander('Model Info'):
    st.markdown('''
    - Model: XGB Regressor
    - Trained on 80% of Ames House Prices from Kaggle
    - Evaluation: 0.138 MSLE & 25,612 RMSE  
                ''')
    

st.subheader('Predicted Sale Price:')
# st.success(f'**${pred:,.1f}**')

# Gauge chart? Fancy predicted price
import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode = 'gauge+number',
    value = pred,
    title = {'text': 'Predicted House Price'},
    gauge = {
        'axis': {'range': [0, 500000]},
        'bar': {'color': 'blue'},
        'steps': [
            {'range': [0,200000], 'color': 'lightgray'},
            {'range': [200000, 400000], 'color': 'gray'}
        ]}
))
st.plotly_chart(fig)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/train.csv')
train_data = load_data()

# Show input
st.write('**User Input for Prediction** (Other unknown house details are set as default)')
with st.expander('Show Inputs (Advanced)', expanded=False):
    st.dataframe(user_inputs)

# Highlight similar listings
similar = train_data[
    (train_data.OverallQual == user_inputs['OverallQual']) & 
    (train_data.GrLivArea >= user_inputs['GrLivArea'] - 500) & 
    (train_data.GrLivArea <= user_inputs['GrLivArea'] + 500) ]

with st.expander('Houses with Similar Properties'):
    fig1, ax1 = plt.subplots()
    sns.histplot(similar.SalePrice, kde=True, ax=ax1, color='tomato')
    ax1.set_title('Sale Price Distribution')
    ax1.axvline(pred, color='skyblue', linestyle='--', label="Your Prediction")
    # ax1.set_xlim(min(similar.SalePrice.min(), pred)-100, max(similar.SalePrice.max(), pred)+100)
    ax1.legend()
    st.pyplot(fig1)

    st.dataframe(similar[['GrLivArea', 'OverallQual', 'SalePrice']].sort_values('SalePrice'))


# Vizzs
st.header('Data Exploration Viz')
st.markdown('Ever wonder how much is the price of houses in Ames, compared to yours?')


#  Histogram of Saleprice. side by side
fig1, ax1 = plt.subplots()
sns.histplot(train_data['SalePrice'], kde=True, ax=ax1, color='tomato')
ax1.set_title('Sale Price Distribution')
ax1.axvline(pred, color='skyblue', linestyle='--', label="Your Prediction")
ax1.legend()

st.subheader('Distribution of Sale Price')
st.pyplot(fig1)


# Another barplot for user
exter = user_inputs['ExterQual']
avg_exter_price = train_data[train_data['ExterQual'] == exter]["SalePrice"].mean()

fig4, ax4 = plt.subplots()
bars = ax4.bar(['Your Price', 'Avg ExterQual Price'],
               [pred, avg_exter_price], color = ['skyblue', 'tomato'])
ax4.set_title(f'Price Comparison in {exter} Quality')
st.pyplot(fig4)

# violinplot, 
col1, col2 = st.columns(2)

fig2, ax2 = plt.subplots()
sns.violinplot(x="ExterQual", y='SalePrice', hue='ExterQual', palette='Set2', data=train_data, ax=ax2)
ax2.set_xlabel('Exterior Quality')
ax2.set_ylabel('House Price')

with col1: 
    st.subheader('Sale Price by Ext. Quality')
    st.pyplot(fig2)
    st.markdown("- Price varies quite significantly across exterior quality;  " \
"'excellent' quality generally have the highest median prices.")

# scatterplot GrLivArea
fig3, ax3 = plt.subplots()
sns.scatterplot(x="GrLivArea", y='SalePrice', 
                data=train_data, ax=ax3, alpha=0.5)
ax3.set_title('Sale Price vs Living Area')
with col2:
    st.subheader('Sale Price vs Living Area')
    st.pyplot(fig3)
    st.markdown("- There's a **positive correlation** between living area and house price.")




    

# Footer
st.markdown('---')
st.markdown('Created by Daud M. Azhari | Data: Ames House Prices from Kaggle | Model: XGB Regressor')




