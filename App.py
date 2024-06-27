import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe = pickle.load(open('pipe.pkl','rb'))

teams = ['Nepal',
 'Scotland',
 'Netherlands',
 'Oman',
 'West Indies',
 'India',
 'England',
 'South Africa',
 'Pakistan',
 'Zimbabwe',
 'New Zealand',
 'Afghanistan',
 'Australia',
 'Ireland',
 'Bangladesh',
 'Sri Lanka',
 'Papua New Guinea',
 'Uganda',
 'Namibia',
 'United States of America']

cities = ['Kirtipur', 'Al Amarat', 'Lauderhill', 'St Lucia', 'Bangalore',
       'Nottingham', 'Cape Town', 'Dubai', 'Johannesburg', 'Harare',
       'Birmingham', 'Wellington', 'Dhaka', 'Hamilton', 'Abu Dhabi',
       'Dublin', 'Chandigarh', 'Sharjah', 'Colombo', 'Pallekele',
       'Southampton', "St George's", 'Melbourne', 'London', 'Accra',
       'Rawalpindi', 'Trinidad', 'Durban', 'Sylhet', 'Hobart',
       'Bridgetown', 'Hyderabad', 'Providence', 'Rajkot', 'Pune',
       'Kolkata', 'Mirpur', 'Mount Maunganui', 'Auckland', 'Guwahati',
       'Nagpur', 'Bristol', 'Kanpur', 'Sydney', 'Centurion', 'Rotterdam',
       'Belfast', 'Chittagong', 'Christchurch', 'Bready', 'Dehradun',
       'Potchefstroom', 'Perth', 'The Hague', 'Windhoek', 'Amstelveen',
       'Roseau', 'Barbados', 'Mumbai', 'Lahore', 'Karachi', 'Cardiff',
       'Bulawayo', 'Dunedin', 'St Vincent', 'Chennai', 'Guyana', 'Indore',
       'Manchester', 'Ranchi', 'Thiruvananthapuram', 'Basseterre',
       'Khulna', 'Adelaide', 'Deventer', 'Bloemfontein', 'Delhi',
       'Canberra', 'Dehra Dun', 'Coolidge', 'Greater Noida', 'Mong Kok',
       'Chester-le-Street', 'St Kitts', 'Hambantota', 'Napier',
       'Port Elizabeth', 'Dharamsala', 'Bengaluru', 'Cuttack',
       'Ahmedabad', 'Gros Islet', 'Brisbane', 'Hangzhou', 'Edinburgh',
       'Chattogram', 'Kampala', 'Kingston', 'Kandy', 'Dambulla',
       'Lucknow', 'Visakhapatnam', 'Antigua', 'Raipur', 'Townsville',
       'Leeds', 'Jamaica', 'Jaipur', 'Carrara', 'East London',
       'Singapore', 'Paarl', 'Tarouba', 'Dharmasala', 'Derry', 'Dominica',
       'Gqeberha', 'Queenstown', 'Fatullah', 'Geelong', 'Nairobi',
       'King City', 'Nelson', 'Londonderry', 'Kimberley', 'Taunton']


st.title('T20 Cricket Score Predictor')

st.image('/Users/yashbhadreshwara/Downloads/ICC-T20-World-Cup-2024.jpg')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city',sorted(cities))

col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for over>5)')
with col5:
    wickets = st.number_input('Wickets Left')

last_five = st.number_input('Runs scored in last 5 overs')

power_play_runs = st.number_input('Runs scored in Powerplay')

if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'power_play_runs': [power_play_runs],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))

st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; padding: 10px; display: flex; justify-content: center; align-items: center; background: transparent;">
    <span style="margin-right: 20px; font-size: 18px;">Connect me here:</span>
        <a href="https://www.instagram.com/yash_bhadreshwara/" target="https://www.instagram.com/yash_bhadreshwara/" style="margin: 0 10px;">
            <i class="fab fa-instagram" style="font-size: 24px;"></i>
        </a>
        <a href="https://www.linkedin.com/in/yash-bhadreshwara/" target="https://www.linkedin.com/in/yash-bhadreshwara/" style="margin: 0 10px;">
            <i class="fab fa-linkedin-in" style="font-size: 24px;"></i>
        </a>
        <a href="https://yashbhadreshwara.github.io" target="https://yashbhadreshwara.github.io" style="margin: 0 10px;">
            <i class="fas fa-briefcase" style="font-size: 24px;"></i>
        </a>
        <a href="https://github.com/YashBhadreshwara" target="https://github.com/YashBhadreshwara" style="margin: 0 10px;">
            <i class="fab fa-github" style="font-size: 24px;"></i>
        </a>
    </div>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
""", unsafe_allow_html=True)