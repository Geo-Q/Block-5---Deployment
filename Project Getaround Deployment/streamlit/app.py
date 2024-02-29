import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np


### Config
st.set_page_config(
    page_title="Getaround",
    page_icon="🚗",
    layout="wide"
)

# App
st.title("Getaround")
st.markdown("👋 Hello there ! Welcome to this Getaround app. We simply track the evolution of cases accross the world.\
    Data comes from the European Centre for Disease Prevention and Control (ECDC).")
st.markdown("You will find here analysis performed on dataset provided by Get Around company from their records.")
st.markdown("Enjoy this simple web dashboard.")
st.markdown("Check out our data here ⬇️")

### Import data
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_excel("get_around_delay_analysis.xlsx")
    data['late'] = data['delay_at_checkout_in_minutes'].apply(lambda x : "on time" if x <= 0 else 'late' if x > 0 else 'NA')
    data['delay'] = data['delay_at_checkout_in_minutes'].apply(lambda x : "on time" if x < 0 else 'late -30min' if x < 30 
                                                            else 'late -1h' if x < 60 else 'late -2h' if x < 120
                                                            else 'late -4h' if x < 240 else 'late +4h' if x>= 240 else 'NA')
    return data

def load_price():
    price = pd.read_csv("get_around_pricing_project.csv")
    return price

data_load_state = st.text('Loading data...')
data = load_data()
price = load_price()

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data) 

## Header
st.header("First analysis")
st.subheader("Delays proportion")
avg_delay = data['delay_at_checkout_in_minutes'].mean()
med_delay = data['delay_at_checkout_in_minutes'].median()

#### CREATE TWO COLUMNS
col1, col2, col3 = st.columns(3)

with col1:
        fig = px.pie(data[data['late'] != 'NA'], names="late")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"There are more users coming late than on time.\
        It can be problematic for the next user taking a car, and for client satisfaction of Getaround.\
        Average delay (including poeple on time) is {round(avg_delay, 2)} minutes, and median is {round(med_delay,2)} minutes.")

with col2:
        st.subheader("Types proportion")
        fig = px.pie(data,names="checkin_type")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"The biggest part of checkin is on mobile. It represents {len(data[data['checkin_type'] == 'mobile'])} reservations.")
    
with col3:
        st.subheader("State proportion")
        fig = px.pie(data,names="state")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"{round(len(data[data['state'] == 'canceled']) / len(data), 2)} percent of reservation are canceled. It represents {len(data[data['state'] == 'canceled'])} reservations. \
            Is it due to delay of previous users?")


st.header("Delays")

st.subheader("Being late by checkin type")
st.markdown("Let's see if users are late in function of checkin type.")
fig = px.histogram(data[data['late'] != 'NA'], x="late", color="checkin_type", barmode = "group",
                    width = 700, height = 500, histnorm = "percent", text_auto = True)
fig.update_traces(textposition = "outside")
st.plotly_chart(fig, use_container_width=True)
st.markdown("60 percent of mobile checkin are late whereas 40 percent for connect checkin.\
    If we want to resolve the delay problem, we should perhaps focus on the mobile checkin mode.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Delays distribution")
    color_discrete_sequence = ["green", "yellow", "orange", "red", "black"]
    st.markdown("Let's see how much are the users late.")
    fig = px.histogram(data[((data['delay'] != 'NA') & (data['delay'] != 'on time'))].sort_values(by="delay_at_checkout_in_minutes"),
                        x="delay", color="delay", color_discrete_sequence=color_discrete_sequence)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Biggest part of delay are under 2h. It's perhaps at 2h that should be the threshold")

with col2:
    st.subheader("Delays distribution per checkin type")
    st.markdown("Let's see how much are the users late in function of checkin type.")
    fig = px.histogram(data[((data['delay'] != 'NA') & (data['delay'] != 'on time'))].sort_values(by="delay_at_checkout_in_minutes"),
                        x="delay", color="delay", facet_row="checkin_type", color_discrete_sequence=color_discrete_sequence)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Mobile checkin type is more concerned by delays. For both checkin type, biggest part of delays is for less than 2h.")


st.header("Cancelation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("State by checkin type")
    st.markdown("Let's see the proportion of canceled car in function of checkin type.")
    fig = px.histogram(data, x = "state", color = "checkin_type", barmode ="group",
                        width= 700, height = 500, histnorm = "percent", text_auto = True)
    fig.update_traces(textposition = "outside")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("It seems a big part of cancelation is from connect checkin type.")

with col2:
    st.subheader("Checkin type and delay of previous car when canceled")
    st.markdown("Let's see, when a car is canceled, what was the checkin type of the previous customer and if he was late.")
    cancel = data[data['state'] == 'canceled']
    cancel.dropna(subset=['previous_ended_rental_id'], inplace=True)
    cancel = cancel.merge(data, left_on='previous_ended_rental_id', right_on='rental_id')
    fig = px.histogram(cancel[cancel['late_y'] != 'NA'], x = "late_y", color = "checkin_type_y",
                        barmode ="group", width= 700, height = 500, text_auto = True)
    fig.update_traces(textposition = "outside")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("The biggest part of previous reservations when a car is canceled is from connect checkin, but most of them were on time. \
        Mobile checkin represent a lowest part, but most of them were late.")


st.header("Calculating threshold")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Threshold")
    st.markdown("Let's remove outlier and keep only positive delays to visualize the distribution of delays and estimate a threshold.")
    df = data.copy()
    df.dropna(subset=['delay_at_checkout_in_minutes'], inplace=True)
    mean = df['delay_at_checkout_in_minutes'].mean()
    std = df['delay_at_checkout_in_minutes'].std()
    late_drivers = df[((df['delay_at_checkout_in_minutes'] < (mean + 3 * std)) & (df['delay_at_checkout_in_minutes'] > 0))]
    fig = px.histogram(late_drivers, x="delay_at_checkout_in_minutes", histfunc='count',
                        nbins=(int(len(late_drivers['delay_at_checkout_in_minutes'])/10)))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Most of delays are under 2h. \
        Let's see difference between time delta recorded and delay per checkin type.")

delta = df.dropna(subset='time_delta_with_previous_rental_in_minutes')
delta = delta.dropna(subset='delay_at_checkout_in_minutes')
delta['delta'] = delta['time_delta_with_previous_rental_in_minutes'] - delta['delay_at_checkout_in_minutes']
delta['enough_delta'] = delta['delta'].apply(lambda x: 'yes' if x >=0 else 'no')
yes_mean = delta[delta['enough_delta'] == 'yes']['time_delta_with_previous_rental_in_minutes'].mean()
no_mobile_mean = delta[(delta['enough_delta'] == "no") & (delta['checkin_type'] == "mobile")]['time_delta_with_previous_rental_in_minutes'].mean()
no_mobile_delta_mean = delta[(delta['enough_delta'] == "no") & (delta['checkin_type'] == "mobile")]['delta'].mean()
no_connect_mean = delta[(delta['enough_delta'] == "no") & (delta['checkin_type'] == "connect")]['time_delta_with_previous_rental_in_minutes'].mean()
no_connect_delta_mean = delta[(delta['enough_delta'] == "no") & (delta['checkin_type'] == "connect")]['delta'].mean()

with col2:

    fig = px.histogram(delta, x="enough_delta", color="checkin_type", barmode = "group")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"We can see that most the car had enough time delta with the next when we have a time delta with previous rental car recorded.\
            Median delay for cars when they have enough time delta is {round(yes_mean, 2)} minutes.\
            For car who hasn't enough time delta, there is a biggest proportion of mobile checkin.\
            It's on mobile checkin type that we should focus to apply a threshold.\
            Time delta mean for late mobile checkin mean is {round(no_mobile_mean,2)} minutes, and the total delta is {round(no_mobile_delta_mean,2)} minutes.\
            Time delta mean for late connect checkin mean is {round(no_connect_mean,2)} minutes, and the total delta is {round(no_connect_delta_mean,2)} minutes.")

st.markdown("To apply a threshold, as we said previouly, we should focus on mobile checkin type. \
        I choose to take the median of late drivers, what is more relevent than mean that can be biased with outliers.")

global_median = late_drivers['delay_at_checkout_in_minutes'].median()
mobile_median =  late_drivers[late_drivers['checkin_type'] == 'mobile']['delay_at_checkout_in_minutes'].median()
connect_median = late_drivers[late_drivers['checkin_type'] == 'connect']['delay_at_checkout_in_minutes'].median()
global_mean = late_drivers['delay_at_checkout_in_minutes'].mean()
mobile_mean =  late_drivers[late_drivers['checkin_type'] == 'mobile']['delay_at_checkout_in_minutes'].mean()
connect_mean = late_drivers[late_drivers['checkin_type'] == 'connect']['delay_at_checkout_in_minutes'].mean()
st.markdown(f"Global median is {global_median} minutes.")
st.markdown(f"Median for connect checkin is {connect_median} minutes.")
st.markdown(f"Median for mobile checkin is {mobile_median} minutes, and it is the threshold we should apply.")


st.header("Money loss")

mean_price_per_min = (price['rental_price_per_day'].mean() / 24 / 60)

col1, col2 = st.columns(2)

with col1:
    st.markdown("Let's see how much money we should loss if we apply a threshold for connect ckeckin.")
    nb_connect = len(late_drivers[late_drivers['checkin_type'] == 'connect'])
    connect_money_loss = nb_connect * connect_mean * mean_price_per_min
    st.markdown(f"For connect checkin, the money loss is about {round(connect_money_loss, 2)} $ for {nb_connect} delays concerned.")

with col2:
    st.markdown("Let's see how much money we should loss if we apply a threshold for mobile ckeckin.")
    nb_mobile = len(late_drivers[late_drivers['checkin_type'] == 'mobile'])
    mobile_money_loss = nb_mobile * mobile_mean * mean_price_per_min
    st.markdown(f"For mobile checkin, the money loss is about {round(mobile_money_loss, 2)} $ for {nb_mobile} delays concerned.")

cancel_money_loss = 6 * (mean_price_per_min *60) * len(cancel[cancel['late_y'] == 'late'])

st.subheader("Hypothetic global money loss")
st.markdown(f"To have the exact money loss, it should be good to have the information about location duration. \
        We should be able to estimate the money loss for cancelation due to delays to add to money loss due to delays.\
        If we imagine to have a 6h average location duration, and suppose that all canceled reservation with a previous car late are due to delays,\
        we have to add a money loss of {round(cancel_money_loss, 2)} $ for cancelation.")
st.markdown(f"If we estimate the global money loss without threshold, it is aroud {round(connect_money_loss + mobile_money_loss + cancel_money_loss,2)} $ of loss.")





# commandes pour tester sur terminal :
# cd Code/Geo-Q/Jedha/Fullstack/M8_Deployment/Project_Getaround/streamlit
# streamlit run --server.port 80 --server.address 0.0.0.0 app.py  

# image et app : getaround-streamlit-gqforjedha

# delpoy : 
# heroku login
# heroku container:login
# heroku create getaround-streamlit-gqforjedha
# docker buildx build . --platform linux/amd64 -t getaround-streamlit-gqforjedha
# docker tag getaround-streamlit-gqforjedha registry.heroku.com/getaround-streamlit-gqforjedha/web
# docker push registry.heroku.com/getaround-streamlit-gqforjedha/web
# heroku container:release web -a getaround-streamlit-gqforjedha
# heroku open -a getaround-streamlit-gqforjedha