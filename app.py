import streamlit as st
import pandas as pd
import time 
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import os

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

st.title("📊 Attendance Dashboard")

count = st_autorefresh(interval=5000, limit=100, key="refresh")

filepath = f"Attendance/Attendance_{date}.csv"

if os.path.exists(filepath):
    df = pd.read_csv(filepath)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    st.warning("No attendance file found for today yet!")

