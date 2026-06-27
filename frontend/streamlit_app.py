"""Local Streamlit UI for the SuperKart API.

Points at the local Flask service by default. Override with the SUPERKART_API env var.
The input schema matches the training features exactly (no training/serving skew).
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("SUPERKART_API", "http://127.0.0.1:7860").rstrip("/") + "/predict"

PRODUCT_TYPES = [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
    "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
    "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
]

st.title("SuperKart Sales Forecasting")
st.caption(f"Local model API: {API_URL}")

col1, col2 = st.columns(2)
with col1:
    Product_Weight = st.number_input("Product Weight", 0.0, value=12.66)
    Product_MRP = st.number_input("Product MRP", 0.0, value=150.0)
    Product_Allocated_Area = st.number_input("Allocated Area", 0.0, 1.0, value=0.07)
    Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Type = st.selectbox("Product Type", PRODUCT_TYPES, index=4)
with col2:
    Store_Id = st.selectbox("Store Id", ["OUT001", "OUT002", "OUT003", "OUT004"])
    Store_Establishment_Year = st.selectbox("Store Establishment Year", [1987, 1998, 1999, 2009])
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox(
        "Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
    )

payload = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Id": Store_Id,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
}

if st.button("Predict", type="primary"):
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        if r.ok:
            st.success(f"Predicted store sales: ₹{r.json()['predicted_sales']:,.2f}")
        else:
            st.error(r.json().get("error", "request failed"))
    except requests.exceptions.RequestException:
        st.error(f"Could not reach the API at {API_URL}. Is the Flask service running?")
