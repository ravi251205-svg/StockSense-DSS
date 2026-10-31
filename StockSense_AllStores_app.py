import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="StockSense DSS", page_icon="📦", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
INVENTORY_FILE = BASE_DIR / "Clean_Inventory.csv"
SALES_FILE = BASE_DIR / "Daily_Sales.csv.gz"

st.title("📦 StockSense")
st.subheader("An Intelligent Inventory Replenishment Decision Support System")
st.caption("Retail inventory and sales analysis across all stores and products.")

@st.cache_data(show_spinner="Loading the complete retail dataset...")
def load_data(inventory_file, sales_file):
    inv = pd.read_csv(inventory_file, low_memory=False)
    sales = pd.read_csv(sales_file, low_memory=False)
    inv["Inventory_End_Date"] = pd.to_datetime(inv["Inventory_End_Date"], errors="coerce")
    inv["Inventory_Start_Date"] = pd.to_datetime(inv["Inventory_Start_Date"], errors="coerce")
    sales["Transaction_Date"] = pd.to_datetime(sales["Transaction_Date"], errors="coerce")
    for c in ["Qty_On_Hand", "Stock_Unit_Cost_Price"]:
        inv[c] = pd.to_numeric(inv[c], errors="coerce")
    for c in ["Gross_Units_Sold", "Returned_Units", "Net_Units", "Sales_Amount", "COGS"]:
        sales[c] = pd.to_numeric(sales[c], errors="coerce").fillna(0)
    for df in (inv, sales):
        df["Product_No"] = df["Product_No"].astype(str)
        df["Store"] = df["Store"].astype(str)
    return inv, sales

if not INVENTORY_FILE.exists() or not SALES_FILE.exists():
    st.error("Place Clean_Inventory.csv and Daily_Sales.csv.gz in the same folder as this Python file.")
    st.stop()

try:
    inventory_raw, sales_raw = load_data(str(INVENTORY_FILE), str(SALES_FILE))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.sidebar.header("Data Filters")
valid_inv = inventory_raw[
    inventory_raw["Qty_On_Hand"].notna()
    & (inventory_raw["Qty_On_Hand"] >= 0)
    & (inventory_raw["Stock Data Flag"].fillna("OK") == "OK")
].copy()

store_options = ["All Stores"] + sorted(valid_inv["Store"].dropna().unique().tolist())
selected_store = st.sidebar.selectbox("Store coverage", store_options)
if selected_store == "All Stores":
    store_inv = valid_inv.copy()
    store_sales = sales_raw.copy()
else:
    store_inv = valid_inv[valid_inv["Store"] == selected_store].copy()
    store_sales = sales_raw[sales_raw["Store"] == selected_store].copy()

if store_inv.empty:
    st.error("No valid inventory records are available for this selection.")
    st.stop()

st.sidebar.subheader("Replenishment Assumptions")
service_factor = st.sidebar.selectbox("Service level factor (Z)", [1.28, 1.645, 1.96, 2.33], index=1)
review_period = st.sidebar.number_input("Inventory review period (days)", min_value=1, max_value=60, value=7)
lead_time = st.sidebar.number_input("Assumed supplier lead time (days)", min_value=1, max_value=90, value=7,
    help="Supplier lead time is not present in the source dataset. This is an editable assumption.")
ordering_cost = st.sidebar.number_input("Assumed ordering cost per order (₹)", min_value=1.0, value=200.0, step=50.0)
annual_holding_rate = st.sidebar.number_input("Assumed annual holding cost (% of unit cost)", min_value=1.0, max_value=100.0, value=20.0)
st.sidebar.caption("Lead time, ordering cost and holding rate are assumptions, not observed dataset fields.")

def product_daily_history(product_no, store):
    p = store_sales[
        (store_sales["Product_No"] == product_no) &
        (store_sales["Store"] == store)
    ].copy()
    if p.empty:
        return pd.DataFrame(columns=["Date", "Units_Sold"])
    daily = p.groupby("Transaction_Date", as_index=False)["Gross_Units_Sold"].sum()
    daily = daily.rename(columns={"Transaction_Date": "Date", "Gross_Units_Sold": "Units_Sold"})
    full_dates = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    return (daily.set_index("Date").reindex(full_dates, fill_value=0)
            .rename_axis("Date").reset_index())

# Build analysis for every valid product-store combination in the selected coverage.
rows = []
latest_inventory = store_inv.sort_values(
    ["Product_No", "Store", "Inventory_End_Date", "Inventory_Start_Date"]
).drop_duplicates(["Product_No", "Store"], keep="last")

for _, inv_row in latest_inventory.iterrows():
    product_no = inv_row["Product_No"]
    store = inv_row["Store"]
    hist = product_daily_history(product_no, store)
    if hist.empty:
        avg_demand, demand_std = 0.0, 0.0
    else:
        recent = hist.tail(min(30, len(hist)))["Units_Sold"]
        avg_demand = float(recent.mean())
        demand_std = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
        if np.isnan(demand_std):
            demand_std = 0.0

    stock = float(inv_row["Qty_On_Hand"])
    unit_cost = float(inv_row["Stock_Unit_Cost_Price"]) if pd.notna(inv_row["Stock_Unit_Cost_Price"]) else 0.0
    safety = service_factor * demand_std * np.sqrt(lead_time)
    rop = avg_demand * lead_time + safety
    annual_demand = avg_demand * 365
    holding_cost = unit_cost * annual_holding_rate / 100
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if annual_demand > 0 and holding_cost > 0 else 0.0
    target = avg_demand * (lead_time + review_period) + safety
    order_qty = int(np.ceil(max(0, target - stock))) if stock < target else 0
    status = "Reorder Required" if stock <= rop else ("Low Stock" if stock < target else "Sufficient Stock")

    rows.append({
        "Store": store,
        "Product No": product_no,
        "Product": inv_row.get("Product_Description", ""),
        "Current Stock": stock,
        "Average Daily Demand": avg_demand,
        "Demand Std Dev": demand_std,
        "Lead Time (Days)": lead_time,
        "Safety Stock": safety,
        "Reorder Point": rop,
        "EOQ": eoq,
        "Target Stock": target,
        "Recommended Order Qty": order_qty,
        "Unit Cost (₹)": unit_cost,
        "Estimated Order Value (₹)": order_qty * unit_cost,
        "Status": status,
        "Supplier": inv_row.get("Supplier", "")
    })

analysis = pd.DataFrame(rows)
if analysis.empty:
    st.warning("No product-store combinations were available for analysis.")
    st.stop()

st.header("1. Complete Inventory Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stores included", f"{analysis['Store'].nunique():,}")
c2.metric("Product-store records", f"{len(analysis):,}")
c3.metric("Total stock units", f"{analysis['Current Stock'].sum():,.0f}")
c4.metric("Reorder required", f"{(analysis['Status'] == 'Reorder Required').sum():,}")
c5.metric("Estimated order value", f"₹{analysis['Estimated Order Value (₹)'].sum():,.2f}")
st.caption(f"Coverage: {selected_store} | All-store inventory records flagged for review: {(inventory_raw['Stock Data Flag'].fillna('OK') != 'OK').sum():,}")

st.divider()
st.header("2. Inventory Analysis")
status_filter = st.multiselect(
    "Filter stock status",
    options=["Reorder Required", "Low Stock", "Sufficient Stock"],
    default=["Reorder Required", "Low Stock", "Sufficient Stock"]
)
display_analysis = analysis[analysis["Status"].isin(status_filter)]
st.dataframe(display_analysis.round(2), use_container_width=True, hide_index=True, height=420)

st.subheader("Stock Status Distribution")
status_counts = analysis["Status"].value_counts().rename_axis("Status").reset_index(name="Product-store records")
fig_status = px.bar(status_counts, x="Status", y="Product-store records", title="Inventory status across selected stores")
st.plotly_chart(fig_status, use_container_width=True)

st.subheader("Highest Recommended Order Quantities")
top_orders = analysis[analysis["Recommended Order Qty"] > 0].nlargest(20, "Recommended Order Qty")
if not top_orders.empty:
    fig_orders = px.bar(top_orders, x="Product No", y="Recommended Order Qty", color="Store",
                        hover_data=["Product", "Status"], title="Top 20 product-store replenishment recommendations")
    st.plotly_chart(fig_orders, use_container_width=True)
else:
    st.info("No products require replenishment under the current assumptions.")

st.header("3. Demand Forecasting")
forecast_pairs = analysis[["Store", "Product No", "Product"]].drop_duplicates().copy()
forecast_pairs["Selection"] = forecast_pairs["Store"] + " | " + forecast_pairs["Product No"] + " | " + forecast_pairs["Product"].astype(str)
forecast_selection = st.selectbox("Select store and product", forecast_pairs["Selection"].tolist())
chosen = forecast_pairs.loc[forecast_pairs["Selection"] == forecast_selection].iloc[0]
selected_product = chosen["Product No"]
selected_product_store = chosen["Store"]
history = product_daily_history(selected_product, selected_product_store)
forecast_horizon = st.slider("Forecast horizon (days)", 1, 30, 7)
if history.empty:
    st.info("No sales history is available for this product-store combination.")
else:
    lookback = min(30, len(history))
    forecast_mean = float(history.tail(lookback)["Units_Sold"].mean())
    future_dates = pd.date_range(history["Date"].max() + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    forecast = pd.DataFrame({"Date": future_dates, "Forecasted Demand": [round(forecast_mean, 2)] * forecast_horizon})
    fig2 = px.line(history, x="Date", y="Units_Sold", title=f"Historical Demand and Moving Average Forecast — {selected_product_store}")
    fig2.add_scatter(x=forecast["Date"], y=forecast["Forecasted Demand"], mode="lines+markers", name="Moving average forecast")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.caption("Baseline forecast: average gross units sold over the latest 30 calendar days. This is not a trained machine-learning model.")

st.header("4. Replenishment Recommendations")
recommendations = analysis[analysis["Recommended Order Qty"] > 0].copy()
if recommendations.empty:
    st.success("No replenishment is recommended under the current assumptions.")
else:
    st.warning(f"{len(recommendations):,} product-store combinations need stock attention.")
    st.dataframe(recommendations.sort_values("Estimated Order Value (₹)", ascending=False).round(2),
                 use_container_width=True, hide_index=True, height=420)

st.header("5. What-if Scenario Analysis")
scenario_selection = st.selectbox("Choose a product-store combination", forecast_pairs["Selection"].tolist(), key="scenario_pair")
scenario_chosen = forecast_pairs.loc[forecast_pairs["Selection"] == scenario_selection].iloc[0]
scenario_product = scenario_chosen["Product No"]
scenario_store = scenario_chosen["Store"]
r = analysis[(analysis["Product No"] == scenario_product) & (analysis["Store"] == scenario_store)].iloc[0]
demand_change = st.slider("Assumed change in daily demand (%)", -50, 100, 20, 5)
scenario_demand = max(0.0, r["Average Daily Demand"] * (1 + demand_change / 100))
scenario_safety = service_factor * r["Demand Std Dev"] * np.sqrt(lead_time)
scenario_rop = scenario_demand * lead_time + scenario_safety
scenario_target = scenario_demand * (lead_time + review_period) + scenario_safety
scenario_qty = int(np.ceil(max(0, scenario_target - r["Current Stock"]))) if r["Current Stock"] < scenario_target else 0
s1, s2, s3 = st.columns(3)
s1.metric("Scenario daily demand", f"{scenario_demand:.2f}")
s2.metric("Scenario reorder point", f"{scenario_rop:.2f}")
s3.metric("Scenario order quantity", f"{scenario_qty:,} units")

st.header("6. Data Quality Notes")
flagged = inventory_raw[inventory_raw["Stock Data Flag"].fillna("OK") != "OK"]
st.write(f"The source inventory contains **{len(flagged):,} records** flagged for review, including negative or missing stock quantities. These records are excluded from replenishment calculations.")
st.write("Supplier lead time, ordering cost and annual holding-cost rate are not provided in the source data. The values in the sidebar are editable assumptions, not actual supplier records.")

st.header("7. Download Reports")
st.download_button("Download Complete Inventory Analysis (CSV)", analysis.to_csv(index=False).encode("utf-8"),
                   "stocksense_all_store_inventory_analysis.csv", "text/csv")
if not history.empty:
    st.download_button("Download Selected Product Forecast (CSV)", forecast.to_csv(index=False).encode("utf-8"),
                       "stocksense_selected_product_forecast.csv", "text/csv")

st.divider()
st.caption("StockSense DSS | MAT5024 Decision Support Systems | Decision-support prototype; verify recommendations before placing supplier orders.")
