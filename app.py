import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="StockSense DSS", page_icon="📦", layout="wide")


st.markdown("""
<style>
    /* Dark modern StockSense theme */
    :root {
        --ss-bg: #0b1220;
        --ss-panel: #111c2e;
        --ss-panel-2: #162338;
        --ss-border: #26364d;
        --ss-text: #e7eef8;
        --ss-muted: #9aabc1;
        --ss-teal: #20c6b7;
        --ss-blue: #4f8cff;
    }
    .stApp {
        background: var(--ss-bg);
        color: var(--ss-text);
    }
    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.92);
    }
    [data-testid="stSidebar"] {
        background: #08101d;
        border-right: 1px solid var(--ss-border);
    }
    [data-testid="stSidebar"] * {
        color: var(--ss-text);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        color: #f3f7fc !important;
        letter-spacing: -0.02em;
    }
    p, label, .stCaption, [data-testid="stMarkdownContainer"] {
        color: var(--ss-text);
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #142238, #101b2d);
        border: 1px solid var(--ss-border);
        padding: 17px 18px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,.16);
    }
    [data-testid="stMetricLabel"] {
        color: var(--ss-muted) !important;
    }
    [data-testid="stMetricValue"] {
        color: #f3f7fc !important;
        font-weight: 700;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--ss-border);
        border-radius: 12px;
        overflow: hidden;
    }
    div[data-testid="stPlotlyChart"] {
        background: var(--ss-panel);
        border: 1px solid var(--ss-border);
        border-radius: 15px;
        padding: 10px;
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #13a99d, #287bdc);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border: 1px solid #55e0d3;
        color: white;
    }
    div[role="radiogroup"] label {
        background: #111c2e;
        border: 1px solid #26364d;
        padding: 9px 12px;
        border-radius: 9px;
        margin-bottom: 7px;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stNumberInput input {
        background: #111c2e;
        color: #e7eef8;
        border-color: #31445f;
        border-radius: 9px;
    }
    hr {
        border-color: #26364d;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    .stocksense-brand {
        background: linear-gradient(120deg, #12243a, #102b35);
        border: 1px solid #25465b;
        padding: 18px 22px;
        border-radius: 16px;
        margin-bottom: 20px;
    }
    .stocksense-brand small {
        color: #8fa7c2;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .stocksense-brand h1 {
        margin: 5px 0 2px 0;
        font-size: 2rem;
    }
    .stocksense-brand p {
        margin: 0;
        color: #a9bbcf;
    }
</style>
""", unsafe_allow_html=True)


BASE_DIR = Path(__file__).resolve().parent
INVENTORY_FILE = BASE_DIR / "Clean_Inventory.csv"
SALES_FILE = BASE_DIR / "Daily_Sales.csv.gz"

st.markdown("""
<div class="stocksense-brand">
  <small>SUPERMARKET DECISION INTELLIGENCE</small>
  <h1>📦 StockSense</h1>
  <p>Inventory control · Warehouse visibility · Demand forecasting · Smarter replenishment</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading supermarket data...")
def load_data(inventory_file, sales_file):
    inv = pd.read_csv(inventory_file, low_memory=False)
    sales = pd.read_csv(sales_file, low_memory=False, compression="infer")

    inv["Inventory_End_Date"] = pd.to_datetime(inv["Inventory_End_Date"], errors="coerce")
    inv["Inventory_Start_Date"] = pd.to_datetime(inv["Inventory_Start_Date"], errors="coerce")
    sales["Transaction_Date"] = pd.to_datetime(sales["Transaction_Date"], errors="coerce")

    for c in ["Qty_On_Hand", "Warehouse_Qty", "Stock_Unit_Cost_Price"]:
        if c not in inv.columns:
            inv[c] = 0
        inv[c] = pd.to_numeric(inv[c], errors="coerce").fillna(0)

    for c in ["Gross_Units_Sold", "Returned_Units", "Net_Units", "Sales_Amount", "COGS"]:
        if c not in sales.columns:
            sales[c] = 0
        sales[c] = pd.to_numeric(sales[c], errors="coerce").fillna(0)

    for df in (inv, sales):
        df["Product_No"] = df["Product_No"].astype(str)
        df["Store"] = df["Store"].astype(str)

    return inv, sales

if not INVENTORY_FILE.exists() or not SALES_FILE.exists():
    st.error("Please place Clean_Inventory.csv and Daily_Sales.csv.gz in the same folder as this Python file.")
    st.stop()

try:
    inventory_raw, sales_raw = load_data(str(INVENTORY_FILE), str(SALES_FILE))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

valid_inv = inventory_raw[
    inventory_raw["Qty_On_Hand"].notna()
    & (inventory_raw["Qty_On_Hand"] >= 0)
    & (inventory_raw["Warehouse_Qty"] >= 0)
    & (inventory_raw["Stock Data Flag"].fillna("OK") == "OK")
].copy()

if valid_inv.empty:
    st.error("No valid inventory records are available.")
    st.stop()

latest_inventory = (
    valid_inv.sort_values(["Product_No", "Inventory_End_Date", "Inventory_Start_Date"])
    .drop_duplicates(["Product_No"], keep="last")
)

# A single-supermarket project: combine the selected supermarket's product records.
store_name = str(latest_inventory["Store"].iloc[0])
sales_store = sales_raw[sales_raw["Store"] == store_name].copy()

@st.cache_data
def build_daily_histories(sales):
    histories = {}
    if sales.empty:
        return histories
    grouped = sales.groupby(["Product_No", "Store", "Transaction_Date"])["Gross_Units_Sold"].sum().reset_index()
    for (product_no, store), group in grouped.groupby(["Product_No", "Store"]):
        daily = group.set_index("Transaction_Date")["Gross_Units_Sold"].sort_index()
        dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        histories[(str(product_no), str(store))] = (
            daily.reindex(dates, fill_value=0).rename_axis("Date").reset_index(name="Units_Sold")
        )
    return histories

histories = build_daily_histories(sales_store)

# Replenishment assumptions are explicitly user-adjustable.
st.sidebar.header("StockSense Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analysis", "Forecast", "What-If Analysis"])
st.sidebar.divider()
st.sidebar.subheader("Planning assumptions")
service_factor = st.sidebar.selectbox("Service level factor (Z)", [1.28, 1.645, 1.96, 2.33], index=1)
review_period = st.sidebar.number_input("Review period (days)", min_value=1, max_value=60, value=7)
lead_time = st.sidebar.number_input("Supplier lead time (days)", min_value=1, max_value=90, value=7)
st.sidebar.caption("Supplier lead time and service-level factor are planning assumptions, not measured supplier data.")

# Build product-level analysis.
rows = []
for _, inv_row in latest_inventory.iterrows():
    product_no = str(inv_row["Product_No"])
    hist = histories.get((product_no, store_name), pd.DataFrame())

    if hist.empty:
        avg_demand, demand_std = 0.0, 0.0
    else:
        recent = hist.tail(min(30, len(hist)))["Units_Sold"]
        avg_demand = float(recent.mean())
        demand_std = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
        if not np.isfinite(demand_std):
            demand_std = 0.0

    store_stock = float(inv_row["Qty_On_Hand"])
    warehouse_stock = float(inv_row["Warehouse_Qty"])
    unit_cost = float(inv_row["Stock_Unit_Cost_Price"])
    safety_stock = service_factor * demand_std * np.sqrt(lead_time)
    reorder_point = avg_demand * lead_time + safety_stock
    target_stock = avg_demand * (lead_time + review_period) + safety_stock
    required_qty = max(0, target_stock - store_stock)
    transfer_qty = min(warehouse_stock, required_qty)
    supplier_qty = max(0, required_qty - transfer_qty)

    if store_stock <= reorder_point:
        status = "Reorder Required"
    elif store_stock < target_stock:
        status = "Low Stock"
    else:
        status = "Sufficient Stock"

    rows.append({
        "Product No": product_no,
        "Product": inv_row.get("Product_Description", ""),
        "Store Stock": store_stock,
        "Warehouse Stock": warehouse_stock,
        "Total Available Stock": store_stock + warehouse_stock,
        "Average Daily Demand": avg_demand,
        "Demand Std Dev": demand_std,
        "Safety Stock": safety_stock,
        "Reorder Point": reorder_point,
        "Target Stock": target_stock,
        "Recommended Transfer Qty": int(np.ceil(transfer_qty)),
        "Supplier Purchase Qty": int(np.ceil(supplier_qty)),
        "Unit Cost (₹)": unit_cost,
        "Estimated Purchase Value (₹)": np.ceil(supplier_qty) * unit_cost,
        "Status": status,
        "Supplier": inv_row.get("Supplier", "")
    })

analysis = pd.DataFrame(rows)
if analysis.empty:
    st.warning("No products were available for analysis.")
    st.stop()

product_options = analysis[["Product No", "Product"]].drop_duplicates()
product_options["Selection"] = product_options["Product No"] + " | " + product_options["Product"].astype(str)

def get_selected_product(selection):
    selected = product_options.loc[product_options["Selection"] == selection].iloc[0]
    product_no = selected["Product No"]
    row = analysis.loc[analysis["Product No"] == product_no].iloc[0]
    hist = histories.get((str(product_no), store_name), pd.DataFrame())
    return row, hist

if page == "Home":
    st.header("Home Dashboard")
    st.write(f"Welcome to StockSense. Current supermarket: **{store_name}**.")

    total_store = analysis["Store Stock"].sum()
    total_warehouse = analysis["Warehouse Stock"].sum()
    total_products = len(analysis)
    reorder_count = int((analysis["Status"] == "Reorder Required").sum())
    low_count = int((analysis["Status"] == "Low Stock").sum())
    purchase_value = analysis["Estimated Purchase Value (₹)"].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Products monitored", f"{total_products:,}")
    k2.metric("Store stock units", f"{total_store:,.0f}")
    k3.metric("Warehouse stock units", f"{total_warehouse:,.0f}")
    k4, k5, k6 = st.columns(3)
    k4.metric("Reorder required", f"{reorder_count:,}")
    k5.metric("Low-stock products", f"{low_count:,}")
    k6.metric("Estimated supplier purchase value", f"₹{purchase_value:,.2f}")

    st.subheader("Stock status")
    status_counts = analysis["Status"].value_counts().rename_axis("Status").reset_index(name="Products")
    st.plotly_chart(px.pie(status_counts, names="Status", values="Products", title="Product stock status", template="plotly_dark"), use_container_width=True)

    st.subheader("Products requiring attention")
    attention = analysis[analysis["Status"] != "Sufficient Stock"].copy()
    if attention.empty:
        st.success("No products currently require stock attention under the selected assumptions.")
    else:
        st.dataframe(attention[["Product", "Store Stock", "Warehouse Stock", "Average Daily Demand", "Status",
                                "Recommended Transfer Qty", "Supplier Purchase Qty"]].round(2),
                     use_container_width=True, hide_index=True)

    st.info("Use the navigation menu to open Analysis, Forecast and What-If Analysis.")

elif page == "Analysis":
    st.header("Inventory & Sales Analysis")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Products", f"{len(analysis):,}")
    a2.metric("Total store stock", f"{analysis['Store Stock'].sum():,.0f}")
    a3.metric("Total warehouse stock", f"{analysis['Warehouse Stock'].sum():,.0f}")
    a4.metric("Products needing attention", f"{(analysis['Status'] != 'Sufficient Stock').sum():,}")

    st.subheader("Product inventory table")
    status_filter = st.multiselect(
        "Filter by stock status",
        ["Reorder Required", "Low Stock", "Sufficient Stock"],
        default=["Reorder Required", "Low Stock", "Sufficient Stock"]
    )
    filtered = analysis[analysis["Status"].isin(status_filter)]
    st.dataframe(filtered.round(2), use_container_width=True, hide_index=True, height=420)

    st.subheader("Store and warehouse quantity comparison")
    quantity_long = analysis[["Product", "Store Stock", "Warehouse Stock"]].melt(
        id_vars="Product", var_name="Stock Location", value_name="Quantity"
    )
    top_qty_products = analysis.nlargest(15, "Total Available Stock")["Product"].tolist()
    quantity_long = quantity_long[quantity_long["Product"].isin(top_qty_products)]
    st.plotly_chart(px.bar(quantity_long, x="Product", y="Quantity", color="Stock Location",
                           barmode="group", title="Stock quantity for the 15 products with highest total stock", template="plotly_dark"),
                    use_container_width=True)

    st.subheader("Sales trend")
    if not sales_store.empty:
        sales_trend = sales_store.groupby("Transaction_Date", as_index=False)["Net_Units"].sum()
        st.plotly_chart(px.line(sales_trend, x="Transaction_Date", y="Net_Units",
                                title="Daily net units sold across the supermarket", template="plotly_dark"),
                        use_container_width=True)
    else:
        st.info("No sales records are available for this supermarket.")

    st.subheader("Download inventory analysis")
    st.download_button("Download Inventory Analysis CSV", analysis.to_csv(index=False).encode("utf-8"),
                       "stocksense_inventory_analysis.csv", "text/csv")

elif page == "Forecast":
    st.header("Product Demand Forecast")
    selection = st.selectbox("Choose a product", product_options["Selection"].tolist(), key="forecast_product")
    row, history = get_selected_product(selection)
    horizon = st.slider("Forecast horizon (days)", 7, 30, 7, key="forecast_horizon")

    if history.empty:
        st.warning("No historical sales data is available for this product.")
    else:
        lookback = min(30, len(history))
        forecast_mean = float(history.tail(lookback)["Units_Sold"].mean())
        future_dates = pd.date_range(history["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        forecast = pd.DataFrame({
            "Date": future_dates,
            "Forecasted Demand": [max(0, forecast_mean)] * horizon
        })
        historical_plot = history.tail(90).copy()
        fig = px.line(historical_plot, x="Date", y="Units_Sold", title=f"Historical sales and {horizon}-day demand forecast", template="plotly_dark")
        fig.add_scatter(x=forecast["Date"], y=forecast["Forecasted Demand"],
                        mode="lines+markers", name="Forecast")
        st.plotly_chart(fig, use_container_width=True)

        f1, f2, f3 = st.columns(3)
        f1.metric("Average daily demand", f"{forecast_mean:.2f} units")
        f2.metric("Forecast horizon", f"{horizon} days")
        f3.metric("Total forecast demand", f"{forecast['Forecasted Demand'].sum():,.0f} units")
        st.subheader("Forecast values")
        st.dataframe(forecast, use_container_width=True, hide_index=True)
        st.download_button("Download Forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                           "stocksense_product_forecast.csv", "text/csv")
        st.caption("Method: baseline moving-average forecast using the latest 30 days of gross daily sales. This is a simple baseline, not a trained machine-learning model.")

elif page == "What-If Analysis":
    st.header("What-If Scenario Analysis")
    st.write("Test how changes in demand or supplier lead time affect the product's stock and replenishment decision.")
    selection = st.selectbox("Choose a product", product_options["Selection"].tolist(), key="whatif_product")
    row, history = get_selected_product(selection)

    st.markdown(f"**Selected product:** {row['Product']}")
    c1, c2 = st.columns(2)
    with c1:
        demand_change = st.slider("Change in daily demand (%)", -50, 100, 20, 5)
    with c2:
        scenario_lead_time = st.slider("Supplier lead time (days)", 1, 30, int(lead_time), 1)

    scenario_demand = max(0.0, float(row["Average Daily Demand"]) * (1 + demand_change / 100))
    scenario_safety = service_factor * float(row["Demand Std Dev"]) * np.sqrt(scenario_lead_time)
    scenario_target = scenario_demand * (scenario_lead_time + review_period) + scenario_safety
    required = max(0.0, scenario_target - float(row["Store Stock"]))
    transfer = min(float(row["Warehouse Stock"]), required)
    supplier_purchase = max(0.0, required - transfer)

    w1, w2, w3 = st.columns(3)
    w1.metric("Scenario daily demand", f"{scenario_demand:.2f} units")
    w2.metric("Warehouse transfer", f"{np.ceil(transfer):,.0f} units")
    w3.metric("Supplier purchase", f"{np.ceil(supplier_purchase):,.0f} units")

    st.subheader("Scenario decision")
    if supplier_purchase > 0:
        st.warning(f"Transfer {np.ceil(transfer):,.0f} units from the warehouse and consider purchasing {np.ceil(supplier_purchase):,.0f} additional units from the supplier.")
    elif transfer > 0:
        st.success(f"Warehouse stock can cover the requirement. Transfer {np.ceil(transfer):,.0f} units; no supplier purchase is indicated by this scenario.")
    else:
        st.success("Current store stock meets the scenario target. No transfer or supplier purchase is indicated.")

    comparison = pd.DataFrame({
        "Measure": ["Daily demand", "Target store stock", "Store stock", "Warehouse stock",
                    "Warehouse transfer", "Supplier purchase"],
        "Current / baseline": [row["Average Daily Demand"], row["Target Stock"], row["Store Stock"],
                               row["Warehouse Stock"], row["Recommended Transfer Qty"], row["Supplier Purchase Qty"]],
        "What-if scenario": [scenario_demand, scenario_target, row["Store Stock"],
                             row["Warehouse Stock"], np.ceil(transfer), np.ceil(supplier_purchase)]
    })
    st.subheader("Baseline vs scenario")
    st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

    st.caption("This is a planning simulation. Lead time and demand changes are user-entered assumptions, not guaranteed future outcomes.")

st.divider()
st.caption("StockSense DSS | MAT5024 Decision Support Systems | Recommendations support the manager's decision; verify stock and supplier details before ordering.")
