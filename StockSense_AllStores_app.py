
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="StockSense DSS",
    page_icon="📦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
INVENTORY_FILE = BASE_DIR / "Clean_Inventory.csv"
SALES_FILE = BASE_DIR / "Daily_Sales.csv.gz"

st.title("📦 StockSense")
st.subheader("An Intelligent Inventory Replenishment Decision Support System")
st.caption("Retail inventory and sales analysis across all stores and products.")

# 2. LOAD DATA

@st.cache_data(show_spinner="Loading inventory and sales data...")
def load_data(inventory_file, sales_file):

    inventory = pd.read_csv(inventory_file, low_memory=False)
    sales = pd.read_csv(sales_file, low_memory=False)

    required_inventory = [
        "Product_No",
        "Store",
        "Qty_On_Hand",
        "Stock_Unit_Cost_Price",
        "Inventory_End_Date",
        "Inventory_Start_Date"
    ]

    required_sales = [
        "Product_No",
        "Store",
        "Transaction_Date",
        "Gross_Units_Sold"
    ]

    missing_inventory = [
        c for c in required_inventory if c not in inventory.columns
    ]

    missing_sales = [
        c for c in required_sales if c not in sales.columns
    ]

    if missing_inventory:
        raise ValueError(
            f"Missing inventory columns: {missing_inventory}"
        )

    if missing_sales:
        raise ValueError(
            f"Missing sales columns: {missing_sales}"
        )

    inventory["Product_No"] = (
        inventory["Product_No"].fillna("Unknown").astype(str)
    )

    inventory["Store"] = (
        inventory["Store"].fillna("Unknown").astype(str)
    )

    sales["Product_No"] = (
        sales["Product_No"].fillna("Unknown").astype(str)
    )

    sales["Store"] = (
        sales["Store"].fillna("Unknown").astype(str)
    )

    inventory["Inventory_End_Date"] = pd.to_datetime(
        inventory["Inventory_End_Date"], errors="coerce"
    )

    inventory["Inventory_Start_Date"] = pd.to_datetime(
        inventory["Inventory_Start_Date"], errors="coerce"
    )

    sales["Transaction_Date"] = pd.to_datetime(
        sales["Transaction_Date"], errors="coerce"
    )

    inventory["Qty_On_Hand"] = pd.to_numeric(
        inventory["Qty_On_Hand"], errors="coerce"
    )

    inventory["Stock_Unit_Cost_Price"] = pd.to_numeric(
        inventory["Stock_Unit_Cost_Price"], errors="coerce"
    )

    sales["Gross_Units_Sold"] = pd.to_numeric(
        sales["Gross_Units_Sold"], errors="coerce"
    ).fillna(0)

    sales = sales.dropna(subset=["Transaction_Date"])

    return inventory, sales


# 3. CHECK FILES AND LOAD DATA

if not INVENTORY_FILE.exists():
    st.error("Clean_Inventory.csv was not found.")
    st.stop()

if not SALES_FILE.exists():
    st.error("Daily_Sales.csv.gz was not found.")
    st.stop()

try:
    inventory_raw, sales_raw = load_data(
        str(INVENTORY_FILE),
        str(SALES_FILE)
    )

except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.stop()


# 4. INVENTORY DATA CLEANING

if "Stock Data Flag" in inventory_raw.columns:
    inventory_raw["Stock Data Flag"] = (
        inventory_raw["Stock Data Flag"].fillna("OK").astype(str)
    )
else:
    inventory_raw["Stock Data Flag"] = "OK"

valid_inv = inventory_raw[
    inventory_raw["Qty_On_Hand"].notna()
    & (inventory_raw["Qty_On_Hand"] >= 0)
    & (inventory_raw["Stock Data Flag"] == "OK")
].copy()

if valid_inv.empty:
    st.error("No valid inventory records are available.")
    st.stop()


# 5. SIDEBAR FILTERS

st.sidebar.header("Data Filters")

store_options = [
    "All Stores"
] + sorted(valid_inv["Store"].unique().tolist())

selected_store = st.sidebar.selectbox(
    "Store coverage",
    store_options
)

if selected_store == "All Stores":
    store_inv = valid_inv.copy()
    store_sales = sales_raw.copy()
else:
    store_inv = valid_inv[
        valid_inv["Store"] == selected_store
    ].copy()

    store_sales = sales_raw[
        sales_raw["Store"] == selected_store
    ].copy()

st.sidebar.subheader("Replenishment Assumptions")

service_factor = st.sidebar.selectbox(
    "Service level factor (Z)",
    [1.28, 1.645, 1.96, 2.33],
    index=1
)

review_period = st.sidebar.number_input(
    "Inventory review period (days)",
    min_value=1,
    max_value=60,
    value=7
)

lead_time = st.sidebar.number_input(
    "Assumed supplier lead time (days)",
    min_value=1,
    max_value=90,
    value=7
)

ordering_cost = st.sidebar.number_input(
    "Assumed ordering cost per order (₹)",
    min_value=1.0,
    value=200.0,
    step=50.0
)

annual_holding_rate = st.sidebar.number_input(
    "Assumed annual holding cost (% of unit cost)",
    min_value=1.0,
    max_value=100.0,
    value=20.0
)

st.sidebar.caption(
    "Lead time, ordering cost and holding rate are editable assumptions."
)


# 6. OPTIMIZED DAILY SALES AGGREGATION

@st.cache_data(show_spinner="Preparing daily sales summaries...")
def prepare_daily_sales(sales):

    daily = (
        sales.groupby(
            ["Product_No", "Store", "Transaction_Date"],
            as_index=False,
            observed=True
        )["Gross_Units_Sold"]
        .sum()
    )

    return daily


sales_daily = prepare_daily_sales(store_sales)

if sales_daily.empty:
    st.warning("No sales records are available for this selection.")
    st.stop()


# 7. LATEST INVENTORY RECORDS

latest_inventory = (
    store_inv.sort_values(
        [
            "Product_No",
            "Store",
            "Inventory_End_Date",
            "Inventory_Start_Date"
        ]
    )
    .drop_duplicates(
        ["Product_No", "Store"],
        keep="last"
    )
    .copy()
)

if latest_inventory.empty:
    st.error("No latest inventory records are available.")
    st.stop()


# 8. VECTORIZED DEMAND CALCULATION

@st.cache_data(show_spinner="Calculating demand and replenishment...")
def calculate_demand(latest_inv, daily_sales, z_factor, lead_days,
                     review_days, order_cost, holding_rate):

    keys = ["Product_No", "Store"]

    # Find first and latest sales dates for every product-store pair
    bounds = (
        daily_sales.groupby(keys, as_index=False)
        .agg(
            First_Date=("Transaction_Date", "min"),
            Last_Date=("Transaction_Date", "max")
        )
    )

    # Use the latest 30 calendar days, or the available history
    bounds["Window_Start"] = (
        bounds["Last_Date"] - pd.Timedelta(days=29)
    )

    bounds["Window_Start"] = bounds[
        ["Window_Start", "First_Date"]
    ].max(axis=1)

    bounds["Number_of_Days"] = (
        (bounds["Last_Date"] - bounds["Window_Start"]).dt.days + 1
    )

    # Attach date limits to the daily sales records
    recent_sales = daily_sales.merge(
        bounds[keys + ["Window_Start", "Last_Date"]],
        on=keys,
        how="inner"
    )

    recent_sales = recent_sales[
        (recent_sales["Transaction_Date"] >= recent_sales["Window_Start"])
        & (recent_sales["Transaction_Date"] <= recent_sales["Last_Date"])
    ].copy()

    # Calculate sales totals and squared sales values
    recent_sales["Squared_Units"] = (
        recent_sales["Gross_Units_Sold"] ** 2
    )

    demand_stats = (
        recent_sales.groupby(keys, as_index=False)
        .agg(
            Sales_Sum=("Gross_Units_Sold", "sum"),
            Sales_Squared_Sum=("Squared_Units", "sum")
        )
    )

    demand_stats = demand_stats.merge(
        bounds[keys + ["Number_of_Days"]],
        on=keys,
        how="left"
    )

    n = demand_stats["Number_of_Days"].clip(lower=1)

    demand_stats["Average Daily Demand"] = (
        demand_stats["Sales_Sum"] / n
    )

    # Sample standard deviation, including zero-sales days
    numerator = (
        demand_stats["Sales_Squared_Sum"]
        - (demand_stats["Sales_Sum"] ** 2 / n)
    ).clip(lower=0)

    denominator = (n - 1).clip(lower=1)

    variance = numerator / denominator

    demand_stats["Demand Std Dev"] = np.where(
        n > 1,
        np.sqrt(variance),
        0.0
    )

    # Merge demand calculations with latest inventory
    result = latest_inv.merge(
        demand_stats[
            keys + [
                "Average Daily Demand",
                "Demand Std Dev"
            ]
        ],
        on=keys,
        how="left"
    )

    result["Average Daily Demand"] = (
        result["Average Daily Demand"].fillna(0)
    )

    result["Demand Std Dev"] = (
        result["Demand Std Dev"].fillna(0)
    )

    # Replenishment calculations
    stock = result["Qty_On_Hand"].fillna(0).clip(lower=0)

    unit_cost = (
        result["Stock_Unit_Cost_Price"].fillna(0).clip(lower=0)
    )

    avg_demand = result["Average Daily Demand"]
    demand_std = result["Demand Std Dev"]

    safety_stock = (
        z_factor * demand_std * np.sqrt(lead_days)
    )

    reorder_point = (
        avg_demand * lead_days + safety_stock
    )

    annual_demand = avg_demand * 365

    holding_cost = unit_cost * holding_rate / 100

    eoq_numerator = (
        2 * annual_demand * order_cost
    )

    eoq = np.sqrt(
        np.divide(
            eoq_numerator,
            holding_cost,
            out=np.zeros(len(result), dtype=float),
            where=holding_cost.to_numpy() > 0
        )
    )

    target_stock = (
        avg_demand * (lead_days + review_days)
        + safety_stock
    )

    recommended_qty = np.ceil(
        (target_stock - stock).clip(lower=0)
    ).astype(int)

    status = np.select(
        [
            stock <= reorder_point,
            stock < target_stock
        ],
        [
            "Reorder Required",
            "Low Stock"
        ],
        default="Sufficient Stock"
    )

    result["Current Stock"] = stock
    result["Unit Cost (₹)"] = unit_cost
    result["Safety Stock"] = safety_stock
    result["Reorder Point"] = reorder_point
    result["EOQ"] = eoq
    result["Target Stock"] = target_stock
    result["Recommended Order Qty"] = recommended_qty

    result["Estimated Order Value (₹)"] = (
        recommended_qty * unit_cost
    )

    result["Status"] = status

    result["Product"] = (
        result["Product_Description"]
        if "Product_Description" in result.columns
        else result["Product_No"]
    )

    result["Supplier"] = (
        result["Supplier"]
        if "Supplier" in result.columns
        else "Not available"
    )

    return result


try:
    analysis = calculate_demand(
        latest_inventory,
        sales_daily,
        service_factor,
        lead_time,
        review_period,
        ordering_cost,
        annual_holding_rate
    )

except Exception as exc:
    st.error(f"Demand calculation error: {exc}")
    st.stop()


if analysis.empty:
    st.warning("No product-store combinations are available.")
    st.stop()


# 9. INVENTORY OVERVIEW

st.header("1. Complete Inventory Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Stores included",
    f"{analysis['Store'].nunique():,}"
)

c2.metric(
    "Product-store records",
    f"{len(analysis):,}"
)

c3.metric(
    "Total stock units",
    f"{analysis['Current Stock'].sum():,.0f}"
)

c4.metric(
    "Reorder required",
    f"{(analysis['Status'] == 'Reorder Required').sum():,}"
)

c5.metric(
    "Estimated order value",
    f"₹{analysis['Estimated Order Value (₹)'].sum():,.2f}"
)

flagged_count = (
    inventory_raw["Stock Data Flag"] != "OK"
).sum()

st.caption(
    f"Coverage: {selected_store} | "
    f"Inventory records flagged for review: {flagged_count:,}"
)


# 10. INVENTORY ANALYSIS TABLE

st.divider()
st.header("2. Inventory Analysis")

status_filter = st.multiselect(
    "Filter stock status",
    [
        "Reorder Required",
        "Low Stock",
        "Sufficient Stock"
    ],
    default=[
        "Reorder Required",
        "Low Stock",
        "Sufficient Stock"
    ]
)

display_analysis = analysis[
    analysis["Status"].isin(status_filter)
].copy()

st.caption(
    f"Showing up to 5,000 of {len(display_analysis):,} matching records. "
    "Download the full analysis below."
)

st.dataframe(
    display_analysis.head(5000).round(2),
    use_container_width=True,
    hide_index=True,
    height=420
)


# 11. STOCK STATUS CHART

st.subheader("Stock Status Distribution")

status_counts = (
    analysis["Status"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Product-store records")
)

fig_status = px.bar(
    status_counts,
    x="Status",
    y="Product-store records",
    color="Status",
    title="Inventory Status Across Selected Stores",
    text="Product-store records"
)

fig_status.update_layout(
    xaxis_title="Stock Status",
    yaxis_title="Number of Records",
    showlegend=False
)

st.plotly_chart(
    fig_status,
    use_container_width=True
)


# 12. TOP REPLENISHMENT CHART

st.subheader("Highest Recommended Order Quantities")

top_orders = (
    analysis[analysis["Recommended Order Qty"] > 0]
    .nlargest(20, "Recommended Order Qty")
)

if not top_orders.empty:

    fig_orders = px.bar(
        top_orders,
        x="Product_No",
        y="Recommended Order Qty",
        color="Store",
        hover_data=["Product", "Status"],
        title="Top 20 Replenishment Recommendations"
    )

    fig_orders.update_layout(
        xaxis_title="Product Number",
        yaxis_title="Recommended Order Quantity"
    )

    st.plotly_chart(
        fig_orders,
        use_container_width=True
    )

else:
    st.info("No products require replenishment under current assumptions.")


# 13. PRODUCT AND STORE SELECTION FOR FORECASTING

st.header("3. Demand Forecasting")

forecast_stores = sorted(
    analysis["Store"].unique().tolist()
)

if selected_store == "All Stores":

    forecast_store = st.selectbox(
        "Select store for forecasting",
        forecast_stores
    )

else:
    forecast_store = selected_store

forecast_products = analysis[
    analysis["Store"] == forecast_store
].copy()

product_options = (
    forecast_products[
        ["Product_No", "Product"]
    ]
    .drop_duplicates("Product_No")
    .sort_values("Product_No")
)

product_labels = {
    row["Product_No"]: (
        f"{row['Product_No']} - {str(row['Product'])[:70]}"
    )
    for _, row in product_options.iterrows()
}

selected_product = st.selectbox(
    "Select product",
    product_options["Product_No"].tolist(),
    format_func=lambda x: product_labels.get(x, x)
)


# 14. SELECTED PRODUCT HISTORY AND FORECAST

@st.cache_data(show_spinner=False)
def get_product_history(daily_sales, product_no, store):

    selected = daily_sales[
        (daily_sales["Product_No"] == product_no)
        & (daily_sales["Store"] == store)
    ].copy()

    if selected.empty:
        return pd.DataFrame(
            columns=["Date", "Units Sold"]
        )

    selected = (
        selected.groupby("Transaction_Date")["Gross_Units_Sold"]
        .sum()
        .sort_index()
    )

    selected.index = pd.DatetimeIndex(selected.index)

    full_history = selected.asfreq(
        "D",
        fill_value=0
    )

    return pd.DataFrame({
        "Date": full_history.index,
        "Units Sold": full_history.values
    })


history = get_product_history(
    sales_daily,
    selected_product,
    forecast_store
)

forecast_horizon = st.slider(
    "Forecast horizon (days)",
    min_value=1,
    max_value=30,
    value=7
)

if history.empty:

    st.info("No sales history is available for this product.")

else:

    lookback = min(30, len(history))

    forecast_mean = float(
        history.tail(lookback)["Units Sold"].mean()
    )

    future_dates = pd.date_range(
        history["Date"].max() + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq="D"
    )

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Demand": [
            round(forecast_mean, 2)
        ] * forecast_horizon
    })

    fig_forecast = px.line(
        history,
        x="Date",
        y="Units Sold",
        title=f"Historical Demand — {selected_product}"
    )

    fig_forecast.add_scatter(
        x=forecast["Date"],
        y=forecast["Forecasted Demand"],
        mode="lines+markers",
        name="Moving Average Forecast"
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    st.metric(
        "Average Forecasted Daily Demand",
        f"{forecast_mean:.2f} units"
    )

    st.dataframe(
        forecast,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Forecast method: average gross units sold over the latest "
        "30 available calendar days. This is a baseline moving-average "
        "forecast, not a trained machine-learning model."
    )


# 15. REPLENISHMENT RECOMMENDATIONS

st.header("4. Replenishment Recommendations")

recommendations = analysis[
    analysis["Recommended Order Qty"] > 0
].copy()

if recommendations.empty:

    st.success(
        "No replenishment is recommended under current assumptions."
    )

else:

    st.warning(
        f"{len(recommendations):,} product-store combinations "
        "need stock attention."
    )

    recommendations = recommendations.sort_values(
        "Estimated Order Value (₹)",
        ascending=False
    )

    st.dataframe(
        recommendations.head(5000).round(2),
        use_container_width=True,
        hide_index=True,
        height=420
    )

    st.caption(
        "The table displays up to 5,000 recommendations. "
        "Download the full report in Section 7."
    )


# 16. WHAT-IF SCENARIO ANALYSIS

st.header("5. What-if Scenario Analysis")

scenario_store = st.selectbox(
    "Choose scenario store",
    forecast_stores,
    key="scenario_store"
)

scenario_products = analysis[
    analysis["Store"] == scenario_store
][["Product_No", "Product"]].drop_duplicates()

scenario_labels = {
    row["Product_No"]: (
        f"{row['Product_No']} - {str(row['Product'])[:70]}"
    )
    for _, row in scenario_products.iterrows()
}

scenario_product = st.selectbox(
    "Choose scenario product",
    scenario_products["Product_No"].tolist(),
    format_func=lambda x: scenario_labels.get(x, x),
    key="scenario_product"
)

scenario_row = analysis[
    (analysis["Product_No"] == scenario_product)
    & (analysis["Store"] == scenario_store)
].iloc[0]

demand_change = st.slider(
    "Assumed change in daily demand (%)",
    min_value=-50,
    max_value=100,
    value=20,
    step=5
)

scenario_demand = max(
    0.0,
    scenario_row["Average Daily Demand"]
    * (1 + demand_change / 100)
)

scenario_safety = (
    service_factor
    * scenario_row["Demand Std Dev"]
    * np.sqrt(lead_time)
)

scenario_rop = (
    scenario_demand * lead_time + scenario_safety
)

scenario_target = (
    scenario_demand * (lead_time + review_period)
    + scenario_safety
)

scenario_qty = int(
    np.ceil(
        max(
            0,
            scenario_target - scenario_row["Current Stock"]
        )
    )
)

s1, s2, s3 = st.columns(3)

s1.metric(
    "Scenario daily demand",
    f"{scenario_demand:.2f}"
)

s2.metric(
    "Scenario reorder point",
    f"{scenario_rop:.2f}"
)

s3.metric(
    "Scenario order quantity",
    f"{scenario_qty:,} units"
)


# 17. DATA QUALITY NOTES

st.header("6. Data Quality Notes")

flagged = inventory_raw[
    inventory_raw["Stock Data Flag"] != "OK"
]

st.write(
    f"The source inventory contains **{len(flagged):,} records** "
    "flagged for review, including negative or missing stock quantities. "
    "These records are excluded from replenishment calculations."
)

st.write(
    "Supplier lead time, ordering cost and annual holding-cost rate "
    "are not provided in the source data. The sidebar values are "
    "editable assumptions, not actual supplier records."
)


# 18. DOWNLOAD REPORTS

st.header("7. Download Reports")

download_columns = [
    "Store",
    "Product_No",
    "Product",
    "Current Stock",
    "Average Daily Demand",
    "Demand Std Dev",
    "Safety Stock",
    "Reorder Point",
    "EOQ",
    "Target Stock",
    "Recommended Order Qty",
    "Unit Cost (₹)",
    "Estimated Order Value (₹)",
    "Status",
    "Supplier"
]

complete_report = analysis[download_columns].copy()

st.download_button(
    "Download Complete Inventory Analysis (CSV)",
    data=complete_report.to_csv(index=False).encode("utf-8"),
    file_name="stocksense_all_store_inventory_analysis.csv",
    mime="text/csv"
)

if not history.empty:

    st.download_button(
        "Download Selected Product Forecast (CSV)",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="stocksense_selected_product_forecast.csv",
        mime="text/csv"
    )


# 19. FOOTER

st.divider()

st.caption(
    "StockSense DSS | MAT5024 Decision Support Systems | "
    "Decision-support prototype. Verify recommendations "
    "before placing supplier orders."
)
