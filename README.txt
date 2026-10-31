StockSense – One Supermarket Dataset with Warehouse Quantity

Files:
1. Supermarket_Clean_Inventory_With_Warehouse.csv
2. Supermarket_Daily_Sales_One_Supermarket.csv.gz

Scope:
- One supermarket (Store = Main Supermarket)
- 87 product/SKU records
- Daily sales history for 87 products over the supplied 365-day period
- Inventory includes both Qty_On_Hand and Warehouse_Qty

Quantity definitions:
- Qty_On_Hand: quantity available in the supermarket's store/selling area.
- Warehouse_Qty: additional quantity held in the supermarket's backroom/storage warehouse.
- Total available stock can be calculated as Qty_On_Hand + Warehouse_Qty.

Compatibility:
- Inventory retains the existing StockSense column names and adds Warehouse_Qty.
- Sales retains the existing StockSense sales schema.
- Replace the app's data files with these files after renaming:
  Supermarket_Clean_Inventory_With_Warehouse.csv -> Clean_Inventory.csv
  Supermarket_Daily_Sales_One_Supermarket.csv.gz -> Daily_Sales.csv.gz

Important:
All records, quantities, prices, suppliers, and sales values are synthetic and intended for academic demonstration.
They are not actual sales, stock, or supplier records from any supermarket or brand.
