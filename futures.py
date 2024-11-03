import pandas as pd

# Specify the file path
file_path = 'SPX_options_1996_2024.parquet'

# To get column names, read the entire Parquet file's metadata
parquet_file = pd.read_parquet(file_path, engine='pyarrow', use_pandas_metadata=True)

# Now you can access the column names
print("Columns in the DataFrame:", parquet_file.columns.tolist())

# To read only the first 100 rows without `nrows`
first_100_rows = pd.read_parquet(file_path, engine='pyarrow').head(100)
print("First 100 rows of the DataFrame:")
print(first_100_rows)

# Counting the number of rows by reading the entire dataframe is not memory-efficient.
# Instead, let's use the PyArrow library directly to count rows more efficiently.
import pyarrow.parquet as pq

parquet_table = pq.read_table(file_path)
row_count = parquet_table.num_rows
print("Total number of rows in the DataFrame:", row_count)


# std error and residuals not the same
# predicted y and estimated y are not the same