import pandas as pd
import os
from langchain_core.tools import tool
from typing import Annotated

@tool
def get_stock_data(
    symbol: Annotated[str, "The strategy ID or symbol"],
    start_date: Annotated[str, "Start date"],
    end_date: Annotated[str, "End date"],
) -> str:
    """
    Retrieve baseline parameters for the given strategy.
    """
    try:
        # 强制读取本地文件，无视网络！
        file_path = os.path.join(os.getcwd(), "disney_baseline.csv")
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        return df.to_string()
    except Exception as e:
        return f"Failed to read local baseline data: {e}"
