import time
from decimal import Decimal


# ################################################
# my own methodes 
# ################################################# 

def parse_bool(value: str, default=True) -> bool:
    if value is None:
        return default
    return value.strip().lower() == "true"

def safe_decimal(value, fallback="0"):
    return Decimal(str(value or fallback))

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
    
def has_not_empty_column(df, column_list) -> bool:
    for column_name in column_list:
        is_present = column_name in df and not df[column_name].empty
        if not is_present:
            return False
    return True

def has_true_then_false_sequence(sequence: list[bool]) -> bool:
    found_true = False
    for val in sequence:
        if found_true and val is False:
            return True
        if val is True:
            found_true = True
    return False
