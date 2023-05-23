import pandas as pd
from pymatgen.core import Structure

def filter_data_by_properties(df,props):
    if isinstance(props,str):
        props = [props]
    return df.dropna(subset=props)


def select_structures(df,structure_type):
    if structure_type == "unrelaxed":
        df["structure"] = Structure.from_dict(df['unrelaxed_struct'])
    else:
        df["structure"] = Structure.from_dict(df["opt_struct"])
    return df