import copy
from pymatgen.core import Structure


def filter_data_by_properties(df,props):
    if isinstance(props,str):
        props = [props]
    return df.dropna(subset=props)


def select_structures(df,structure_type):
    df_copy = copy.deepcopy(df)

    if structure_type == "unrelaxed":
        df_copy = df_copy.dropna(subset=["unrelaxed_struct"])
        df_copy["structure"] = df_copy.apply(lambda x: Structure.from_dict(x["unrelaxed_struct"]), axis=1)
    elif structure_type == "relaxed":
        df_copy = df_copy.dropna(subset=["opt_struct"])
        df_copy["structure"] = df_copy.apply(lambda x: Structure.from_dict(x["opt_struct"]), axis=1)
    elif structure_type == "spud":
        df_copy = df_copy.dropna(subset=["spud_struct"])
        df_copy["structure"] = df_copy.apply(lambda x: Structure.from_dict(x["spud_struct"]), axis=1)
    elif structure_type == "M3Gnet_relaxed":
        df_copy = df_copy.dropna(subset=["M3Gnet_relaxed_struct"])
        df_copy["structure"] = df_copy.apply(lambda x: Structure.from_dict(x["M3Gnet_relaxed_struct"]), axis=1)
    else:
        raise ValueError("structure_type must be 'unrelaxed' or 'relaxed'")
    return df_copy
