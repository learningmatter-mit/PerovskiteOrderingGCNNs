import os
import pandas as pd
import json

def update_pure(props):
    df = pd.read_json("data/edge_dataset.json")
    for prop in props:       
        pure_dict_init = {}
        for index, row in df.iterrows():
            if len(row["composition"]["sites"]["A"]) == 1 and len(row["composition"]["sites"]["B"]) == 1:
                A_site = row["composition"]["sites"]["A"][0]
                B_site = row["composition"]["sites"]["B"][0]
                curr_perov = (A_site, B_site)
                if curr_perov in pure_dict_init:
                    if row["dft_energy_per_atom"] < pure_dict_init[curr_perov][1]:
                        pure_dict_init[curr_perov] = [row[prop], row["dft_energy_per_atom"]]
                else:
                    pure_dict_init[curr_perov] = [row[prop], row["dft_energy_per_atom"]]

        pure_dict = {}

        for key in pure_dict_init:
            pure_dict[key[0] + '.' + key[1]] = pure_dict_init[key][0]

        WORKDIR = os.getcwd()
        if not os.path.exists(WORKDIR + '/processing/interpolation/Pure_ref'):
            os.makedirs(WORKDIR + '/processing/interpolation/Pure_ref')

        with open(WORKDIR + "/processing/interpolation/Pure_ref/" + prop + ".json", "w") as outfile:
            json.dump(pure_dict, outfile)

def filter_alloys(df):
    filt = []
    for index, row in df.iterrows():
        if len(row["composition"]["sites"]["A"]) > 1 or len(row["composition"]["sites"]["B"]) > 1:
            filt.append(index)
    if len(filt) == 0:
        print("All Pure")
        return None
    filt_structs = df.loc[filt]
    return filt_structs

def filter_interpolation(df, prop, Interpolation, Difference):    
    interp_out = []
    diff_out = []
    indexes = []
    for i in range(len(Interpolation)):
        interp_out.append(Interpolation[i])
        diff_out.append(Difference[i])
        indexes.append(i)  
    df_out = df.iloc[indexes].copy()
    df_out[prop + '_interp'] = interp_out
    df_out[prop + '_diff'] = diff_out
    return df_out, interp_out, diff_out

def interpolator(df, prop):
    WORKDIR = os.getcwd()
    with open(WORKDIR + "/processing/interpolation/Pure_ref/" + prop + ".json") as json_file:
        Pure_ref = json.load(json_file)

    Interpolation = []
    Difference = []

    for index, row in df.iterrows():
        A_sites = row["composition"]["sites"]["A"]
        B_sites = row["composition"]["sites"]["B"]
        bounds = {}
        found_bounds = True
        
        for A in A_sites:
            for B in B_sites:
                if A + '.' + B in Pure_ref:
                    bounds[A + '.' + B] = []
                    compA = float(row["composition"]["composition"][A])
                    compB = float(row["composition"]["composition"][B])
                    curr_comp = compA * compB
                    bounds[A + '.' + B].append(curr_comp)
                    bounds[A + '.' + B].append(Pure_ref[A + '.' + B])
                else:
                    found_bounds = False
        total_comp = 0
        for bound in bounds:
            total_comp += bounds[bound][0]
        if total_comp != 1.0:
            for bound in bounds:
                bounds[bound][0] /= total_comp
        if (found_bounds):
            curr_interpolation = 0
            for bound in bounds:
                curr_interpolation += bounds[bound][0] * bounds[bound][1]
        else:
            curr_interpolation = None

        Interpolation.append(curr_interpolation)
        
        if curr_interpolation:
            Difference.append(row[prop] - curr_interpolation)
        else:
            Difference.append(None)

    return Interpolation, Difference

def apply_interpolation(df, prop):
    update_pure([prop])
    df_interp = filter_alloys(df)
    interp, diff = interpolator(df_interp, prop)
    df_interp, interp_filter, diff_filter = filter_interpolation(df_interp, prop, interp, diff)
    prop_diff_name = prop + "_diff"
    prop_diff_names = [prop_diff_name]
    df_interp = df_interp.dropna(subset=[prop_diff_name], how='all')
    return df_interp