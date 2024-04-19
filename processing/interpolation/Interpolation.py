import pandas as pd


def update_pure(props):
    df = pd.read_json("data/edge_dataset.json")
    for prop in props:       
        pure_dict_init = {}
        for _, row in df.iterrows():
            if len(row["composition"]["sites"]["A"]) == 1 and len(row["composition"]["sites"]["B"]) == 1:
                A_site = row["composition"]["sites"]["A"][0]
                B_site = row["composition"]["sites"]["B"][0]
                curr_perov = (A_site, B_site)
                if curr_perov in pure_dict_init:
                    if row["dft_energy_per_atom"] < pure_dict_init[curr_perov][1]:
                        pure_dict_init[curr_perov] = [row[prop], row["dft_energy_per_atom"]]
                else:
                    pure_dict_init[curr_perov] = [row[prop], row["dft_energy_per_atom"]]
            else:
                raise Warning("A non-binary composition exist in edge_dataset")

        Pure_ref = {}
        for key in pure_dict_init:
            Pure_ref[key[0] + '.' + key[1]] = pure_dict_init[key][0]

    return Pure_ref


def filter_alloys(df):
    filt = []
    for index, row in df.iterrows():
        if len(row["composition"]["sites"]["A"]) > 1 or len(row["composition"]["sites"]["B"]) > 1:
            filt.append(index)
        else:
             raise Warning("A binary composition exists in the examined dataset")
    if len(filt) == 0:
        raise ValueError("All binary compositions")
    filt_structs = df.loc[filt]

    return filt_structs


def interpolator(df, prop, Pure_ref):
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
#            raise Warning("Cannot find all binary edges for an composition")

    return Interpolation, Difference


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
    return df_out


def apply_interpolation(df, prop):
    Pure_ref = update_pure([prop])
    df_interp = filter_alloys(df)
    interp, diff = interpolator(df_interp, prop, Pure_ref)
    df_interp = filter_interpolation(df_interp, prop, interp, diff)
    prop_diff_name = prop + "_diff"
    df_interp = df_interp.dropna(subset=[prop_diff_name], how='all')
    return df_interp
