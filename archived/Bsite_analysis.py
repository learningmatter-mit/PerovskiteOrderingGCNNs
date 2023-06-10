from inference.predict import predict

def get_rocksalt_layered_diff(model_settings):

    df = predict("holdout_set_B_sites",model_settings,"dft_e_ehull")

    comps = set(df["formula"])
    dft = []
    ML = []
    for comp in comps:
        comp_selected = df.loc[df['formula'] == comp]
        ids = list(comp_selected["unrelaxed_cryst_id"])
        layered_dft = comp_selected.loc[comp_selected["unrelaxed_cryst_id"]==min(ids)]["dft_e_hull"]
        rs_dft = comp_selected.loc[comp_selected["unrelaxed_cryst_id"]==max(ids)]["dft_e_hull"]
        layered_ml = comp_selected.loc[comp_selected["unrelaxed_cryst_id"]==min(ids)]["predicted_dft_e_hull"]
        rs_ml = comp_selected.loc[comp_selected["unrelaxed_cryst_id"]==max(ids)]["predicted_dft_e_hull"]
        dft.append(float(layered_dft)-float(rs_dft))
        ML.append(float(layered_ml)-float(rs_ml))

    return dft, ML

    