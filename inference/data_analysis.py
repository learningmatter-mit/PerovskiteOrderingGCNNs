import re
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from training.sigopt_utils import build_sigopt_name


plt.rcParams["figure.figsize"] = (13, 8)
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["xtick.major.width"] = 2
plt.rcParams['text.usetex'] = False
plt.rc('lines', linewidth=3, color='g')
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def plot_hex(target_prop, true_values, pred_values, test_set_type, experimental_setting, sigopt_name, exp_id, pure_interp=False, additional_string='all', mae_mean=None, mae_std=None):
    if target_prop == "dft_e_hull":
        if test_set_type == "test_set":
            hex_xylim = [-0.1, 0.5]
            vmax = 15
        elif test_set_type == "holdout_set_B_sites":
            if additional_string == 'all':
                hex_xylim = [-0.1, 0.4]
                vmax = 15
            # elif additional_string == 'layered_rocksalt':
            #     hex_xylim = [-0.15, 0.3]
            #     vmax = 15
            elif additional_string == 'energies_vs_groundstate':
                hex_xylim = [-0.02, 0.18]
                vmax = 15
        elif test_set_type == "holdout_set_series":
            if additional_string == 'all':
                hex_xylim = [0, 0.4]
                vmax = 15
            elif additional_string == 'energies_vs_groundstate':
                hex_xylim = [-0.02, 0.12]
                vmax = 15                    
    elif target_prop == "Op_band_center":
        if test_set_type == "test_set":
            hex_xylim = [-7, 0]
            vmax = 15
        elif test_set_type == "holdout_set_B_sites":
            if additional_string == 'all':
                hex_xylim = [-4.5, -0.5]
                vmax = 15
            elif additional_string == 'energies_vs_groundstate':
                hex_xylim = [-0.2, 2.2]
                vmax = 15
        elif test_set_type == "holdout_set_series":
            hex_xylim = [-4.5, -0.5]
            vmax = 15

    # hex_figsize = (4, 3.2)
    hex_figsize = (3, 2.2)
            
    fig, ax = plt.subplots(figsize=hex_figsize)
    
    if target_prop == "dft_e_hull":
        # ax.set_xlabel("DFT $E_{\mathrm{hull}}$ (eV/atom)")
        # ax.set_ylabel("ML $E_{\mathrm{hull}}$ (eV/atom)")
        ax.set_xlabel("DFT")
        ax.set_ylabel("ML")
        # ax.set_xticks([0.05, 0.15])
        # ax.set_yticks([0.05, 0.15])
        plt.xticks(size=12)
        plt.yticks(size=12)
    else:
        # ax.set_xlabel("DFT O 2p band center (eV)")
        # ax.set_ylabel("ML O 2p band center (eV)")
        ax.set_xlabel("DFT")
        ax.set_ylabel("ML")
        plt.xticks(size=12)
        plt.yticks(size=12)
    
    ax.axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), color='black', linestyle='--', linewidth=1)

    hb = ax.hexbin(
        true_values, pred_values,
        cmap='inferno_r', gridsize=30, bins=None, mincnt=1, edgecolors='none',
        extent=[hex_xylim[0], hex_xylim[1], hex_xylim[0], hex_xylim[1]],
        vmin=0, vmax=vmax,
        )

    r, _ = pearsonr(true_values, pred_values)
    if pure_interp:
        mae = mean_absolute_error(true_values, pred_values)
        ax.annotate("MAE = %.3f" % (mae), xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', size=12)
    else:
        ax.annotate("MAE = %.3f\nr = %.2f" % (mae_mean, r), xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', size=10)
        # ax.annotate("$r$ = %.2f" % (r), xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', size=14)

    cb = fig.colorbar(hb)
    cb.set_label('Count', size=12)
    cb.ax.set_yticklabels([0, 10], fontsize=12);
    plt.tight_layout()
    
    if pure_interp:
        plt.savefig("figures/" + test_set_type + "_pure_interpolation_hexbin_" + str(additional_string) + ".pdf")
        print("Completed pure interpolation")
    else:
        plt.savefig("figures/" + test_set_type + "_" + sigopt_name + "_" + exp_id +  "_hexbin_" + str(additional_string) + ".pdf")
        print("Completed " + sigopt_name + " " + str(exp_id))

    plt.close()


def plot_hex_all(target_prop, test_set_types, experimental_settings, num_best_models=3):
    for experimental_setting in experimental_settings:
        sigopt_name = build_sigopt_name("data/", target_prop, experimental_setting["struct_type"], experimental_setting["interpolation"], experimental_setting["model_type"])
        directory = "./best_models/" + experimental_setting["model_type"] + "/" + sigopt_name + "/" +str(experimental_setting["exp_id"])
        test_set_dfs = {}

        for test_set_type in test_set_types:
            test_set_dfs[test_set_type] = []
            for i in range(num_best_models):
                with open(directory + "/" + "best_" + str(i) + "/" + test_set_type + "_predictions.json") as f:
                    test_set_dfs[test_set_type].append(pd.read_json(f))

            true_values = test_set_dfs[test_set_type][0][target_prop].to_numpy()

            temp_tuple = ()
            maes = []
            for i in range(num_best_models):
                predicted_values = test_set_dfs[test_set_type][i]['predicted_' + target_prop].to_numpy()
                temp_tuple += (predicted_values,)
                maes.append(mean_absolute_error(true_values, predicted_values))
            pred_values_mean = np.mean(np.vstack(temp_tuple), axis=0)
            mae_mean = np.mean(maes)
            mae_std = np.std(maes)

            plot_hex(target_prop, true_values, pred_values_mean, test_set_type, experimental_setting, sigopt_name, str(experimental_setting["exp_id"]), mae_mean=mae_mean, mae_std=mae_std)

            # if experimental_setting == experimental_settings[-1]:
            #     pred_values = test_set_dfs[test_set_type][0][target_prop + '_interp'].to_numpy()
            #     plot_hex(target_prop, true_values, pred_values, test_set_type, experimental_setting, None, None, pure_interp=True)


def plot_violin_filter_comps(target_prop, test_set_dfs, series, column_conc, column_entry, i, j, get_true_values=False):
    temp_df = test_set_dfs[i][
            test_set_dfs[i].formula.str.contains(series[j][0][0]) &
            test_set_dfs[i].formula.str.contains(series[j][0][1]) &
            test_set_dfs[i].formula.str.contains(series[j][1][0]) &
            test_set_dfs[i].formula.str.contains(series[j][1][1])
        ]
    to_plot = pd.DataFrame(columns=[column_conc, column_entry])
    k = 0

    if get_true_values:
        prop = target_prop
    else:
        prop = 'predicted_' + target_prop

    for framework, subdf in temp_df.groupby('framework'):
        conc = float(re.findall(r'%s(0\.\d+)' % series[j][0][0], framework)[0])
        for entry in subdf[prop]:
            to_plot.loc[k] = [conc, entry]
            k += 1

    return to_plot, temp_df


def plot_violin(target_prop, experimental_setting, series, num_best_models, test_set_dfs, test_set_type, sigopt_name, j, get_true_values=False):
    column_conc = series[j][0][0] + " on A site"
    additional_string = series[j][0][0] + "$_x$" + series[j][0][1] + "$_{1-x}$" + series[j][1][0] + "$_{0.5}$" + series[j][1][1] + "$_{0.5}$O$_3$"

    if target_prop == "dft_e_hull":
        ylim = [0, 0.3]
        column_entry = "$E_{\mathrm{hull}}$ (eV/atom)"
    elif target_prop == "Op_band_center":
        ylim = [-4, -1]
        column_entry = "O 2p band center (eV)"
    
    sns.set(rc={'figure.figsize':(4, 2)})

    if get_true_values:
        to_plot_final, temp_df = plot_violin_filter_comps(target_prop, test_set_dfs, series, column_conc, column_entry, 0, j, get_true_values=True)            
        
        true_values = temp_df[target_prop].to_numpy()
        # pred_values = temp_df[target_prop + '_interp'].to_numpy()
        # plot_hex(target_prop, true_values, pred_values, test_set_type, experimental_setting, None, None, pure_interp=True, additional_string=additional_string)

    else:
        to_plots = []
        temp_tuple = ()
        
        maes = []       
        for i in range(num_best_models):
            to_plot, temp_df = plot_violin_filter_comps(target_prop, test_set_dfs, series, column_conc, column_entry, i, j)
            to_plots.append(to_plot)
            predicted_values = temp_df['predicted_' + target_prop].to_numpy()
            temp_tuple += (predicted_values,)

            if i == 0:
                true_values = temp_df[target_prop].to_numpy()

            maes.append(mean_absolute_error(true_values, predicted_values))            
        
        mae_mean = np.mean(maes)
        mae_std = np.std(maes)
        
        to_plot_final = pd.DataFrame()
        to_plot_final[column_conc] = to_plots[0][column_conc]
        
        pred_values_mean = np.mean(np.vstack(temp_tuple), axis=0) 
        # plot_hex(target_prop, true_values, pred_values_mean, test_set_type, experimental_setting, sigopt_name, str(experimental_setting["exp_id"]), additional_string=additional_string, mae_mean=mae_mean, mae_std=mae_std)

        temp_tuple = ()
        for i in range(num_best_models):
            temp_tuple += (to_plots[i][[column_entry]],)
        to_plot_final[column_entry] = pd.concat(temp_tuple, axis=1).mean(axis=1)        

    ax = sns.violinplot(x=column_conc, y=column_entry, data=to_plot_final, inner="points", scale='width', cut=1)
    ax.set_ylim(ylim)
    # ax.set_title(additional_string)
    plt.tight_layout()
    
    if get_true_values:
        plt.savefig("figures/" + test_set_type + "_true_values_series_" + additional_string + ".pdf")
        print("Completed true values")
    else:
        plt.savefig("figures/" + test_set_type + "_" + sigopt_name + "_" + str(experimental_setting["exp_id"]) +  "_series_" + additional_string + ".pdf")
        print("Completed " + sigopt_name + " " + str(experimental_setting["exp_id"]))
    
    plt.close()


def plot_violin_all(target_prop, experimental_settings, series, num_best_models=3):
    test_set_type = "holdout_set_series"

    for experimental_setting in experimental_settings:
        sigopt_name = build_sigopt_name("data/", target_prop, experimental_setting["struct_type"], experimental_setting["interpolation"], experimental_setting["model_type"])
        directory = "./best_models/" + experimental_setting["model_type"] + "/" + sigopt_name + "/" +str(experimental_setting["exp_id"])
        test_set_dfs = []

        for i in range(num_best_models):
            with open(directory + "/" + "best_" + str(i) + "/" + test_set_type + "_predictions.json") as f:
                test_set_dfs.append(pd.read_json(f))
    
        for j in range(len(series)):
            plot_violin(target_prop, experimental_setting, series, num_best_models, test_set_dfs, test_set_type, sigopt_name, j)

            if experimental_setting == experimental_settings[-1]:
                plot_violin(target_prop, experimental_setting, series, None, test_set_dfs, test_set_type, None, j, get_true_values=True)


def plot_training_e3nn_contrastive(target_prop, experimental_settings):
    fig, axs = plt.subplots(3, 4, figsize=(16,8))
    i = 0
    for experimental_setting in experimental_settings:
        if experimental_setting["model_type"] == "e3nn_contrastive":
            for j in range(3):
                sigopt_name = build_sigopt_name("data/", target_prop, experimental_setting["struct_type"], experimental_setting["interpolation"], experimental_setting["model_type"])
                directory = "./best_models/" + experimental_setting["model_type"] + "/" + sigopt_name + "/" + str(experimental_setting["exp_id"]) + "/best_" + str(j)
                best_model = torch.load(directory + '/best_model.torch')

                steps = list(range(len(best_model['history'])))
                train_loss = [x['train']['loss'] for x in best_model['history']]
                train_direct = [x['train']['direct'] for x in best_model['history']]
                train_contrastive = [x['train']['contrastive'] for x in best_model['history']]
                valid_loss = [x['valid']['loss'] for x in best_model['history']]
                valid_direct = [x['valid']['direct'] for x in best_model['history']]
                valid_contrastive = [x['valid']['contrastive'] for x in best_model['history']]

                axs[j][i].plot(steps, train_loss, label='train_loss', color='black')
                axs[j][i].plot(steps, train_direct, label='train_direct', color='black', linestyle='dashed')
                axs[j][i].plot(steps, train_contrastive, label='train_contrastive', color='black', linestyle='dotted')
                axs[j][i].plot(steps, valid_loss, label='valid_loss', color='red')
                axs[j][i].plot(steps, valid_direct, label='valid_direct', color='red', linestyle='dashed')
                axs[j][i].plot(steps, valid_contrastive, label='valid_contrastive', color='red', linestyle='dotted')
                axs[j][i].set_ylim(0, 0.1)
                axs[j][i].set_xlim(0, 100)
        i += 1

    fig.supxlabel('Epoch')
    fig.supylabel('Loss')
    plt.tight_layout()
    plt.savefig("figures/training_e3nn_contrastive.pdf")
    plt.close()

