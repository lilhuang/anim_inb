import matplotlib.pyplot as plt
import numpy as lumpy
import os

import pdb

def plot():
    root = "/fs/cfar-projects/anim_inb"
    # ours_root = "outputs/SU_final_SU_git_METRICS_results_"
    ours_root = "outputs/final_suzannes_git_METRICS_results_"
    # rife_root = "arXiv2020-RIFE/outputs/final_SU_git_test/metrics"
    rife_root = "arXiv2020-RIFE/outputs/final_Suzanne_exr_git/metrics"
    # ai_root = "AnimeInterp/outputs/final_SU_git_TEST_results/metrics"
    ai_root = "AnimeInterp/outputs/final_suzannes_git_TEST_results/metrics"

    rife_recall_path = os.path.join(root, rife_root, "recall_arrs.npy")
    rife_prec_path = os.path.join(root, rife_root, "prec_arrs.npy")
    ai_recall_path = os.path.join(root, ai_root, "recall_arrs.npy")
    ai_prec_path = os.path.join(root, ai_root, "prec_arrs.npy")
    # ours_recall_path = os.path.join(root, ours_root, "recall_arr_epoch_90.npy")
    ours_recall_path = os.path.join(root, ours_root, "recall_arr_epoch_40.npy")
    # ours_prec_path = os.path.join(root, ours_root, "prec_arr_epoch_90.npy")
    ours_prec_path = os.path.join(root, ours_root, "prec_arr_epoch_4d0.npy")

    rife_recall = lumpy.mean(lumpy.load(rife_recall_path), axis=0)
    rife_prec = lumpy.mean(lumpy.load(rife_prec_path), axis=0)
    ai_recall = lumpy.mean(lumpy.load(ai_recall_path), axis=0)
    ai_prec = lumpy.mean(lumpy.load(ai_prec_path), axis=0)
    ours_recall = lumpy.load(ours_recall_path)
    ours_prec = lumpy.load(ours_prec_path)

    plt.plot(ours_recall, ours_prec, 'blue', label="ours")
    plt.plot(rife_recall, rife_prec, 'green', label="rife")
    plt.plot(ai_recall, ai_prec, 'orange', label="animeinterp")
    plt.legend()
    plt.xlabel("recall")
    plt.ylabel("precision")
    figpath = os.path.join(root, "all_roc_suzannes.png")
    plt.savefig(figpath)
    plt.clf()



if __name__ == "__main__":
    plot()















