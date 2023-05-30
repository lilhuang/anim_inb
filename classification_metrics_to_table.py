import matplotlib.pyplot as plt
import numpy as lumpy
import os

import pdb

def write_metrics_to_txt_SU():
    root = "/fs/cfar-projects/anim_inb"
    ours_root_SU = "outputs/SU_final_SU_git_METRICS_results_"
    ours_root_suzannes = "outputs/final_suzannes_git_METRICS_results_"
    rife_root_SU = "arXiv2020-RIFE/outputs/final_SU_git_test/metrics"
    rife_root_suzannes = "arXiv2020-RIFE/outputs/final_Suzanne_exr_git/metrics"
    ai_root_SU = "AnimeInterp/outputs/final_SU_git_TEST_results/metrics"
    ai_root_suzannes = "AnimeInterp/outputs/final_suzannes_git_TEST_results/metrics"

    rife_SU_recall_path = os.path.join(root, rife_root_SU, "recall_arrs.npy")
    rife_SU_prec_path = os.path.join(root, rife_root_SU, "prec_arrs.npy")
    rife_SU_acc_path = os.path.join(root, rife_root_SU, "acc_arrs.npy")
    rife_SU_f1_path = os.path.join(root, rife_root_SU, "f1_arrs.npy")

    ai_SU_recall_path = os.path.join(root, ai_root_SU, "recall_arrs.npy")
    ai_SU_prec_path = os.path.join(root, ai_root_SU, "prec_arrs.npy")
    ai_SU_acc_path = os.path.join(root, ai_root_SU, "acc_arrs.npy")
    ai_SU_f1_path = os.path.join(root, ai_root_SU, "f1_arrs.npy")

    ours_SU_recall_path = os.path.join(root, ours_root_SU, "recall_arr_epoch_90.npy")
    ours_SU_prec_path = os.path.join(root, ours_root_SU, "prec_arr_epoch_90.npy")
    ours_SU_acc_path = os.path.join(root, ours_root_SU, "acc_arr_epoch_90.npy")
    ours_SU_f1_path = os.path.join(root, ours_root_SU, "f1_arr_epoch_90.npy")

    rife_SU_recall = lumpy.mean(lumpy.load(rife_SU_recall_path), axis=0)
    rife_SU_prec = lumpy.mean(lumpy.load(rife_SU_prec_path), axis=0)
    rife_SU_acc = lumpy.mean(lumpy.load(rife_SU_acc_path), axis=0)
    rife_SU_f1 = lumpy.mean(lumpy.load(rife_SU_f1_path), axis=0)

    ai_SU_recall = lumpy.mean(lumpy.load(ai_SU_recall_path), axis=0)
    ai_SU_prec = lumpy.mean(lumpy.load(ai_SU_prec_path), axis=0)
    ai_SU_acc = lumpy.mean(lumpy.load(ai_SU_acc_path), axis=0)
    ai_SU_f1 = lumpy.mean(lumpy.load(ai_SU_f1_path), axis=0)

    ours_SU_recall = lumpy.load(ours_SU_recall_path)
    ours_SU_prec = lumpy.load(ours_SU_prec_path)
    ours_SU_acc = lumpy.load(ours_SU_acc_path)
    ours_SU_f1 = lumpy.load(ours_SU_f1_path)

    all_recall = [ai_SU_recall, rife_SU_recall, ours_SU_recall]
    all_prec = [ai_SU_prec, rife_SU_prec, ours_SU_prec]
    all_acc = [ai_SU_acc, rife_SU_acc, ours_SU_acc]
    all_f1 = [ai_SU_f1, rife_SU_f1, ours_SU_f1]
    pdb.set_trace()
    all_metrics = [all_recall, all_prec, all_acc, all_f1]
    metrics = ["recall", "precision", "accuracy", "f1"]
    methods = ["AnimeInterp", "RIFE", "Ours"]

    default_threshold = 0.5
    thresholds_1 = lumpy.arange(10)*0.1
    thresholds_2 =  (lumpy.arange(10)*0.01)+0.9
    thresholds = lumpy.concatenate((thresholds_1, thresholds_2))

    outfile = "metrics_latex_SU.txt"
    lines = []
    for i, metric in enumerate(metrics):
        lines.append("\\toprule\n")
        firstline = metric + " thresholds"
        for threshold in thresholds:
            firstline = firstline + " & {:.2f}".format(threshold) + " "
        firstline = firstline + "\\\\\n"
        lines.append(firstline)
        lines.append("\\midrule\n")
        for j, method in enumerate(methods):
            nextline = method
            for k in range(len(thresholds)):
                nextline = nextline + " & {:.4f}".format(all_metrics[i][j][k])
            nextline = nextline + " \\\\\n"
            lines.append(nextline)
        lines.append("\\bottomrule\n\n\n")
    with open(outfile, "w") as file:
        file.writelines(lines)


def write_metrics_to_txt_suzannes():
    root = "/fs/cfar-projects/anim_inb"
    ours_root_SU = "outputs/SU_final_SU_git_METRICS_results_"
    ours_root_suzannes = "outputs/final_suzannes_git_METRICS_results_"
    rife_root_SU = "arXiv2020-RIFE/outputs/final_SU_git_test/metrics"
    rife_root_suzannes = "arXiv2020-RIFE/outputs/final_Suzanne_exr_git/metrics"
    ai_root_SU = "AnimeInterp/outputs/final_SU_git_TEST_results/metrics"
    ai_root_suzannes = "AnimeInterp/outputs/final_suzannes_git_TEST_results/metrics"

    rife_suzannes_recall_path = os.path.join(root, rife_root_suzannes, "recall_arrs.npy")
    rife_suzannes_prec_path = os.path.join(root, rife_root_suzannes, "prec_arrs.npy")
    rife_suzannes_acc_path = os.path.join(root, rife_root_suzannes, "acc_arrs.npy")
    rife_suzannes_f1_path = os.path.join(root, rife_root_suzannes, "f1_arrs.npy")

    ai_suzannes_recall_path = os.path.join(root, ai_root_suzannes, "recall_arrs.npy")
    ai_suzannes_prec_path = os.path.join(root, ai_root_suzannes, "prec_arrs.npy")
    ai_suzannes_acc_path = os.path.join(root, ai_root_suzannes, "acc_arrs.npy")
    ai_suzannes_f1_path = os.path.join(root, ai_root_suzannes, "f1_arrs.npy")

    ours_suzannes_recall_path = os.path.join(root, ours_root_suzannes, "recall_arr_epoch_40.npy")
    ours_suzannes_prec_path = os.path.join(root, ours_root_suzannes, "prec_arr_epoch_40.npy")
    ours_suzannes_acc_path = os.path.join(root, ours_root_suzannes, "acc_arr_epoch_40.npy")
    ours_suzannes_f1_path = os.path.join(root, ours_root_suzannes, "f1_arr_epoch_40.npy")

    rife_suzannes_recall = lumpy.mean(lumpy.load(rife_suzannes_recall_path), axis=0)
    rife_suzannes_prec = lumpy.mean(lumpy.load(rife_suzannes_prec_path), axis=0)
    rife_suzannes_acc = lumpy.mean(lumpy.load(rife_suzannes_acc_path), axis=0)
    rife_suzannes_f1 = lumpy.mean(lumpy.load(rife_suzannes_f1_path), axis=0)

    ai_suzannes_recall = lumpy.mean(lumpy.load(ai_suzannes_recall_path), axis=0)
    ai_suzannes_prec = lumpy.mean(lumpy.load(ai_suzannes_prec_path), axis=0)
    ai_suzannes_acc = lumpy.mean(lumpy.load(ai_suzannes_acc_path), axis=0)
    ai_suzannes_f1 = lumpy.mean(lumpy.load(ai_suzannes_f1_path), axis=0)

    ours_suzannes_recall = lumpy.load(ours_suzannes_recall_path)
    ours_suzannes_prec = lumpy.load(ours_suzannes_prec_path)
    ours_suzannes_acc = lumpy.load(ours_suzannes_acc_path)
    ours_suzannes_f1 = lumpy.load(ours_suzannes_f1_path)

    all_recall = [ai_suzannes_recall, rife_suzannes_recall, ours_suzannes_recall]
    all_prec = [ai_suzannes_prec, rife_suzannes_prec, ours_suzannes_prec]
    all_acc = [ai_suzannes_acc, rife_suzannes_acc, ours_suzannes_acc]
    all_f1 = [ai_suzannes_f1, rife_suzannes_f1, ours_suzannes_f1]

    all_metrics = [all_recall, all_prec, all_acc, all_f1]
    metrics = ["recall", "precision", "accuracy", "f1"]
    methods = ["AnimeInterp", "RIFE", "Ours"]

    default_threshold = 0.5
    thresholds_1 = lumpy.arange(10)*0.1
    thresholds_2 =  (lumpy.arange(10)*0.01)+0.9
    thresholds = lumpy.concatenate((thresholds_1, thresholds_2))

    outfile = "metrics_latex_suzannes.txt"
    lines = []
    for i, metric in enumerate(metrics):
        lines.append("\\toprule\n")
        firstline = metric + " thresholds"
        for threshold in thresholds:
            firstline = firstline + " & {:.2f}".format(threshold) + " "
        firstline = firstline + "\\\\\n"
        lines.append(firstline)
        lines.append("\\midrule\n")
        for j, method in enumerate(methods):
            nextline = method
            for k in range(len(thresholds)):
                nextline = nextline + " & {:.4f}".format(all_metrics[i][j][k])
            nextline = nextline + " \\\\\n"
            lines.append(nextline)
        lines.append("\\bottomrule\n\n\n")
    with open(outfile, "w") as file:
        file.writelines(lines)
        
        
    



if __name__ == "__main__":
    # write_metrics_to_txt_SU()
    write_metrics_to_txt_suzannes()















