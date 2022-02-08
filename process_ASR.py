import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from armor_py.options import args_parser
from armor_py.utils import alter_re, del_blank_line, alter, test_mkdir, test_cpdir


def asr_per_process():
    model_name = args.model_name
    client_num_in_total = args.client_num_in_total
    dataset = args.dataset
    path = dataset_path + "client_num_{}/Attack_fixed/Client_".format(client_num_in_total) + model_name + "/"
    path_log = path + "attack_log/"

    path_acc = path + "Acc/"
    path_asr = path + "ASR/"
    path_aatr = path + "AATR/"
    pic_path = dataset_path + "result_pic/"
    out_path = dataset_path + "result_out/"

    test_mkdir(pic_path)
    test_mkdir(out_path)
    test_cpdir(path_log, path_acc)
    test_cpdir(path_log, path_asr)
    test_mkdir(path_aatr)

    src_path_AATR = dataset_path + "ATR_pic/out/" + args.dataset + "_client_num_{}_{}.out".format(client_num_in_total,model_name)
    full_path_AATR = path_aatr + args.dataset + "_client_num_{}_{}.out".format(client_num_in_total,model_name)
    shutil.copyfile(src_path_AATR, full_path_AATR)

    files_acc = []
    for file_acc in os.listdir(path_acc):
        if file_acc.endswith(".out"):
            files_acc.append(path_acc + file_acc)

    files_asr = []
    for file_asr in os.listdir(path_asr):
        if file_asr.endswith(".out"):
            files_asr.append(path_asr + file_asr)

    ### Acc ###
    for file in files_acc:
        alter_re(file, "noise_scale=.*", "")
        alter_re(file, "Model Path.*", "")
        alter(file, "nohup: ignoring input", "")
        alter(file, "################################ Attack begin ################################", "")
        alter(file, "##############################################################################", "")
        alter(file, "Adversary Examples Generated on Client ", "")
        alter_re(file, "Test on Client .* Acc: ", "")
        alter_re(file, "\(%\) / ASR: .*", "")
        del_blank_line(file)

    ### ASR ###
    for file in files_asr:
        alter_re(file, "noise_scale=.*", "")
        alter_re(file, "Model Path.*", "")
        alter(file, "nohup: ignoring input", "")
        alter(file, "################################ Attack begin ################################", "")
        alter(file, "##############################################################################", "")
        alter(file, "Adversary Examples Generated on Client ", "")
        alter_re(file, "Test on Client .* ASR: ", "")
        alter_re(file, "\(%\).*", "")
        del_blank_line(file)

    file_data = "Noise\tAttack\tAcc_avg\tASR_avg\tAcc_std\tASR_std\n"
    result_file = out_path + dataset + "_client_num_{}".format(client_num_in_total) + "_" + model_name + ".out"
    acc_list = []
    asr_list = []
    aatr_list = []
    self_list = []

    ########################### loading AATR ##########################
    alter_re(full_path_AATR, "Noise.*", "")
    alter_re(full_path_AATR, "0..*\t", "")
    alter(full_path_AATR, "%", "")
    del_blank_line(full_path_AATR)
    file_AATR = open(full_path_AATR)
    for i in file_AATR:
        aatr_list.append(float(i) / 100)

    #################### calculate average Acc ASR ####################

    for global_noise_scale in noise_scale:
        full_path_Acc = path_acc + prefix + "_{:.3f}.out".format(global_noise_scale)
        full_path_ASR = path_asr + prefix + "_{:.3f}.out".format(global_noise_scale)
        file_Acc = open(full_path_Acc)
        file_ASR = open(full_path_ASR)
        acc, asr, asr_self = [0] * client_num_in_total, [0] * client_num_in_total, [0] * client_num_in_total
        acc_avg, acc_var, acc_std = [0] * client_num_in_total, [0] * client_num_in_total, [0] * client_num_in_total
        asr_avg, asr_var, asr_std = [0] * client_num_in_total, [0] * client_num_in_total, [0] * client_num_in_total

        idx = 0
        for i in file_Acc:
            if idx % (client_num_in_total+1) == 0:
                generated_idx = int(i)
                acc[generated_idx] = [0] * client_num_in_total
            else:
                acc[generated_idx][idx % (client_num_in_total+1) - 1] = float(i) / 100
            idx = idx + 1

        idx = 0
        for i in file_ASR:
            if idx % (client_num_in_total+1) == 0:
                generated_idx = int(i)
                asr[generated_idx] = [0] * client_num_in_total
            else:
                asr[generated_idx][idx % (client_num_in_total+1) - 1] = float(i) / 100
            idx = idx + 1

        for generated_idx in range(client_num_in_total):
            asr_self[generated_idx] = asr[generated_idx][generated_idx]
            del asr[generated_idx][generated_idx]
            asr_avg[generated_idx] = np.average(asr[generated_idx])
            asr_var[generated_idx] = np.var(asr[generated_idx])
            asr_std[generated_idx] = np.std(asr[generated_idx])

            acc_avg[generated_idx] = np.average(acc[generated_idx])
            acc_var[generated_idx] = np.var(acc[generated_idx])
            acc_std[generated_idx] = np.std(acc[generated_idx])

        asr_self_avg = np.average(asr_self)
        asr_avg_avg = np.average(asr_avg)
        asr_std_avg = np.average(asr_std)

        acc_avg_avg = np.average(acc_avg)
        acc_std_avg = np.average(acc_std)

        acc_list.append(acc_avg_avg)
        asr_list.append(asr_avg_avg)
        self_list.append(asr_self_avg)

        line = "noise={:.3f} asr_self_avg={:.2f}% Acc_avg={:.2f}% ASR_avg={:.2f}% Acc_std={:.3f} ASR_std={:.3f}\n". \
            format(global_noise_scale, asr_self_avg * 100,
                   acc_avg_avg * 100, asr_avg_avg * 100,
                   acc_std_avg, asr_std_avg)
        file_data += line

    with open(result_file, "w", encoding="utf-8") as f:
        f.write(file_data)

    alter_re(result_file, "noise=", "")
    alter_re(result_file, " asr_self_avg=", "\t")
    alter_re(result_file, " Acc_avg=", "\t")
    alter_re(result_file, " ASR_avg=", "\t")
    alter_re(result_file, " Acc_std=", "\t")
    alter_re(result_file, " ASR_std=", "\t")
    del_blank_line(result_file)

    plt.figure()
    fontsize_ticks = 22
    fontsize_label = 26
    fontsize_legend = 16
    markersize = '5'
    linewidth = 1.5

    x = noise_scale
    plt.xticks(fontsize=fontsize_ticks)
    plt.xlabel(r'noise scale $\sigma$',fontsize=fontsize_label)

    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0.2, 1.1, 0.2),fontsize=fontsize_ticks)
    plt.ylabel("Acc/ASR",fontsize=fontsize_label)

    plt.grid(True, linestyle='-.')
    plt.plot(x, acc_list, color="r", linestyle="-.", marker="^", markersize=markersize, linewidth=linewidth,
             label="Acc of all clients")
    plt.plot(x, self_list, color="g", linestyle=":", marker="*", markersize=markersize, linewidth=linewidth,
             label="ASR on adversary")
    plt.plot(x, asr_list, color="b", linestyle="-", marker="s", markersize=markersize, linewidth=linewidth,
             label="ASR on benign")
    plt.plot(x, aatr_list, color="m", linestyle="-", marker="s", markersize=markersize, linewidth=linewidth,
             label="AATR on benign")
    plt.legend(loc='upper right',fontsize=fontsize_legend,framealpha=0.5)

    plt.tight_layout()
    plt.savefig(pic_path + dataset + "_client_num_{}".format(client_num_in_total) + "_" + model_name + ".pdf")
    plt.close()
    print(dataset + " Client Num = {} ".format(client_num_in_total) + model_name + " finished")


if __name__ == '__main__':
    args = args_parser()
    prefix = "pgd"
    args.dataset = "cifar"
    # args.dataset = "mnist"

    if args.dataset == "cifar":
        noise_scale = np.linspace(0, 0.12, 25)
    elif args.dataset == "mnist":
        noise_scale = np.linspace(0, 0.32, 33)
    model_list = ['no_noise_adv', 'noise_adv', 'noise_only']

    dataset_path = "by_client/" + args.dataset + "/"

    for args.client_num_in_total in [5, 25, 50, 75, 100]:
        for args.model_name in model_list:
            asr_per_process()
