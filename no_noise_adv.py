import copy
import os
import matplotlib.image
import numpy as np
import torch
from easydict import EasyDict
from attack.projected_gradient_descent import projected_gradient_descent
from armor_py.utils import del_tensor_element, fix_random
from armor_py.models import CNN_CIFAR, CNN_MNIST
from armor_py.options import args_parser
from armor_py.sampling import ld_mnist, ld_cifar10
from armor_py.update import ServerRetrain
np.set_printoptions(threshold=np.inf)
matplotlib.use('Agg')


def per_attack_no_noise():
    fix_random(1)
    Client, Client_retrained = {}, {}
    x, ae_x, y, ae_y = {}, {}, {}, {}
    path = "by_client/" + args.dataset + "/client_num_{}/".format(args.client_num_in_total)
    prefix = "Global"
    gan_prefix = "Client_no_noise_adv/"
    pgd_data = ""
    list_data = ""

    pgd_path = path + "Attack_fixed/" + gan_prefix + "/attack_log/"
    if not os.path.exists(pgd_path):
        os.makedirs(pgd_path)

    list_path = path + "Attack_fixed/" + gan_prefix + "/attack_list/"
    if not os.path.exists(list_path):
        os.makedirs(list_path)

    pgd_file = pgd_path + "pgd_{:.3f}".format(args.global_noise_scale) + ".out"
    attack_list = list_path + "attack_list_{:.3f}".format(args.global_noise_scale) + ".out"
    pgd_data += "noise_scale={:.3f}, eps={:.3f}, eps_step={:.3f}, iter_round={}\n".\
        format(args.global_noise_scale, eps, eps_step, iter_round)
    list_data += "noise_scale={:.3f}, eps={:.3f}, eps_step={:.3f}, iter_round={}\n".\
        format(args.global_noise_scale, eps, eps_step, iter_round)

    gan_path = path + gan_prefix + "{:.3f}/".format(args.global_noise_scale)
    if not os.path.exists(gan_path):
        os.makedirs(gan_path)

    # generate adversary examples
    net = []
    for corrupted_idx in range(args.client_num_in_total):
        net.append(copy.deepcopy(net_glob))
        file_path = path + prefix + "/" + prefix + "_{:.3f}.pth".format(args.global_noise_scale)
        net[corrupted_idx].load_state_dict(torch.load(file_path))
        net[corrupted_idx].eval()

        test_round = 0
        for images, labels in data.test:
            fix_random(corrupted_idx)
            if test_round >= 1:
                break
            images = images.to(device)
            labels = labels.to(device)
            images_pgd = projected_gradient_descent(net[corrupted_idx], images, eps, eps_step, iter_round, np.inf)
            x[corrupted_idx] = images
            ae_x[corrupted_idx] = images_pgd
            y[corrupted_idx] = labels
            _, ae_y[corrupted_idx] = net[corrupted_idx](images_pgd).max(1)
            test_round += 1

    # use adversary examples to retrain
    w_locals, loss_locals, acc_locals = [], [], []
    for corrupted_idx in range(args.client_num_in_total):
        fix_random(corrupted_idx)
        retrain = ServerRetrain(args, ae_x[corrupted_idx], y[corrupted_idx], device)
        net[corrupted_idx].train()
        w, loss, acc = retrain.update_weights(net=copy.deepcopy(net[corrupted_idx]), device=device)
        w_locals.append(copy.deepcopy(w))
        net[corrupted_idx].load_state_dict(w)
        loss_locals.append(copy.deepcopy(loss))
        acc_locals.append(copy.deepcopy(acc))
        Client_retrained[corrupted_idx] = w
        torch.save(Client_retrained[corrupted_idx], gan_path + "Client_retrained_{}.pth".format(corrupted_idx))

    # attack by each client
    pgd_data += "################################ Attack begin ################################\n"
    list_data += "################################ Attack begin ################################\n"

    for corrupted_idx in range(args.client_num_in_total):
        fix_random(corrupted_idx)
        pgd_data += "##############################################################################\n"
        pgd_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        list_data += "##############################################################################\n"
        list_data += "Adversary Examples Generated on Client {}\n".format(corrupted_idx)

        test_round = 0
        for images, labels in data.test:
            if test_round >= 1:
                break
            images = images.to(device)
            labels = labels.to(device)
            images_pgd = projected_gradient_descent(net[corrupted_idx], images, eps, eps_step, iter_round, np.inf)

            for idx in range(args.client_num_in_total):
                report = EasyDict(nb_test=0, nb_correct=0, correct_pgd_predict=0, correct_pgd_in_corrected=0)
                _, y_pred = net[idx](images).max(1)
                _, y_pred_pgd = net[idx](images_pgd).max(1)
                report.nb_test += labels.size(0)
                report.nb_correct += y_pred.eq(labels).sum().item()

                # 0 predict incorrectly
                # 1 predict correctly & attack failed
                # 2 predict correctly & attack succeed

                list_mask = y_pred.eq(labels)
                list_value = ~y_pred_pgd.eq(y_pred)                             # attack successfully
                list_result = (list_mask & list_value).long().cpu().numpy()     # predict correctly & attack successfully 2
                list_mask = list_mask.long().cpu().numpy()                      # predict correctly 1
                list_result = str(list_result + list_mask).replace("\n", "")
                list_result = list_result.replace("[", "")
                list_result = list_result.replace("]", "")
                list_data += list_result + "\n"

                y_pred_correct = y_pred
                y_pred_correct_pgd = y_pred_pgd
                for i in range(images.shape[0]):
                    if y_pred[images.shape[0] - 1 - i] != labels[images.shape[0] - 1 - i]:
                        y_pred_correct = del_tensor_element(y_pred_correct, images.shape[0] - 1 - i)
                        y_pred_correct_pgd = del_tensor_element(y_pred_correct_pgd, images.shape[0] - 1 - i)
                report.correct_pgd_in_corrected += y_pred_correct_pgd.eq(y_pred_correct).sum().item()

                if idx == corrupted_idx:
                    pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%) ************* Generated\n".format(
                        idx,
                        (report.nb_correct / report.nb_test * 100.0),
                        ((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0))
                else:
                    pgd_data += "Test on Client {}: Clean Acc: {:.2f}(%) / ASR: {:.2f}(%)\n".format(
                        idx,
                        (report.nb_correct / report.nb_test * 100.0),
                        ((1 - report.correct_pgd_in_corrected / report.nb_correct) * 100.0))

                with open(pgd_file, "w", encoding="utf-8") as f:
                    f.write(pgd_data)

                with open(attack_list, "w", encoding="utf-8") as f:
                    f.write(list_data)

            pgd_data += "##############################################################################\n"
            list_data += "##############################################################################\n"

            with open(pgd_file, "w", encoding="utf-8") as f:
                f.write(pgd_data)
            test_round += 1


if __name__ == '__main__':
    args = args_parser()
    device = torch.device("cuda:{}".format(args.cuda))
    fix_random(0)

    if args.dataset == "cifar":
        args.clip_threshold = 30
        eps_step = 0.008
        iter_round = 20
        eps = 0.025
        data = ld_cifar10()
        net_glob = CNN_CIFAR()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 1000

    elif args.dataset == "mnist":
        args.clip_threshold = 20
        eps_step = 0.01
        iter_round = 40
        eps = 0.2
        data = ld_mnist()
        net_glob = CNN_MNIST()
        net_glob.to(device)
        net_glob.eval()
        args.random_seed = 50

    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} begins...".format(args.client_num_in_total, args.global_noise_scale))
    per_attack_no_noise()
    print("dataset = " + args.dataset + ", num of client = {} , noise = {:.3f} completed!".format(args.client_num_in_total, args.global_noise_scale))
