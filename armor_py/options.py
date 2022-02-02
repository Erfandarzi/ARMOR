import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Client_noise_adv')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--global_noise_scale', type=float, default=0)
    parser.add_argument('--clip_threshold', type=int, default=20)
    parser.add_argument('--num_items_server', type=int, default=1000, help="number of samples for adversarial training")
    parser.add_argument('--comm_round', type=int, default=50, help="rounds of training")
    parser.add_argument('--client_num_in_total', type=int, default=50, help="number of users in total")
    parser.add_argument('--batch_size', type=int, default=10, help="local batch size")
    parser.add_argument('--epoch', type=int, default=1, help="local epoch")
    parser.add_argument('--lr', type=float, default=0.07, help='learning rate')
    parser.add_argument('--retrain_round', type=int, default=1, help="num of server retrain round")
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether iid or not, 1 for iid, 0 for non-iid')
    args = parser.parse_args()
    return args
