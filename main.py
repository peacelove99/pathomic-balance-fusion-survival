import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from timeit import default_timer as timer

from datasets.dataset_survival import Generic_MIL_Survival_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse():
    parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')

    parser.add_argument('--seed', type=int,
                        default=1, help='Random seed for reproducible experiment (default: 1)')

    parser.add_argument('--task', type=str,
                        default='tcga_luad', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_luad)')

    parser.add_argument('--results_dir', type=str,
                        default='./results', help='Results directory (Default: ./results)')

    parser.add_argument('--which_splits', type=str,
                        default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')

    parser.add_argument('--model_type', type=str, choices=['snn', 'amil', 'mcat', 'motcat', 'pgbf'],
                        default='pgbf', help='Type of model (Default: pgbf)')

    parser.add_argument('--mode_data', type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'],
                        default='coattn', help='Specifies which modalities to use / collate function in dataloader.')

    parser.add_argument('--k', type=int,
                        default=5, help='Number of folds (default: 5)')
    parser.add_argument('--k_start', type=int,
                        default=-1, help='Start fold (Default: -1, first fold)')
    parser.add_argument('--k_end', type=int,
                        default=-1, help='End fold (Default: -1, last fold)')

    parser.add_argument('--apply_sig', action='store_true',
                        default=False, help='Use genomic features as signature embeddings.')

    parser.add_argument('--data_root_dir', type=str,
                        default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')

    return parser.parse_args()


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    if args.k_start == -1:
        s_fold = 0
    else:
        s_fold = args.k_start
    if args.k_end == -1:
        e_fold = args.k
    else:
        e_fold = args.k_end
    print("开始折{},结束折{}".format(s_fold, e_fold))

    latest_val_cindex = []  # 存每一折的 cindex
    folds = np.arange(s_fold, e_fold)
    print("开始{}折交叉验证".format(folds[-1]))
    for i in folds:
        print("################# Training {} fold ###################".format(i))
        start_time = timer()  # 计时开始

        seed_torch(args.seed)  # 设置随机种子
        results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))  # 保存第 i 折结果的地址
        if os.path.isfile(results_pkl_path):  # 第 i 折结果存在 则跳过
            print("Skipping Split %d" % i)
            continue
        print('\n加载数据')

        csv_path = './dataset_csv/%s_all_clean.csv.zip' % args.task
        print("csv_path:", csv_path)
        dataset = Generic_MIL_Survival_Dataset(csv_path=csv_path,
                                               mode=args.mode_data,
                                               apply_sig=args.apply_sig,
                                               data_dir=args.data_root_dir,
                                               shuffle=False,
                                               seed=args.seed,
                                               print_info=True,
                                               patient_strat=False,
                                               n_bins=4,
                                               label_col='survival_months',
                                               ignore=[])
        train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset)
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))

        # args.omic_sizes = train_dataset.omic_sizes
        # print('Genomic Dimensions', args.omic_sizes)

        # val_latest, cindex_latest = train(datasets, i, args)
        # latest_val_cindex.append(cindex_latest)  # 保存第 i 折cindex
        # save_pkl(results_pkl_path, val_latest)  # 保存第 i 折val_latest

        end_time = timer()  # 计时结束
        print('Fold %d Time: %f seconds' % (i, end_time - start_time))

    # results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})
    # save_name = 'summary.csv'

    # results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))


if __name__ == "__main__":
    start = timer()

    args = parse()

    seed_torch(args.seed)  # 设置随机种子
    print("设置随机种子:", args.seed)
    encoding_size = 1024
########################################################################################################################
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    print("结果目录:", args.results_dir)

    model_set = args.model_type + '_' + args.mode_data
    experiment_set = args.task + '_s{}'.format(args.seed)

    args.results_dir = os.path.join(args.results_dir, args.which_splits, model_set, experiment_set)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    print("结果子目录:", args.results_dir)

    if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):  # 防止重复实验
        print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
        sys.exit()

    args.split_dir = os.path.join('./splits', args.which_splits, args.task)
    print("数据分割目录:", args.split_dir)
########################################################################################################################
    print("################# Settings ###################")

    # print(args)
    for attr, value in vars(args).items():
        print(f"{attr.replace('_', ' ').capitalize()}: {value}")

    print('Done!\n')
########################################################################################################################

    main(args)

    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
