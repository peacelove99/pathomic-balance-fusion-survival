import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
from timeit import default_timer as timer

from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.core_utils import train
from utils.file_utils import save_pkl

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

    parser.add_argument('--mode', type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'],
                        default='coattn', help='Specifies which modalities to use / collate function in dataloader.')

    parser.add_argument('--k', type=int,
                        default=5, help='Number of folds (default: 5)')
    parser.add_argument('--k_start', type=int,
                        default=-1, help='Start fold (Default: -1, first fold)')
    parser.add_argument('--k_end', type=int,
                        default=-1, help='End fold (Default: -1, last fold)')

    parser.add_argument('--apply_sig', action='store_true',
                        default=True, help='Use genomic features as signature embeddings.')

    parser.add_argument('--data_root_dir', type=str,
                        default='/home/cvnlp/WSI_DATA/TCGA_LUAD_feature', help='Data directory to WSI features (extracted via CLAM')
                        # default='/media/lenovo/D2B96B35B0D939DD/WSI_DATA/TCGA_LUAD_feature', help='Data directory to WSI features (extracted via CLAM')

    parser.add_argument('--log_data', action='store_true',
                        default=False, help='Log data using tensorboard')

    parser.add_argument('--loss', type=str, choices=['ce_surv', 'nll_surv', 'nll_surv_kl', 'nll_surv_mse', 'nll_surv_l1', 'nll_surv_cos', 'nll_surv_ol'],
                        default='nll_surv_l1', help='slide-level classification loss function (default: ce)')

    parser.add_argument('--alpha_surv', type=float,
                        default=0.0, help='How much to weigh uncensored patients')

    parser.add_argument('--reg_type', type=str, choices=['None', 'omic', 'pathomic'],
                        default='None', help='Which network submodules to apply L1-Regularization (default: None)')

    parser.add_argument('--drop_out', action='store_true',
                        default=True, help='Enable dropout (p=0.25)')

    parser.add_argument('--fusion', type=str, choices=['None', 'concat', 'bilinear'],
                        default='concat', help='Type of fusion. (Default: concat).')

    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'],
                        default='adam')

    parser.add_argument('--lr', type=float,
                        default=2e-4, help='Learning rate (default: 0.0001)')

    parser.add_argument('--reg', type=float,
                        default=1e-5, help='L2-regularization weight decay (default: 1e-5)')

    parser.add_argument('--weighted_sample', action='store_true',
                        default=True, help='Enable weighted sampling')

    parser.add_argument('--batch_size', type=int,
                        default=1, help='Batch Size (Default: 1, due to varying bag sizes)')

    parser.add_argument('--early_stopping', action='store_true',
                        default=False, help='Enable early stopping')

    parser.add_argument('--start_epoch', type=int,
                        default=0, help='start_epoch.')

    parser.add_argument('--max_epochs', type=int,
                        default=20, help='Maximum number of epochs to train (default: 20)')

    parser.add_argument('--lambda_reg', type=float,
                        default=1e-4, help='L1-Regularization Strength (Default 1e-4)')

    parser.add_argument('--gc', type=int,
                        default=32, help='Gradient Accumulation Step.')

    parser.add_argument('--ot_reg', type=float,
                        default=0.05, help='epsilon of OT (default: 0.1)')

    parser.add_argument('--ot_tau', type=float,
                        default=0.5, help='tau of UOT (default: 0.5)')

    parser.add_argument('--ot_impl', type=str,
                        default='pot-uot-l2', help='impl of ot (default: pot-uot-l2)')

    parser.add_argument('--alpha', type=float,
                        default=0.0001, help='impl of ot (default: pot-uot-l2)')



    return parser.parse_args()


def seed_torch(seed=2024):
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
    print("################# Settings ###################")

    # print(args)
    for attr, value in vars(args).items():
        print(f"{attr.replace('_', ' ').capitalize()}: {value}")

    print('Done!\n')
    ########################################################################################################################

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
    summary_all_folds = {}
    print("开始{}折交叉验证".format(folds[-1]))
    for i in folds:
        print("################# Training {} fold ###################".format(i))
        start_time = timer()  # 计时开始

        seed_torch(args.seed)  # 设置随机种子
        args.results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))  # 保存第 i 折结果的地址
        if os.path.isfile(args.results_pkl_path):  # 第 i 折结果存在 则跳过
            print("Skipping Split %d" % i)
            continue
        print('\n加载数据')

        args.n_classes = 4

        csv_path = './dataset_csv/%s_all_clean.csv.zip' % args.task
        print("csv_path:", csv_path)
        dataset = Generic_MIL_Survival_Dataset(csv_path=csv_path,
                                               mode=args.mode,
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

        args.omic_sizes = train_dataset.omic_sizes
        print('Genomic Dimensions', args.omic_sizes)

        # val_latest, cindex_latest = train(datasets, i, args)
        summary_results, print_results = train(datasets, i, args)

        # latest_val_cindex.append(cindex_latest)  # 保存第 i 折cindex
        save_pkl(args.results_pkl_path, summary_results)  # 保存第 i 折val_latest
        summary_all_folds[i] = print_results

        end_time = timer()  # 计时结束
        print('Fold %d Time: %f seconds' % (i, end_time - start_time))

    print('=============================== summary ===============================')
    result_cindex = []
    for i, k in enumerate(summary_all_folds):
        c_index = summary_all_folds[k]['result'][0]
        print("Fold {}, C-Index: {:.4f}".format(k, c_index))
        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write("Fold {}, C-Index: {:.4f}\n".format(k, c_index))
        result_cindex.append(c_index)
    result_cindex = np.array(result_cindex)
    print("Avg C-Index of {} folds: {:.3f}, stdp: {:.3f}, stds: {:.3f}".format(
        len(summary_all_folds), result_cindex.mean(), result_cindex.std(), result_cindex.std(ddof=1)))

    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        for attr, value in vars(args).items():
            f.write(f"{attr.replace('_', ' ').capitalize()}: {value}\n")
        f.write("Avg C-Index of {} folds: {:.3f}, stdp: {:.3f}, stds: {:.3f}\n".format(
        len(summary_all_folds), result_cindex.mean(), result_cindex.std(), result_cindex.std(ddof=1)))

    return result_cindex.mean()

    # results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})
    # save_name = 'summary.csv'
    #
    # results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))


if __name__ == "__main__":
    start = timer()

    args = parse()
########################################################################################################################
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    print("结果目录:", args.results_dir)

    # model_set = args.model_type + '_' + args.mode
    # experiment_set = args.task + '_s{}'.format(args.seed)
    #
    # args.results_dir = os.path.join(args.results_dir, args.which_splits, model_set, experiment_set)
    # if not os.path.isdir(args.results_dir):
    #     os.makedirs(args.results_dir)
    # print("结果子目录:", args.results_dir)

    if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):  # 防止重复实验
        print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
        sys.exit()

    args.split_dir = os.path.join('./splits', args.which_splits, args.task)
    print("数据分割目录:", args.split_dir)
########################################################################################################################

    best_rest = 0
    best_set = ""

    for a in [0, 1]:
        args.omic_encoder = a
        for b in [0, 1, 2]:
            args.path_encoder = b
            for c in ["TMI_2024", "MOTCat", "MCAT", "CMTA"]:
                args.coattn_model = c
                for d in [0, 1, 2, 3]:
                    args.path_decoder = d
                    for e in [0, 1, 2]:
                        args.omic_decoder = e
                        for f in [0, 1]:
                            args.fusion_layer = f
                            for g in [6, 12, 18, 24, 30]:
                                args.topk = g
                                for h in [0.05, 0.1]:
                                    args.ot_reg = h
                                    for i in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
                                        args.dropout = i

                                        experiment_set = f"{a}_{b}_{c}_{d}_{e}_{f}_{g}_{h:.2f}_{i:.2f}"
                                        args.results_dir = os.path.join(args.results_dir, experiment_set)
                                        if not os.path.isdir(args.results_dir):
                                            os.makedirs(args.results_dir)


                                        result = main(args)
                                        if result > best_rest:
                                            best_rest = result
                                            best_set = experiment_set
    print('best_rest', best_rest)
    print('best_set', best_set)


    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
