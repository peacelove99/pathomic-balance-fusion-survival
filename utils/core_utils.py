import os

from datasets.dataset_generic import save_splits
from models.model_MCAT import MCAT_Surv
from models.model_MOTCat import MOTCAT_Surv
from models.model_PGBF import PGBF_Surv
from models.model_tmi2024 import GraphMixer_Surv
from utils.coattn_train_utils import *
from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(datasets: tuple, cur: int, args: Namespace):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    ####################################################################################################################
    # 创建一折结果目录 writer_dir = args.results_dir/str(cur) eg. /results/5foldcv/mcat_coattn/tcga_luad_s1/0
    args.writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)
    # 每15秒将缓存的数据刷新到磁盘。
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.writer_dir, flush_secs=15)
    else:
        writer = None
    ####################################################################################################################
    print('\nInit train/val/test splits...', end=' ')

    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('Done!')
    ########################################################################################################################
    print('\nInit loss function...', end=' ')
    # if args.task_type == 'survival':
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        print("Training with CrossEntropySurvLoss")
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        print("Training with NLLSurvLoss")
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
        print("Training with CoxSurvLoss")
    else:
        raise NotImplementedError
    # else:
    #     raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    ########################################################################################################################
    print('\nInit Model...', end=' ')

    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type == 'mcat':
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'motcat':
        model_dict = {'ot_reg': args.ot_reg, 'ot_tau': args.ot_tau, 'ot_impl': args.ot_impl, 'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MOTCAT_Surv(**model_dict)
    elif args.model_type == 'graphmixer':
        model_dict = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes, 'omic_sizes': train_split.omic_sizes, 'num_features': args.input_dim}
        model = GraphMixer_Surv(**model_dict)
    elif args.model_type == 'pgbf':
        model_dict = {'omic_sizes': args.omic_sizes}
        model = PGBF_Surv(**model_dict)
    else:
        raise NotImplementedError

    model = model.to(device)
    print_network(model)

    print('Done!')
    ########################################################################################################################
    print('\nInit optimizer ...', end=' ')

    optimizer = get_optim(model, args)

    print('Done!')
    ########################################################################################################################
    print('\nInit Loaders...', end=' ')

    train_loader = get_split_loader(train_split, training=True, testing=False, weighted=args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing=False, mode=args.mode, batch_size=args.batch_size)

    print('Done!')
    ########################################################################################################################
    print('\nSetup EarlyStopping...', end=' ')

    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=30, verbose=True)
    else:
        early_stopping = None

    print('Done!')
    ########################################################################################################################
    print('\nSetup Validation C-Index Monitor...', end=' ')

    monitor_cindex = Monitor_CIndex()

    print('Done!')
    ########################################################################################################################
    print("running with {} {}".format(args.model_type, args.mode))

    max_c_index = 0.
    epoch_max_c_index = 0
    best_val_dict = {}

    for epoch in range(args.start_epoch, args.max_epochs):
        train_loop_survival_coattn(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, args)
        val_latest, c_index_val, stop = validate_survival_coattn(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args)


        if c_index_val > max_c_index:
            max_c_index = c_index_val
            epoch_max_c_index = epoch
            save_name = 's_{}_checkpoint'.format(cur)

            torch.save(model.state_dict(), os.path.join(args.results_dir, save_name + ".pt".format(cur)))
            best_val_dict = val_latest

    if args.log_data:
        writer.close()
    print("================= summary of fold {} ====================".format(cur))
    print_results = {'result': (max_c_index, epoch_max_c_index)}
    print("result: {:.4f}".format(max_c_index))
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write('result: {:.4f}, epoch: {}\n'.format(max_c_index, epoch_max_c_index))

    return best_val_dict, print_results
