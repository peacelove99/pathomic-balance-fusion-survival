import os

from datasets.dataset_generic import save_splits
from models.model_MCAT import MCAT_Surv
from utils.coattn_train_utils import *
from utils.utils import *


def train(datasets: tuple, cur: int, args: Namespace):
    """
        train for a single fold
    """
    ########################################################################################################################
    print('\nTraining Fold {}!'.format(cur))

    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('Done!')
    ########################################################################################################################
    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
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
    else:
        raise NotImplementedError

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
    else:
        raise NotImplementedError

    model = model.to(torch.device('cuda'))
    print_network(model)

    print('Done!')
    ########################################################################################################################
    print('\nInit optimizer ...', end=' ')

    optimizer = get_optim(model, args)

    print('Done!')
    ########################################################################################################################
    print('\nInit Loaders...', end=' ')

    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample,
                                    mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)

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
        train_loop_survival_coattn(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc)
        validate_survival_coattn(cur, epoch, model, val_loader, args.n_classes, early_stopping, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir)

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)
    print('Val c-Index: {:.4f}'.format(val_cindex))
    writer.close()
    return results_val_dict, val_cindex
