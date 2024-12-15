import os

from torch import nn
from tqdm import tqdm

from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(epoch, data_loader, model, criterion, optimizer, args, writer):
    model.train()

    train_loss = 0.0
    all_risk_scores = np.zeros((len(data_loader)))
    all_censorships = np.zeros((len(data_loader)))
    all_event_times = np.zeros((len(data_loader)))
    # dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(epoch))

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_name) in enumerate(data_loader):
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        slide_id = data_loader.dataset.slide_data['slide_id'].iloc[batch_idx]
        # print('slide_id:', slide_id)  # debug slide_name

        result, result_omic, result_path = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        sur_loss = criterion[0](hazards=result['hazards'], S=result['S'], Y=label, c=c)
        sim_loss_omic = criterion[1](result_omic['encoder'].detach(), result_omic['decoder'])
        sim_loss_path = criterion[1](result_path['encoder'].detach(), result_path['decoder'])
        loss = sur_loss + args.alpha * (sim_loss_omic + sim_loss_path)
        train_loss += loss.item()

        # if reg_fn is None:
        #     loss_reg = 0
        # else:
        #     loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(result['S'], dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(data_loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate(epoch, data_loader, model, criterion, args, writer):
    model.eval()

    val_loss = 0.0
    all_risk_scores = np.zeros((len(data_loader)))
    all_censorships = np.zeros((len(data_loader)))
    all_event_times = np.zeros((len(data_loader)))
    # dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(epoch))

    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c,slide_name) in enumerate(data_loader):
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        slide_id = data_loader.dataset.slide_data['slide_id'].iloc[batch_idx]
        # print('slide_id:', slide_id)  # debug slide_name

        with torch.no_grad():
            result, result_omic, result_path = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        sur_loss = criterion[0](hazards=result['hazards'], S=result['S'], Y=label, c=c)
        sim_loss_omic = criterion[1](result_omic['encoder'].detach(), result_omic['decoder'])
        sim_loss_path = criterion[1](result_path['encoder'].detach(), result_path['decoder'])
        loss = sur_loss + args.alpha * (sim_loss_omic + sim_loss_path)
        val_loss += loss.item()
        # if reg_fn is None:
        #     loss_reg = 0
        # else:
        #     loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(result['S'], dim=1).cpu().numpy()  # 计算风险分数
        all_risk_scores[batch_idx] = risk  # 存储风险分数
        all_censorships[batch_idx] = c.cpu().numpy()  # 存储审查状态
        all_event_times[batch_idx] = event_time  # 存储事件时间
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time.item(), 'censorship': c.item()}})

    val_loss /= len(data_loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    val_epoch_str = "val c-index: {:.4f}".format(c_index)
    print(val_epoch_str, 'val_loss:', val_loss)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str + '\n')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    # if early_stopping:
    #     assert results_dir
    #     early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
    #
    #     if early_stopping.early_stop:
    #         print("Early stopping")
    #         return True

    return c_index
    # return patient_results, c_index
