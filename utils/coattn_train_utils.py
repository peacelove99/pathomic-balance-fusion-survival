import os

from torch import nn

from utils.utils import *


def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, args=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_name) in enumerate(loader):
        # print('slide_name:',slide_name)  # debug slide_name

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        # hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        # loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        result, result_omic, result_path = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        loss = loss_fn(hazards=result['hazards'], S=result['S'], Y=label, c=c)
        loss_value = loss.item()
        loss_omic = loss_fn(hazards=result_omic['hazards'], S=result_omic['S'], Y=label, c=c)
        loss_omic_value = loss_omic.item()
        loss_path = loss_fn(hazards=result_path['hazards'], S=result_path['S'], Y=label, c=c)
        loss_path_value = loss_path.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(result['S'], dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size:'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk)))
        loss = loss / gc + loss_reg
        loss.backward()

        ratio_omic = loss_omic_value / loss_path_value
        ratio_path = 1 / ratio_omic

        tanh = nn.Tanh()
        relu = nn.ReLU(inplace=True)

        if ratio_omic > 1:
            coeff_omic = 1 - tanh(args.alpha * relu(ratio_omic))
            coeff_path = 1
        else:
            coeff_path = 1 - tanh(args.alpha * relu(ratio_path))
            coeff_omic = 1

        for name, parms in model.named_parameters():
            print('name:', name)

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

        writer.add_scalar('data/coefficient omic', coeff_omic, epoch)
        writer.add_scalar('data/coefficient path', coeff_path, epoch)


def validate_survival_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, slide_name) in enumerate(loader):
        # print('slide_name:',slide_name)  # debug slide_name
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            # hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            result, result_omic, result_path = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        # loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss = loss_fn(hazards=result['hazards'], S=result['S'], Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(result['S'], dim=1).cpu().numpy()  # 计算风险分数
        all_risk_scores[batch_idx] = risk  # 存储风险分数
        all_censorships[batch_idx] = c.cpu().numpy()  # 存储审查状态
        all_event_times[batch_idx] = event_time  # 存储事件时间
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(),
                                           'survival': event_time.item(), 'censorship': c.item()}})

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    val_epoch_str = "val c-index: {:.4f}".format(c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str + '\n')

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return patient_results, c_index, False
