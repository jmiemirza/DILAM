import os
from pathlib import Path
from threading import Thread
import torch.optim as optim
import numpy as np
import torch
from tqdm import tqdm
from utils.general import box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images
from utils.torch_utils import time_synchronized
import torch.nn as nn

from utils.testing_yolov3 import test
from utils.data_loader import get_loader, set_severity
import globals
import logging
from models.knn_classifier import KITTIWeatherKNNClassifier
from statistics import mean
from utils.results_manager import ResultsManager
from models.linear_classifier import train_cls_head

log = logging.getLogger('TRAINING')
log_fileonly = logging.getLogger('TESTING.FILEONLY')


def dilam(net, opt, rect=True, pnp_split='test', verbose=False):
    device = next(net.parameters()).device

    use_knn_classifier = False
    if opt.knn:
        opt.dilam_adapt_all = True
        use_knn_classifier = True
    else:
        if not os.path.exists(opt.cls_ckpt_path):
            train_cls_head(net, opt, rect=True, fname=opt.cls_ckpt_path)
        net.class_head.load_state_dict(torch.load(opt.cls_ckpt_path, map_location=device))

    augment = not opt.no_augment_dilam

    # Adapt using dilam. This creates the memory bank, which is used in plug & play.
    if not opt.no_dilam_adapt:
        dilam_adapt(model=net, opt=opt, adapt_all_bn_layers=opt.dilam_adapt_all, augment=augment)

    # load memory bank
    if opt.dilam_adapt_all:
        net.bn_affine = torch.load(f'{opt.checkpoints_path}/mem_bank_ALL_BN_LAYERS.pt', map_location=device)
    else:
        net.bn_affine = torch.load(f'{opt.checkpoints_path}/mem_bank.pt', map_location=device)

    # Run dilam plug & play for every task, using incremental learning evaluation protocol
    results = ResultsManager()
    tasks = ['initial'] + globals.TASKS
    for idx, opt.task in enumerate(tasks):
        log.info(f'Start evaluation for Task-{idx} ({opt.task}).')
        current_results = []
        current_results_map50to95 = []
        net.eval()

        for i in range(0, idx + 1):
            opt.task = tasks[i]
            set_severity(opt)

            # This is a bit of a hack that sets augment to false
            # for the initial task only, to evaluate the initial task with the same
            # augment settings as in the other baselines
            aug = False if opt.task == 'initial' else augment

            dataloader = get_loader(opt, split=pnp_split, pad=0.5, rect=rect)
            test_res = dilam_plug_and_play(
                batch_size=opt.batch_size,
                model=net,
                iou_thres=opt.iou_thres,
                conf_thres=opt.conf_thres,
                dataloader=dataloader,
                opt=opt,
                augment=aug,
                verbose=verbose,
                use_knn_classifier=use_knn_classifier
            )
            map50 = test_res[0][2] * 100
            map50to95 = test_res[0][3] * 100

            current_results.append(map50)
            current_results_map50to95.append(map50to95)
            log.info(f'mAP@50 on Task-{i} ({tasks[i]}): {map50:.1f}')

            if i == idx:
                mean_result = mean(current_results)
                mean_result_map50to95 = mean(current_results_map50to95)
                log.info(f'Mean mAP@50 over current task ({tasks[i]}) '
                         f'and previously seen tasks: {mean_result:.1f}')
                severity_str = '' if opt.task == 'initial' else f'{opt.severity}'
                results.add_result('DILAM', f'{tasks[i]} {severity_str}', mean_result, 'online')
                results.add_result('DILAM', f'{tasks[i]} {severity_str}', mean_result, 'offline')

                results.add_result_map50to95('DILAM', f'{tasks[idx]} {severity_str}', mean_result_map50to95, 'online')
                results.add_result_map50to95('DILAM', f'{tasks[idx]} {severity_str}', mean_result_map50to95, 'offline')



def dilam_adapt(
    single_cls=False,
    augment=False,
    verbose=True,
    model=None,
    save_dir=Path(''),  # for saving images
    save_txt=False,  # for auto-labelling
    save_hybrid=False,  # for hybrid auto-labelling
    save_conf=False,  # save auto-label confidences
    plots=True,
    wandb_logger=None,
    compute_loss=None,
    half_precision=False,
    opt=None,
    adapt_all_bn_layers=True, # don't adapt first 2 bn layers when this is False
    run_num=0,
    nc=8
):
    device = next(model.parameters()).device
    nc = 1 if single_cls else nc
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    tmp_batch_size = opt.batch_size
    opt.batch_size = opt.dilam_adapt_batch_size

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    for opt.task in globals.TASKS:
        if not set_severity(opt):
            continue

        model.load_state_dict(torch.load(opt.ckpt_path, map_location=device))
        # model.class_head.load_state_dict(torch.load(opt.cls_ckpt_path, map_location=device))

        # Loaders
        tmp_task = opt.task
        opt.task = 'initial'
        dataloader_train = get_loader(opt, split='train', pad=0.5, rect=True)
        opt.task = tmp_task

        dataloader_val = get_loader(opt, split='val', pad=0.5, rect=True)

        tmp_opt_imgsz = opt.img_size
        opt.img_size = [840, 840]
        dataloader_test = get_loader(opt, split='val', pad=0.5, rect=True)
        opt.img_size = tmp_opt_imgsz

        l1_loss = nn.L1Loss(reduction='mean')

        # Set requires_grad for bn layers to adapt
        model.requires_grad_(False)
        bn_layer_cnt = 0
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                bn_layer_cnt += 1
                if adapt_all_bn_layers or bn_layer_cnt > 2:
                    m.requires_grad_(True)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # log.info(f'Req grad param cnt: {count_parameters(model)}')
        # opt.verbose = True

        log.info(f'Run {run_num} - Task {opt.task}')

        chosen_bn_layers = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                chosen_bn_layers.append(m)
        if not adapt_all_bn_layers:
            chosen_bn_layers = chosen_bn_layers[2:]

        n_chosen_layers = len(chosen_bn_layers)
        save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]
        clean_mean_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
        clean_var_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
        clean_mean_list_final = []
        clean_var_list_final = []

        with torch.no_grad():
            for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_train)):
                img = img.to(device, non_blocking=True)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                model.eval()
                hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]
                _ = model(img)

                for i in range(n_chosen_layers):
                    clean_mean_act_list[i].update(get_clean_out(save_outputs[i]))  # compute mean from clean data
                    clean_var_act_list[i].update(get_clean_out_var(save_outputs[i]))  # compute variane from clean data
                    save_outputs[i].clear()
                    hook_list[i].remove()

            for i in range(n_chosen_layers):
                clean_mean_list_final.append(clean_mean_act_list[i].avg)  # [C, H, W]
                clean_var_list_final.append(clean_var_act_list[i].avg)  # [C, H, W]

        para_to_opt = list()

        bn_param_cnt = 0
        for name, param in model.named_parameters():
            if 'bn' in name and not 'class_head' in name:
                # when adapting all bn layers, add all their params.
                # when not adapting first 2 bn layers ignore first 4 bn params
                bn_param_cnt += 1
                if adapt_all_bn_layers or bn_param_cnt > 4:
                    para_to_opt.append(param)

        log.info(f'###########Param to Optimize###########: {len(para_to_opt)}')
        optimizer = optim.SGD(para_to_opt, lr=opt.lr, momentum=0.9, weight_decay=5e-4)

        log.info('ADAPTATING BATCH NORMS...')
        ap_epochs = list()

        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_val)):
            model.eval()
            optimizer.zero_grad()
            save_outputs_tta = [SaveOutput() for _ in range(n_chosen_layers)]

            hook_list_tta = [chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
                                for x in range(n_chosen_layers)]
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = img.to(device)
            _ = model(img)
            batch_mean_tta = [get_out(save_outputs_tta[x]) for x in range(n_chosen_layers)]
            batch_var_tta = [get_out_var(save_outputs_tta[x]) for x in range(n_chosen_layers)]

            loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(device)
            loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(device)

            for i in range(n_chosen_layers):
                loss_mean += l1_loss(batch_mean_tta[i].to(device), clean_mean_list_final[i].to(device))
                loss_var += l1_loss(batch_var_tta[i].to(device), clean_var_list_final[i].to(device))

            loss = loss_mean + loss_var

            loss.backward()
            optimizer.step()
            for z in range(n_chosen_layers):
                save_outputs_tta[z].clear()
                hook_list_tta[z].remove()

        # saving gamma(weight) and beta(bias) in a dict in model
        state_dict = model.state_dict()
        model.bn_affine[opt.task] = {}
        bn2d_idx = 0
        for layer_name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if adapt_all_bn_layers or bn2d_idx >= 2:
                    model.bn_affine[opt.task][layer_name] = {
                        'weight': state_dict[layer_name + '.weight'].detach().clone(),
                        'bias': state_dict[layer_name + '.bias'].detach().clone()
                    }
                bn2d_idx += 1

        # save model adapted to task opt.task
        torch.save(model.state_dict(), f'{opt.checkpoints_path}/dilam_adapted_{opt.task}.pt')

        ap = test(batch_size=tmp_batch_size,
                  augment=augment,
                  dataloader=dataloader_test,
                  model=model,
                  multi_label=True)[-1]

        Path(f'results_kitti_affine/{opt.task}/all/{run_num}').mkdir(parents=True, exist_ok=True)
        np.save(f'results_kitti_affine/{opt.task}/all/{run_num}/{opt.task}.npy', ap)

        # compare_net_state_dict(model, opt)

    save_path = f'{opt.checkpoints_path}/mem_bank'
    save_path += '_ALL_BN_LAYERS.pt' if adapt_all_bn_layers else '.pt'
    torch.save(model.bn_affine, save_path)

    opt.batch_size = tmp_batch_size


def compare_net_state_dict(net, args):
    """
        Sanity check function, printing names of changed layers in net,
        compared to clear checkpoint
    """
    device = next(net.parameters()).device
    net_sd = net.state_dict()
    sd = torch.load(args.ckpt_path, map_location=device)

    for (net_k, net_v), (loaded_k, loaded_v) in zip(net_sd.items(), sd.items()):
        if net_k != loaded_k:
            log.info(f'{net_k} != {loaded_k}')

        if not torch.allclose(net_v, loaded_v.to(torch.device('cuda:0'))):
            log.info(f'{net_k} mismatch')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []


def get_out(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.mean(out, dim=0)
    return out


def get_clean_out(output_holder):
    out = torch.vstack(output_holder.outputs)

    out = torch.mean(out, dim=0)
    return out


def get_out_var(output_holder):
    out = torch.vstack(output_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def get_clean_out_var(out_holder):
    out = torch.vstack(out_holder.outputs)
    out = torch.var(out, dim=0)
    return out


def take_mean(input_ten):
    input_ten = torch.mean(input_ten, dim=0)
    return input_ten


def init_knn_classifier(args, train_split='train'):
    knn = KITTIWeatherKNNClassifier()

    hists_path = f'{args.checkpoints_path}/knn_{train_split}_hists.npy'
    labels_path = f'{args.checkpoints_path}/knn_{train_split}_labels.npy'

    if os.path.exists(hists_path) and os.path.exists(labels_path):
        train_ds = np.load(hists_path)
        train_labels = np.load(labels_path)
        knn.fit(train_ds, train_labels)
    else:
        knn.create_train_dataset_and_fit(args)

    return knn


@torch.no_grad()
def dilam_plug_and_play(
    model=None,
    opt=None,
    half_precision=False,
    batch_size=32,
    imgsz=1216,
    conf_thres=0.001,
    iou_thres=0.6,  # for NMS
    single_cls=False,
    augment=False,
    verbose=True,
    dataloader=None,
    save_dir=Path('images'),  # for saving images
    save_txt=False,  # for auto-labelling
    save_hybrid=False,  # for hybrid auto-labelling
    save_conf=False,  # save auto-label confidences
    plots=False,
    compute_loss=None,
    use_knn_classifier=True
):
    if use_knn_classifier:
        knn = init_knn_classifier(opt)

    training = model
    device = next(model.parameters()).device
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    nc = 8
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img_1 = img.to(device, non_blocking=True)
        img_1 = img.half() if half else img.float()  # uint8 to fp16/32
        img_1 /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time_synchronized()
        model.eval()
        img_1 = img_1.to(device)

        # use knn to predict task and plug in according affine params
        if use_knn_classifier:
            pred_out = knn.predict_batch(img, verbose=False)
            predicted_task = pred_out[0]
            true_task = globals.KITTI_CLS_WEATHER.index(opt.task)
            if true_task != predicted_task and verbose:
                log.info(f'KNN CLS MISS: pred: {predicted_task}, true: {true_task}')
                log.info(f'Predictions in batch: {pred_out[1]}')
            if predicted_task == 0:
                model.load_state_dict(torch.load(opt.ckpt_path, map_location=device))
            else:
                tasks = globals.KITTI_CLS_WEATHER
                state_dict = model.state_dict()
                for layer_name, val in model.bn_affine[tasks[predicted_task]].items():
                    state_dict[layer_name + '.weight'] = val['weight']
                    state_dict[layer_name + '.bias'] = val['bias']
                model.load_state_dict(state_dict)
            model.eval()
            out, train_out = model(img_1, augment=augment)
        # predict task with linear classifier, plugs in affine params in forward
        elif augment:
            if opt.dilam_adapt_all:
                out, train_out = model.double_forward_augment_cls_affine(img_1, opt.ckpt_path, opt.cls_ckpt_path, verbose=verbose, task=opt.task)
            else:
                out, train_out = model.forward_augment_cls_affine(img_1, opt.ckpt_path, opt.cls_ckpt_path, verbose=verbose)
        else:
            if opt.dilam_adapt_all:
                out, train_out = model.double_forward_cls_affine(img_1, opt.ckpt_path, opt.cls_ckpt_path, verbose=verbose, task=opt.task)
            else:
                out, train_out = model.forward_cls_affine(img_1, opt.ckpt_path, opt.cls_ckpt_path, verbose=verbose)

        t0 += time_synchronized() - t

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # if plots and batch_i < 3:
        if plots:
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            # print(image_id)
            f = save_dir / f'{str(batch_i).zfill(6)}.png'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            # f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            # Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    # print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    log_fileonly.info(s)
    log.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # print(ap50)

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            log.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        log.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))


    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, ap50



def create_bn_affine_ckpt_from_provided_model_ckpts(net, args, all_bn_layers=False):
    """
        Auxiliary function which can be used to generate memory bank from
        checkpoints for each task by saving batch norm weights and biases
        of all batch norm layers (except the first 2 if all_bn_layers set to False).
        Might need some modification, depending on the checkpoints used.
    """
    device = next(net.parameters()).device
    for args.task in globals.TASKS:
        print('TASK ', args.task)

        sd = torch.load(f'{args.task}.pth', map_location=device)['net']
        for k in list(sd.keys()):
            sd[k[6:]] = sd.pop(k) # slice away leading 'model.' in keys
        set_severity(args)
        net.model.load_state_dict(sd)

        state_dict = net.state_dict()
        net.bn_affine[args.task] = {}
        bn2d_idx = 0
        for layer_name, m in net.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if all_bn_layers or bn2d_idx >= 2:
                    net.bn_affine[args.task][layer_name] = {
                        'weight': state_dict[layer_name + '.weight'].detach().clone(),
                        'bias': state_dict[layer_name + '.bias'].detach().clone()
                    }
                bn2d_idx += 1

    fname = 'mem_bank'
    fname += '_ALL_BN_LAYERS.pt' if all_bn_layers else '.pt'
    torch.save(net.bn_affine, fname)


