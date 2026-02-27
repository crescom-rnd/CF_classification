# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
import torch
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from scipy.spatial import distance

torch.autograd.set_detect_anomaly(True)




if __name__ == "__main__":
    from ..core.evaluate import accuracy
    from ..core.inference import get_final_preds
    from ..utils.transforms import flip_back
    from ..utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def train_ori(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred, = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, trainmode, calculate_coordination_mode = False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    ngcount = 0
    okcount = 0
    totalcount = 0
    distances = 0

    # switch to train mode

    if trainmode == True:
        model.train()  
    else:
        model.eval() 

    end = time.time()

    if trainmode:

        for i, (input, target, target_weight, meta) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            if isinstance(outputs, list):
                loss = criterion(outputs[0], target, target_weight)
                for output in outputs[1:]:
                    loss += criterion(output, target, target_weight)
            else:
                output = outputs
                loss = criterion(output, target, target_weight)

            # loss = criterion(output, target, target_weight)

            # compute gradient and do update step
            if trainmode ==True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            _, avg_acc, cnt, pred,target_coord, pred_coord, dists = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=input.size(0)/batch_time.val,
                        data_time=data_time, loss=losses, acc=acc)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                save_debug_images(config, input, meta, target, pred*4, output,
                                prefix)
    else:
        with torch.no_grad():

            for i, (input, target, target_weight, meta) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                outputs = model(input)

                target = target.cuda(non_blocking=True)



                target_weight = target_weight.cuda(non_blocking=True)

                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += criterion(output, target, target_weight)
                else:
                    output = outputs
                    loss = criterion(output, target, target_weight)

                #

                # loss = criterion(output, target, target_weight)

                # compute gradient and do update step
                if trainmode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

                _, avg_acc, cnt, pred, target_coord, pred_coord, dists = accuracy(output.detach().cpu().numpy(),
                                                target.detach().cpu().numpy())
                if calculate_coordination_mode:
                    ngcount_per_batch, okcount_per_batch, totalcount_per_batch = calculate_coordinate(pred_coord, target_coord)
                    ngcount += ngcount_per_batch
                    okcount += okcount_per_batch
                    totalcount += totalcount_per_batch
                    for dist in dists:
                        distances += sum(dist)



                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                        'Speed {speed:.1f} samples/s\t' \
                        'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            speed=input.size(0)/batch_time.val,
                            data_time=data_time, loss=losses, acc=acc)
                    logger.info(msg)

                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']
                    writer.add_scalar('train_loss', losses.val, global_steps)
                    writer.add_scalar('train_acc', acc.val, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

                    prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                    save_debug_images(config, input, meta, target, pred*4, output,
                                    prefix)

    return loss, avg_acc, target, outputs, acc.val, ngcount, okcount, totalcount, distances


import torch.nn.functional as F

def calculate_coordinate(pred_batch, target_batch):
    #
    ngcount_per_batch = 0
    okcount_per_batch = 0
    totalcount_per_batch = 0

    for pred_image, target_image in zip(pred_batch, target_batch):
        far1, far2 = find_farthest_points(target_image)
        threshold_dist = distance.euclidean(far1, far2)

        for num, (pt, gt) in enumerate(zip(pred_image, target_image)):
            if int(gt[0]) == 0 and int(gt[1]) == 0:
                continue
            dist = distance.euclidean(pt, gt)
            threshold = 10
            if dist > threshold_dist / threshold:
                ngcount_per_batch += 1
            else:
                okcount_per_batch +=1
            totalcount_per_batch +=1
        # 좌표 구하기
        # for output_label, target_label in zip(output_image, target_image):
        #     _, row_output = torch.max(output_label, dim =1)
        #     _, col_output = torch.max(row_output, dim = 0)
        #     output_x_index, output_y_index = col_output.item(), row_output[col_output].item()
        #
        #
        #
        #     _, row_target = torch.max(target_label, dim =1)
        #     _, col_target = torch.max(row_target, dim = 0)
        #     target_x_index, target_y_index = col_target.item(), row_target[col_target].item()
        #
        #     # if target_x_index == 0 and target_y_index == 0:
        #     #     print(row_target, col_target)
        #     # print(output_x_index, output_y_index)
        #     output_coord_per_image.append((output_x_index, output_y_index))
        #     target_coord_per_image.append((target_x_index, target_y_index))
        # far1, far2 = find_farthest_points(target_coord_per_image)
        # threshold_dist = distance.euclidean(far1, far2)
        # for num, (pt, gt) in enumerate(zip(output_coord_per_image, target_coord_per_image)):
        #     if (int(gt[0]) == 0) and (int(gt[1]) == 0):
        #         continue
        #     dist = distance.euclidean(pt, gt)
        #     threshold = 3
        #     if dist > threshold_dist / threshold:
        #         ngcount_per_batch +=1
        #     else:
        #         okcount_per_batch +=1
        #     totalcount_per_batch +=1
    return ngcount_per_batch, okcount_per_batch, totalcount_per_batch



def find_farthest_points(coordinates: list):
    # 모든 좌표 쌍 간의 거리 계산
    distances = cdist(coordinates, coordinates, metric='euclidean')

    # 가장 먼 거리와 해당 좌표 쌍 인덱스 찾기
    i, j = np.unravel_index(np.argmax(distances), distances.shape)

    return coordinates[i], coordinates[j]



def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Clip the perturbed image to be within the valid data range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image



from torchattacks import FGSM

def generate_adversarial_examples(model, eps, x, y):
    attacker = FGSM(model, eps)
    x_adv = attacker(x, y)
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv


def train_adversarial(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, trainmode):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode

    if trainmode == True:
        model.train()  
    else:
        model.eval() 

    end = time.time()


    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)
            
        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        if trainmode ==True:
            epsilon = 0.07
            adv_input = fgsm_attack(input, epsilon, input)
            adv_output = model(adv_input)
            adv_loss = criterion(adv_output, target, target_weight)
            total_loss = loss + adv_loss


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # print(input.sign(), 'gmd??')

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred, target_coord, pred_coord = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)
            

    
            

    return loss, avg_acc, target, outputs, acc.val


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, _, _, pred, _ = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images
            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)
                return_acc = acc.val

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator, return_acc


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


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
        self.avg = self.sum / self.count if self.count != 0 else 0
