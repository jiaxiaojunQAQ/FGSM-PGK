import argparse
import copy
import logging
import os
import time

import numpy as np
import torch

from ImageNet_models import *
# from preact_resnet import PreActResNet18
from utils02 import *

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='tiny_imagenet/tiny-imagenet-200', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--epochs_reset', default=40, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--factor', default=0.8, type=float, help='Label Smoothing')
    parser.add_argument('--lamda', default=1, type=float, help='Label Smoothing')
    parser.add_argument('--momentum_decay', default=0.1, type=float, help='momentum_decay')
    parser.add_argument('--EMA_value', default=0.9, type=float)
    return parser.parse_args()
from torch.nn import functional as F
def _label_smoothing(label, factor):
    one_hot = np.eye(200)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(200 - 1))

    return result

from torch.autograd import Variable

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss
upper_limit_y = 1
lower_limit_y = 0

class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }







def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, 'FGSM_PGK_TinyImageNet')

    output_path = os.path.join(output_path, 'momentum_decay_' + str(args.momentum_decay))
    output_path = os.path.join(output_path, 'lamda_' + str(args.lamda))
    output_path = os.path.join(output_path, 'EMA_value_' + str(args.EMA_value))




    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

   # train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    train_loader, test_loader = New_ImageNet_get_all_loaders_64(args.data_dir,args.batch_size)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    # model = PreActResNet18().cuda()
    # model.train()

    #print(cifar_x.shape)
    print('==> Building model..')
    logger.info('==> Building model..')
    if args.model == "VGG":
        model = VGG('VGG19')
    elif args.model == "ResNet18":
        model = ResNet18()
    elif args.model == "PreActResNest18":
        model = PreActResNet18()
    elif args.model == "WideResNet":
        model = WideResNet()
    # model=torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()
    teacher_model = EMA(model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    num_of_example = 100000
    batch_size = args.batch_size

    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)




    lr_steps = args.epochs * iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * 100/110, lr_steps * 105 / 110],
                                                         gamma=0.1)

    # Training
    prev_robust_acc = 0.
    # start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    epoch_attack_list=[]
    # global delta_list
    delta_list = []
    x_list=[]
    y_list=[]
    for i, (X, y) in enumerate(train_loader):
        cifar_x, cifar_y = X.cuda(), y.cuda()
    print(cifar_x.shape)
    import random



    for epoch in range(args.epochs):

        batch_size = args.batch_size
        cur_order = np.random.permutation(num_of_example)
        print(cur_order)
        iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
        batch_idx = -batch_size
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        attack_acc=0
        attack_count=0
        teacher_model.model.eval()
        print("epoch:,",epoch)
        print(iter_num)
        if epoch %args.epochs_reset== 0:
            temp=torch.rand(100000,3,64,64)
            if args.delta_init != 'previous':
                all_delta = torch.zeros_like(temp).cuda()
                all_momentum=torch.zeros_like(temp).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    all_delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                #all_delta.data = clamp(all_delta, lower_limit - cifar_x, upper_limit - cifar_x)
                #all_delta.requires_grad = True
                all_delta.data = clamp(alpha * torch.sign(all_delta), -epsilon, epsilon)
                #all_delta.data[:cifar_x.size(0)] = clamp(all_delta[:cifar_x.size(0)], lower_limit - cifar_x, upper_limit - cifar_x)
        print(all_delta[1])
        idx = torch.randperm(cifar_x.shape[0])
        print(idx)


        cifar_x =cifar_x[idx, :,:,:].view(cifar_x.size())

        cifar_y = cifar_y[idx].view(cifar_y.size())
        all_delta=all_delta[idx, :, :, :].view(all_delta.size())
        all_momentum=all_momentum[idx, :, :, :].view(all_delta.size())
        print(cifar_x.shape)
        print(cifar_y.shape)
        print(all_delta.shape)
        for i in range(iter_num):

            batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
            X=cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            y= cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            delta =all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            next_delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()

            momentum=all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            X=X.cuda()
            y=y.cuda()
            batch_size = X.shape[0]


            print(X.shape)


            # if i == 0:
            #     images = make_grid(X, 3, 0)
            #     save_image(images, 'epoch'+str(epoch)+'.jpg')

            label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor)).cuda()).float()
            # delta.requires_grad = True
            # delta.data = clamp(alpha * torch.sign(delta), -epsilon, epsilon)
            # delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            #
            # delta_y = torch.zeros_like(label_smoothing).cuda()

            delta.requires_grad = True
            # delta_y.requires_grad = True
            #delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            ori_output = model(X + delta[:X.size(0)])
            clean_acc = (ori_output.max(1)[1] == y).sum().item()
            ori_loss = LabelSmoothLoss(ori_output, label_smoothing.float())


            decay=args.momentum_decay
            # with amp.scale_loss(loss, opt) as scaled_loss:
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()
            # y_grad = delta_y.grad.detach()
            adv_delta=delta.detach().clone()

            adv_delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            adv_delta.data[:X.size(0)] = clamp(adv_delta[:X.size(0)], lower_limit - X, upper_limit - X)

            # delta_y.data = clamp(delta_y + torch.tensor(args.epsilon_y) * torch.sign(y_grad),
            #                      -torch.tensor(args.epsilon_y),
            #                       torch.tensor(args.epsilon_y))
            # delta_y.data = clamp(delta_y, lower_limit_y - label_smoothing, upper_limit_y - label_smoothing)


            adv_delta = adv_delta.detach()

            # delta_y=delta_y.detach()
            output = model(X + adv_delta[:X.size(0)])
            adv_acc = (output.max(1)[1] == y).sum().item()
            grad_norm = torch.norm(x_grad, p=1)
            attack_value = 2- (adv_acc / (clean_acc+1))
            momentum = (x_grad / grad_norm) * attack_value + momentum * decay
            # if adv_acc / clean_acc < args.attack_value:
            #     momentum = x_grad / grad_norm + momentum * decay
            # else:
            #     momentum=momentum
            #     attack_count=attack_count+1

            next_delta.data = clamp(delta + alpha * torch.sign(momentum), -epsilon, epsilon)
            next_delta.data[:X.size(0)] = clamp(next_delta[:X.size(0)], lower_limit - X, upper_limit - X)

            # print(label_smoothing[0])
            # print(adv_label[0])
            # adv_label = F.normalize((label_smoothing + delta_y), p=1, dim=-1)
            loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            loss = LabelSmoothLoss(output, (label_smoothing).float())+args.lamda*loss_fn(output.float(), ori_output.float())
            opt.zero_grad()
            # with amp.scale_loss(loss, opt) as scaled_loss:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            print(loss)
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
            # if adv_acc / (clean_acc+1)< args.EMA_value:
            #     teacher_model.update_params(model)
            #     teacher_model.apply_shadow()
            if adv_acc / (clean_acc+1) < args.EMA_value:
                teacher_model.alpha=0.999
                teacher_model.alpha

            else:
                weight = (adv_acc / (clean_acc+1)) / args.EMA_value
                teacher_model.alpha=0.99999 *weight
                if teacher_model.alpha>1:
                    teacher_model.alpha=1
            teacher_model.update_params(model)
            teacher_model.apply_shadow()



            all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = momentum



            all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]=next_delta
            # print(all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].equal(
            #      delta))
            #print(delta)
            # images = make_grid(255*delta, 3, 0)
            # save_image(images, 'test.jpg')
        #print(delta_list[1])

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        epoch_attack_list.append(train_acc / train_n)
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)

        logger.info('==> Building model..')
        if args.model == "VGG":
            model_test = VGG('VGG19').cuda()
        elif args.model == "ResNet18":
            model_test = ResNet18().cuda()
        elif args.model == "PreActResNest18":
            model_test = PreActResNet18().cuda()
        elif args.model == "WideResNet":
            model_test = WideResNet().cuda()
        # model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(teacher_model.model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(model_test.state_dict(), os.path.join(output_path, 'best_model.pth'))

    torch.save(model_test.state_dict(), os.path.join(output_path, 'final_model.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    logger.info(epoch_attack_list)
    print(epoch_clean_list)
    print(epoch_pgd_list)
    print(epoch_attack_list)
    print('attack_count',attack_count)
if __name__ == "__main__":
    main()
