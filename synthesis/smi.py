import time
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from tqdm import tqdm
from ._utils import UnlabeledImageDataset, DataIter, ImagePool
from .base import BaseSynthesis
from .hooks import DeepInversionHook
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch

import matplotlib.pyplot as plt
import numpy as np
import sns

def calculate_adjusted_logits(logits, targets, c_func, lambda_param):
    """
    计算调整后的 logits

    参数:
    logits: 模型的输出 [batch_size, num_classes]
    targets: 真实标签 [batch_size]
    c_func: 调整项函数，接受两个类别索引作为输入并返回一个标量
    lambda_param: 调节参数 λ

    返回:
    调整后的 logits
    """
    batch_size, num_classes = logits.size()
    
    # 获取目标类的 logits
    target_logits = logits[range(batch_size), targets]
    
    # 初始化调整项
    c_values = torch.zeros_like(logits)
    
    # 计算调整项 c(y, k)
    for i in range(batch_size):
        for k in range(num_classes):
            c_values[i, k] = c_func(targets[i].item(), k)
    
    # 计算调整后的 logits
    adjusted_logits = (logits + c_values - target_logits.unsqueeze(1)) / lambda_param
    
    return adjusted_logits

# 示例调整项函数 c(y, k)
def example_c_func(y, k):
    # 这是一个简单的示例函数，可以根据需要进行调整
    return 0 if y != k else 0.0

def optimized_c_func(y, num_classes):
    # 创建一个 num_classes x num_classes 的矩阵，其中对角线为0，其余为1
    c_matrix = torch.ones(num_classes, num_classes) - torch.eye(num_classes)
    return c_matrix[y]

# 自定义损失函数
def custom_cross_entropy_loss(logits, targets, c_func, lambda_param = 1):
    """
    自定义交叉熵损失函数

    参数:
    logits: 模型的输出 [batch_size, num_classes]
    targets: 真实标签 [batch_size]
    c_func: 调整项函数，接受两个类别索引作为输入并返回一个标量
    lambda_param: 调节参数 λ

    返回:
    自定义损失值
    """
    # 计算调整后的 logits
    adjusted_logits = calculate_adjusted_logits(logits, targets, c_func, lambda_param)
    
    # 使用 F.cross_entropy 计算损失
    loss = lambda_param * F.cross_entropy(adjusted_logits, targets)
    
    return loss

def calculate_c_yk(targets, num_classes):
    """
    计算调整项 c(y, k) 矩阵，利用矢量化和广播机制

    参数:
    targets: 真实标签 [batch_size]
    num_classes: 类别总数

    返回:
    调整项 c(y, k) 矩阵 [batch_size, num_classes]
    """
    batch_size = targets.size(0)

    # 创建一个目标标签矩阵，形状为 [batch_size, num_classes]
    targets_matrix = targets.unsqueeze(1).repeat(1, num_classes)
    
    # 创建一个类别索引矩阵，形状为 [batch_size, num_classes]
    class_indices = torch.arange(num_classes, device=targets.device).unsqueeze(0).repeat(batch_size, 1)
    
    # 计算 c(y, k) 矩阵，使用矢量化操作
    c_yk = torch.where(targets_matrix == class_indices, 0.0, -10.0)
    
    return c_yk

def ldr_loss(logits, targets, c):
    """
    计算LDR损失
    :param logits: 模型的输出 (logits)，形状为 (batch_size, num_classes)
    :param targets: 真实的标签 (整数，表示类别索引)，形状为 (batch_size,)
    :param c: 惩罚函数，函数 c(y, k) 返回一个标量
    :return: LDR损失
    """
    r = 60
    logits = logits.to('cuda')
    targets = targets.to('cuda')
    batch_size, num_classes = logits.shape

    # 计算 f_y
    f_y = logits[torch.arange(batch_size, device='cuda'), targets]  # 形状为 (batch_size,)

    # 使用广播机制计算 c(y, k) - f_y 部分
    # c_yk = torch.stack([torch.tensor([c(y.item(), k) for k in range(num_classes)]) for y in targets.cpu()]).to('cuda') 
    # c_yk = torch.stack([optimized_c_func(y.item(), num_classes) for y in targets]).to('cuda')
    c_yk = calculate_c_yk(targets, num_classes)
    # c_yk = 0
    logits_adjusted = (logits + c_yk - f_y.unsqueeze(1)) / r

    # 计算 exp(logits_adjusted) 并求和
    exp_logits_adjusted = torch.exp(logits_adjusted)
    sum_exp = torch.sum(exp_logits_adjusted, axis=1) * r

    # 计算最终的LDR损失
    ldr_loss = torch.mean(torch.log(sum_exp))

    # del f_y, exp_logits_adjusted, sum_exp
    return ldr_loss


# 惩罚函数 c(y, k)
def penalty_function(y, k):
    if y == k:
        return 0
    else:
        return -10  # 如果标签不同，惩罚为1

def cross_entropy_loss(y_true, y_pred):
    """
    计算交叉熵损失

    Parameters:
    y_true (np.ndarray): 实际标签的 one-hot 编码数组，形状为 (N, C)
    y_pred (np.ndarray): 模型输出的概率分布数组，形状为 (N, C)

    Returns:
    float: 交叉熵损失
    """
    # 防止取对数时出现零
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
    # 计算交叉熵损失
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{:.2f}s'.format(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_top_k_relative_indices_including_first(pre_attention, K):##返回每个批次前 K 个最大注意力权重的相对索引的张量,第0维拼接一个0
    batch_size, N = pre_attention.shape
    K = min(K, N)
    remaining_attention = pre_attention
    top_values, top_indices = torch.topk(remaining_attention, K, dim=1)
    top_indices_adjusted = top_indices + 1
    first_index = torch.zeros((batch_size, 1), dtype=torch.long, device=pre_attention.device)##初始化0
    top_k_indices = torch.cat((first_index, top_indices_adjusted), dim=1)##拼接索引
    return top_k_indices

def clip_images(image_tensor, mean, std):##图像归一化裁剪，使之满足要求
    mean = np.array(mean)
    std = np.array(std)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def get_image_prior_losses(inputs_jit):##计算总方差损失
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1,loss_var_l2

def jsdiv( logits, targets, T=1.0, reduction='batchmean' ):
    P = F.softmax(logits / T, dim=1)
    Q = F.softmax(targets / T, dim=1)
    M = 0.5 * (P + Q)
    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    return 0.5 * F.kl_div(torch.log(P), M, reduction=reduction) + 0.5 * F.kl_div(torch.log(Q), M, reduction=reduction)

def jitter_and_flip(inputs_jit, lim=1./8., do_flip=True):##图像随即抖动和翻转
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)
    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
    # Flipping
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit,off1,off2,flip and do_flip

def jitter_and_flip_index(pre_index_matrix, off1, off2, flip, patch_size=16, num_patches_per_dim=14):
    off1_int, off1_frac = int(off1 // patch_size), off1 % patch_size / patch_size
    off2_int, off2_frac = int(off2 // patch_size), off2 % patch_size / patch_size
    patch_indices = torch.arange(1, num_patches_per_dim * num_patches_per_dim + 1).reshape(num_patches_per_dim, num_patches_per_dim).to(pre_index_matrix.device)
    patch_indices = torch.roll(patch_indices, shifts=(off1_int, off2_int), dims=(0, 1))
    if abs(off1_frac) >= 0.5:
        direction = 1 if off1_frac > 0 else -1
        patch_indices = torch.roll(patch_indices, shifts=(direction, 0), dims=(0, 1))
    if abs(off2_frac) >= 0.5:
        direction = 1 if off2_frac > 0 else -1
        patch_indices = torch.roll(patch_indices, shifts=(0, direction), dims=(0, 1))
    if flip:
        patch_indices = torch.flip(patch_indices, dims=[1])
    flat_patch_indices = patch_indices.flatten()
    non_zero_mask = pre_index_matrix != 0
    indices = (flat_patch_indices == pre_index_matrix[non_zero_mask].unsqueeze(-1)).nonzero(as_tuple=True)
    rows = indices[1] // num_patches_per_dim
    cols = indices[1] % num_patches_per_dim
    new_indices = rows * num_patches_per_dim + cols + 1
    new_index_matrix = torch.zeros_like(pre_index_matrix)
    new_index_matrix[non_zero_mask] = new_indices
    return new_index_matrix

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

# 自定义 FLOPs 计算逻辑
def add_flop_jit(inputs, outputs):
    # inputs[0] 是第一个输入张量，其他参数可以从 inputs 获取
    return inputs[0].numel()

def mul_flop_jit(inputs, outputs):
    return inputs[0].numel()

def softmax_flop_jit(inputs, outputs):
    # Softmax 近似 FLOPs = num_elements * 5
    return inputs[0].numel() * 5

def gelu_flop_jit(inputs, outputs):
    # GELU 近似 FLOPs = num_elements * 7
    return inputs[0].numel() * 7

def visualize_full_attention(attention_weights):

    """
    可视化完整的注意力权重热力图。
    
    Args:
        attention_weights (torch.Tensor): 注意力权重张量, 形状 (B, heads, N, N)
    """
    # Step 1: 对头维度进行平均，保留序列长度的注意力矩阵
    # 假设 attention_weights.shape = (B, heads, N, N)
    attention_weights = torch.mean(attention_weights[-1], dim=1)[:, 0, :][:, 1:]

    # Step 3: 选择一个批次 (这里选第一个) 并转为 NumPy
    full_attention_map = attention_weights[0].detach().cpu().numpy()  # (N-1, N-1)

    # Step 4: 绘制热力图
    plt.figure(figsize=(8, 8))
    sns.heatmap(full_attention_map, cmap="viridis", square=True, cbar=True)
    plt.title("Full Attention Heatmap (Last Layer)")
    plt.xlabel("Patch Index")
    plt.ylabel("Patch Index")
    plt.show()

# 自定义 FLOPs 分析器
class CustomFlopCountAnalysis(FlopCountAnalysis):
    def __init__(self, model, inputs):
        super().__init__(model, inputs)
        self.custom_handles = {
            "aten::add": add_flop_jit,
            "aten::mul": mul_flop_jit,
            "aten::softmax": softmax_flop_jit,
            "aten::gelu": gelu_flop_jit,
        }

    def dispatch(self, op: str, inputs, outputs):
        if op in self.custom_handles:
            return self.custom_handles[op](inputs, outputs)
        # return super().dispatch(op, inputs, outputs)

class SMI(BaseSynthesis):
    def __init__(self, teacher,teacher_name, student, num_classes, img_shape=(3, 224, 224),patch_size=16,
                 iterations=2000, lr_g=0.25,
                 synthesis_batch_size=128, sample_batch_size=128, start=0,
                 adv=0.0, bn=0, oh=1,tv1=0.0, tv2=1e-5, l2=0.0, ldr=1, ratio=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ,
                 save_dir='', transform=None,
                 normalizer=None, device='cpu',
                 bnsource='resnet50v2',init_dataset=None):
        super(SMI, self).__init__(teacher, student)
        assert len(img_shape)==3, "image size should be a 3-dimension tuple"

        self.save_dir = save_dir
        self.img_size = img_shape
        self.patch_size=patch_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.ldr = ldr
        self.ratio = ratio
        self.start = start
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.data_iter = None
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset=init_dataset

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.bn = bn
        if self.bn != 0:
            if bnsource == 'resnet50v2':
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).cuda(
                    device)
                print(count_parameters(self.prior),'resnet50v2')
            elif bnsource == 'resnet50v1':
                self.prior = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda(
                    device)
                print(count_parameters(self.prior),'resnet50v1')
            else:
                raise NotImplementedError
            self.prior.eval()
            self.prior.cuda()
        # Scaling factors
        self.adv = adv
        self.oh = oh
        self.tv1 = tv1
        self.tv2 = tv2
        self.l2 = l2
        self.num_classes = num_classes

        # training configs
        self.device = device

        # setup hooks for BN regularization
        if self.bn!=0:
            self.bn_hooks = []
            for m in self.prior.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.bn_hooks.append( DeepInversionHook(m) )
            assert len(self.bn_hooks)>0, 'input model should contains at least one BN layer for DeepInversion'

    def synthesize(self, targets=None,num_patches=197,prune_it=[-1],prune_ratio=[0],start=0):
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        inputs = torch.randn( size=[self.synthesis_batch_size, *self.img_size], device=self.device ).requires_grad_()
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            # targets = torch.randint(low=start, high=start+1, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])

        best_inputs = inputs.data

        current_abs_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        next_relative_index = torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device)
        inputs_aug = inputs##保存原始图像，之后使用inputs_aug替代
        kl_loss = 0
        ldr=False
        top_K = num_patches
        for it in tqdm(range(self.iterations)):
            r = self.ratio
            if it+1 in prune_it:##剪枝前一步不进行随即抖动和翻转操作
                inputs_aug = inputs
                current_abs_index_aug = current_abs_index
                # with torch.no_grad():
                t_out, attention_weights, _ = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index,r)
                # memory_stats = torch.cuda.memory_stats()
                # print(memory_stats)
            elif it in prune_it:##执行剪枝操作
                ##提取最后一层的权重，对头维度进行平均，并丢弃第一个元素
                #变量包含了除了 [CLS] 标记之外的所有位置的注意力权重
                attention_weights = torch.mean(attention_weights[-1], dim=1)[:, 0, :][:, 1:]  # (B,heads,N,N)->(B,p-1)
                prune_ratio_value = prune_ratio[prune_it.index(it)]
                top_K=int(attention_weights.shape[1] * (1.0 - prune_ratio_value))##137
                print('top_K:',top_K,'###',it)
                next_relative_index = get_top_k_relative_indices_including_first(pre_attention=attention_weights, K=top_K).to(self.device)##返回每个批次前 K 个最大注意力权重的相对索引的张量
                inputs_aug = (inputs)
                current_abs_index_aug = current_abs_index
                # with torch.no_grad():
                t_out, attention_weights, current_abs_index = self.teacher(inputs_aug, current_abs_index_aug,next_relative_index,r)
                print(count_parameters(self.teacher),"teacher")
                # memory_stats = torch.cuda.memory_stats()
                # print(memory_stats)
            else:
                inputs_aug,off1,off2,flip = jitter_and_flip(inputs)
                if current_abs_index.shape[1]==num_patches:##patch_size==16,current_abs_index不变
                    current_abs_index_aug = current_abs_index
                else:
                    current_abs_index_aug =jitter_and_flip_index(current_abs_index,off1,off2,flip,self.patch_size,int(224//self.patch_size))
                # del off1,off2,flip
                # print("uioaefwgwi")
                t_out,attention_weights,_ = self.teacher(inputs_aug,current_abs_index_aug,next_relative_index,r)
                # with torch.no_grad():
                #     t_out,attention_weights,_ = self.teacher(inputs_aug,current_abs_index_aug,next_relative_index,r)

            loss_bn=0
            loss_oh = F.cross_entropy( t_out, targets)
            loss_tv1,loss_tv2 = get_image_prior_losses(inputs)##l1,l2总方差损失,l1未使用
            # loss_l2 = torch.norm(inputs, 2)#l2 范数正则化系数

            if ldr==True:
                # weights = torch.mean(attention_weights[-2], dim=1)[:, 0, :][:, 1:]
                # p = torch.distributions.Normal(torch.zeros_like(weights), torch.ones_like(weights))
                # q = torch.distributions.Normal(weights.mean(), weights.std())
                # # kl_loss += sqrt(torch.distributions.kl_divergence(p, q).sum().detach())
                # kl_loss += torch.distributions.kl_divergence(p, q).sum().detach()
                # # print(kl_loss)
                # # kl_loss += torch.distributions.kl_divergence(p, q).sum()
                # if it < 1000:
                #     kl_loss = torch.tensor(0., device='cuda:0')
                #     aa = 1
                # else:
                #     # aa=3##定义LDR鲁棒超参
                #     # aa = sqrt(kl_loss)* 0.0002
                #     aa = sqrt(kl_loss)* 0.0002

                aa = 0.01##定义LDR鲁棒超参
                one_hot = F.one_hot(targets).float()
                one_hot = torch.zeros(one_hot.size(0), 1000, device=targets.device)
                one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
                exp = torch.exp(t_out)
                sum_ = torch.sum(exp, dim=1).reshape(-1, 1)
                softmax = exp / sum_
                log_softmax = torch.log(softmax/aa)
                loss_oh_2 = -torch.sum(one_hot * log_softmax) / (targets.shape[0]*aa)
                # print(loss_oh_2)
            # loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv1 * loss_tv1 + self.tv2*loss_tv2 + self.l2 * loss_l2##只使用了loss_oh、loss_tv2
            if ldr == True:
                loss = 0.01 * loss_oh_2 + self.tv2 * loss_tv2 + self.oh * loss_oh
            else:
                ldr_loss_ce = ldr_loss(t_out, targets, penalty_function)
                # ldr_loss_ce = 0
                # loss = self.tv2 * loss_tv2 + self.oh * loss_oh
                loss = self.tv2 * loss_tv2 + self.oh * loss_oh + ldr_loss_ce * self.ldr 
                # if it <50:
                #     loss = self.tv2 * loss_tv2 + ldr_loss_ce + self.oh * loss_oh
                # elif it>50 and it<500:
                #     loss = self.tv2 * loss_tv2 + self.oh * loss_oh * 0.1 + ldr_loss_ce * 0.1
                # else:
                # loss = self.tv2 * loss_tv2 + self.oh * loss_oh + ldr_loss_ce * 0.1
                # loss = self.tv2 * loss_tv2 + ldr_loss_ce * 0.1
            if (it == 5000) or (it == 10000) or (it == 400000):
                # memory_stats = torch.cuda.memory_stats()
                # print(memory_stats)
                # 创建一个示例输入张量
                num_patches_t = min(top_K,num_patches)
                print(num_patches_t)
                class ModelWrapper(nn.Module):
                    def __init__(self, model, current_abs_index, next_relative_index, r):
                        super(ModelWrapper, self).__init__()
                        self.model = model
                        self.current_abs_index = current_abs_index
                        self.next_relative_index = next_relative_index
                        self.r = r
                    def forward(self, x):
                        return self.model(x, self.current_abs_index, self.next_relative_index, self.r)
                wrapped_model = ModelWrapper(self.teacher, current_abs_index, next_relative_index, r)
                # 计算FLOPS
                flops = FlopCountAnalysis(wrapped_model, inputs_aug)
                # print(flops)
                print(f"FLOPs: {flops.total() / 1e9} GFLOPs")
                # print(flops.by_module())

                # 使用扩展的 FLOPs 统计工具
                # flops = FlopCountAnalysis(wrapped_model, inputs_aug, supported_ops=custom_handles)
                # print(f"Total FLOPs: {flops.total()}")
            # if (it==30) or (it==500):
            #     memory_stats = torch.cuda.memory_stats()
            #     print(memory_stats)
            #     num_patches_t = min(top_K,num_patches)
            #     print(num_patches_t)
            #     class ModelWrapper(nn.Module):
            #         def __init__(self, model, current_abs_index, next_relative_index, r):
            #             super(ModelWrapper, self).__init__()
            #             self.model = model
            #             self.current_abs_index = current_abs_index
            #             self.next_relative_index = next_relative_index
            #             self.r = r
            #         def forward(self, x):
            #             return self.model(x, self.current_abs_index, self.next_relative_index, self.r)
            #     wrapped_model = ModelWrapper(self.teacher, current_abs_index, next_relative_index, r)
            #     from thop import profile
            #     flops, params = profile(wrapped_model, inputs=(inputs_aug,))
            #     print(f"FLOPs: {flops}, Parameters: {params}")
            if best_cost > loss.item():##记录最好的数据，保存
                best_cost = loss.item()
                # print(best_cost)
                best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            # max_norm = 2.0
            # torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), max_norm)
            optimizer.step()
            inputs.data = clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)
        # print("feqawiyfgqe")

        self.student.train()
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        if len(prune_ratio)==1 and prune_ratio[0]==0: #add non-masked image
            self.data_pool.add( best_inputs )

        with torch.no_grad():
            # r = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            t_out,attention_weights,current_abs_index = self.teacher(best_inputs.detach(),torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device),torch.LongTensor(list(range(num_patches))).repeat(best_inputs.shape[0], 1).to(self.device), r)

        attention_weights = torch.mean(attention_weights[-1], dim=1)[:, 0, :][:, 1:]  # (B,heads,N,N)->(B,p-1)

        def cumulative_mul(lst):
            current_mul = 1
            for num in lst:
                current_mul = current_mul*(1.-num)
            return current_mul
        top_K=int(num_patches*(cumulative_mul(prune_ratio)))
        next_relative_index = get_top_k_relative_indices_including_first(pre_attention=attention_weights, K=top_K).to(self.device)

        mask = torch.zeros(next_relative_index.shape[0], int(sqrt(num_patches)), int(sqrt(num_patches)))
        for b in range(next_relative_index.shape[0]):
            mask[b, (next_relative_index[b][1:] - 1) // int(sqrt(num_patches)), (next_relative_index[b][1:] - 1) % int(sqrt(num_patches))] = 1#填充掩码
        expanded_mask = mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)##扩展张量
        expanded_mask = expanded_mask.to(self.device)
        masked_best_inputs = best_inputs * expanded_mask.unsqueeze(1)
        if not(len(prune_ratio)==1 and prune_ratio[0]==0): #add masked image
            self.data_pool.add( masked_best_inputs )

        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {'synthetic': best_inputs,'masked_synthetic':masked_best_inputs}
        
    def sample(self):
        return self.data_iter.next()