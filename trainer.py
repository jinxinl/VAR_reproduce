import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        '''
        params: device: 模型参数所在设备
                patch_nums: 不同分辨率阶段的patch数量, e.g.(14, 28) 表示第一阶段18*18, 第二阶段24*24
                resos: 对应每个阶段分辨率的大小
                vae_local: image Encoder, VQ-VAE model
                var_wo_ddp: VAR model, 未经DDP包装
                var: VAR model, DDP包装的用于分布式训练的model
                var_opt: AmpOptimizer
                label_smooth: 标签平滑系数
        '''
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2 # 显式注明类型
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng # 删除原有的随机数生成器
        self.var_wo_ddp.rng = torch.Generator(device=device) # 新建设备对应的生成器，为了确保随机数生成与当前设备一致
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none') # 平滑标签和逐元素损失
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean') # 平滑标签和均值损失
        self.L = sum(pn * pn for pn in patch_nums) # 计算总patch数
        self.last_l = patch_nums[-1] * patch_nums[-1] # 最后阶段的patch数
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L # 形状为(1,L)的归一化的向量（原先是全1）
        self.patch_nums, self.resos = patch_nums, resos
        # 计算每个阶段所需要的patch数
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1 # 初始默认不进行渐进式训练
        self.first_prog = True
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training # without distebute data parallel 
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        '''
        params: it: iteration time in current epoch
                g_it: gloabal iteration times
                stepping: 是否进行梯度更新
                metric_lg: MetricLogger
                tb_lg: TensorBoard Logger
                inp_B3HW: input image, (B,3,H,W)
                label: label, (B)
                prog_si: 渐进式训练阶段索引
                prog_wp_it: 渐进式训练预热阶段迭代次数
        '''
        # if progressive training
        # preprocess
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si: # 阶段变化
            if self.last_prog_si != -1: self.first_prog = False # 启用了渐进式训练且并非第一阶段
            self.last_prog_si = prog_si # 更新训练阶段的索引
            self.prog_it = 0 # 当前阶段iter数为0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01) # 渐进式热身系数，阈值0.01控制
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping # 设置是否梯度更新
        
        # img-token sequence
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW) # shape (B,L,V)
        gt_BL = torch.cat(gt_idx_Bl, dim=1) # concatenate on dimension Channel
        # token -> quantization feature
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) 
        
        # 混合精度上下文
        with self.var_opt.amp_ctx: # amp_ctx：自动混合精度上下文，自动管理FP16/FP32的转换
            self.var_wo_ddp.forward
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l) # VAR 处理的是量化后的特征
            # compute original loss
            # reshape logits_BLV as (B*L,V) gt_BL as (B*L)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si] # 当前阶段的起止位置
                # 断言 模型output、真实label、渐进阶段的(序列)长度L要相等
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone() # 截取有效权重
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1) # 应用热身系数实现平滑过渡
            else:               # not in progressive training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean() # 渐进加权后求均值
            if it % 50 == 0:
                print(f"iter: {it}\tloss: {loss}")
            #print(type(loss))
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters: # 当前轮训练结果写入melog日志
            # 整体指标
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            # 渐进式
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                # 计算最后阶段之后的loss acc
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item() 
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            # 捕获异常None
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            else:
                grad_norm= 0.0 ## 相当于不进行梯度更新
            # 记录到melg中
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0: # 总轮数每隔 500 轮写入tblog日志
            # 分布式训练
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen) # 将进程进行聚合统计
            # 计算codebook使用率
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master(): # 主进程
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage) # 分阶段记录码本使用率
                # 分阶段性能记录
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break # si大于当前阶段prog_si，还处于未激活阶段
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item() # 阶段损失
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1 # 阶段复位
        return grad_norm, scale_log2 # 返回梯度信息
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
