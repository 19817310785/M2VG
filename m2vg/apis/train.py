import time
import torch
import random    # 👉 补回
import numpy     # 👉 补回
from m2vg.datasets import extract_data
from m2vg.utils import get_root_logger, reduce_mean, is_main
import wandb
try:
    import apex
except:
    pass

# 👉 【把不小心删掉的这个函数原封不动贴回来】
def set_random_seed(seed, deterministic=False):
    """Args:
    seed (int): Seed to be used.
    deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
        to True and `torch.backends.cudnn.benchmark` to False.
        Default: False.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(epoch, cfg, model, optimizer, loader):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    # 只保留最基础的三个 loss 列表
    loss_det_list, loss_mask_list, loss_label_nt_list, loss_exist_list = [], [], [], []

    for batch, inputs in enumerate(loader):
        data_time = time.time() - end
        if not cfg.distributed:
            inputs = extract_data(inputs)
        losses, predictions = model(**inputs, epoch=epoch, rescale=False)

        loss_det = losses.pop("loss_det", torch.tensor([0.0], device=device))
        loss_mask = losses.pop("loss_mask", torch.tensor([0.0], device=device))
        loss_label_nt = losses.pop("loss_label_nt", torch.tensor([0.0], device=device))
        loss_exist = losses.pop("loss_exist", torch.tensor([0.0], device=device))

        # 纯净版的总 loss，没有任何其他干扰
        loss = loss_det + loss_mask + loss_label_nt + loss_exist

        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if cfg.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
        optimizer.step()

        if cfg.distributed:
            loss_det = reduce_mean(loss_det)
            loss_mask = reduce_mean(loss_mask)
            loss_label_nt = reduce_mean(loss_label_nt)
            loss_exist = reduce_mean(loss_exist)

        loss_det_list.append(loss_det.item())
        loss_mask_list.append(loss_mask.item())
        loss_label_nt_list.append(loss_label_nt.item())
        loss_exist_list.append(loss_exist.item())

        if is_main():
            if (batch + 1) % cfg.log_interval == 0 or batch + 1 == batches:
                logger = get_root_logger()
                logger.info(
                    f"train - epoch [{epoch + 1}]-[{batch + 1}/{batches}] "
                    + f"time: {(time.time() - end):.2f}, data_time: {data_time:.2f}, "
                    + f"loss_det: {sum(loss_det_list) / len(loss_det_list) :.4f}, "
                    + f"loss_mask: {sum(loss_mask_list) / len(loss_mask_list):.4f}, "
                    + f"loss_nt: {sum(loss_label_nt_list) / len(loss_label_nt_list):.4f}, "
                    + f"loss_exist: {sum(loss_exist_list) / len(loss_exist_list):.4f}, "
                    + f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                )

                wandb.log(
                    {
                        "loss_det": sum(loss_det_list) / len(loss_det_list),
                        "loss_mask": sum(loss_mask_list) / len(loss_mask_list),
                        "loss_label_nt": sum(loss_label_nt_list) / len(loss_label_nt_list),
                        "loss_exist": sum(loss_exist_list) / len(loss_exist_list),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

        end = time.time()
