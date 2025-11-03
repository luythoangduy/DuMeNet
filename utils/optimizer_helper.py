import torch


def get_optimizer(parameters, config):
    # Tạo bản sao của config.kwargs, loại bỏ các tham số không cần thiết
    optimizer_kwargs = {k: v for k, v in config.kwargs.items() 
                       if k not in ['backbone_lr_ratio', 'backbone_weight_decay']}
    
    if config.type == "AdamW":
        return torch.optim.AdamW(parameters, **optimizer_kwargs)
    elif config.type == "Adam":
        return torch.optim.Adam(parameters, **optimizer_kwargs)
    elif config.type == "SGD":
        return torch.optim.SGD(parameters, **optimizer_kwargs)
    else:
        raise NotImplementedError