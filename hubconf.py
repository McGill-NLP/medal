import torch

def __build_model(model_class, ckpt, pretrained=True, device='cpu', progress=True, check_hash=True):
    net = model_class(device=device)

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            ckpt, 
            map_location=torch.device(device), 
            progress=progress, 
            check_hash=check_hash
        )

        for key, value in state_dict.items():
            new_key = key[len('module.'): ] if key.startswith('module.') else key
            if new_key not in net.state_dict():
                print(new_key, 'not expected')
                continue
            try:
                net.state_dict()[new_key].copy_(value)
            except:
                print(new_key, 'not loaded')
                continue

    return net


def electra(pretrained=True, device='cpu', progress=True):
    from models.electra import Electra
    return __build_model(
        Electra, 
        ckpt="https://github.com/BruceWen120/medal/releases/download/data/electra.pt",
        pretrained=pretrained, device=device, progress=progress)


def lstm(pretrained=True, device='cpu', progress=True, check_hash=True):    
    from models.rnn import RNN
    return __build_model(
        RNN,
        ckpt="https://github.com/BruceWen120/medal/releases/download/data/lstm.pt",
        pretrained=pretrained, device=device, progress=progress
    )


def lstm_sa(pretrained=True, device='cpu', progress=True, check_hash=True):
    from models.lstm_sa import RNNAtt
    return __build_model(
        RNNAtt,
        ckpt="https://github.com/BruceWen120/medal/releases/download/data/lstm_sa.pt",
        pretrained=pretrained, device=device, progress=progress
    )
