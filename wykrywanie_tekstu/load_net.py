from .craft import CRAFT
import torch
import torch.backends.cudnn as cudnn

from collections import OrderedDict

def load_net(trained_model = './wykrywanie_tekstu/weights/CRAFT.pth',cuda = False, ):
    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    net = CRAFT() 
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model, weights_only=False)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu', weights_only=False)))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    
    return net