import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
import wykrywanie_tekstu.imgproc
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .imgproc import cvt2HeatmapImg, resize_aspect_ratio, normalizeMeanVariance, loadImage

from .craft import CRAFT

from collections import OrderedDict
def split_img(
        net,
        image,
        trained_model = './weights/CRAFT.pth', 
        text_threshold = 0.7,
        low_text = 0.4,
        link_threshold = 0.4,
        cuda = False, 
        canvas_size = 1280, 
        mag_ratio=1.5,
        poly=False,
        show_time=False,
        refiner_model='weights/craft_refiner_CTW1500.pth'
    ):
    
    # def copyStateDict(state_dict):
    #     if list(state_dict.keys())[0].startswith("module"):
    #         start_idx = 1
    #     else:
    #         start_idx = 0
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = ".".join(k.split(".")[start_idx:])
    #         new_state_dict[name] = v
    #     return new_state_dict


    def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        # resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()
            
        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img)

        return boxes, polys, ret_score_text

    # net = CRAFT() 
    # if cuda:
    #     net.load_state_dict(copyStateDict(torch.load(trained_model, weights_only=False)))
    # else:
    #     net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu', weights_only=False)))

    # if cuda:
    #     net = net.cuda()
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = False

    # net.eval()

    # LinkRefiner
    refine_net = None
   
    image = np.array(image)

    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)

    images = []
    for poly in polys:
        min_x = int(poly[0][0])
        max_x = int(poly[2][0])
        min_y = int(poly[0][1])
        max_y = int(poly[2][1])
        
        new_img = np.array(image[min_y:max_y, min_x:max_x])
        
        images.append(new_img)
        
    return images
    
    

# images = split_img(trained_model = "./weights/CRAFT.pth", image_path='./img/test.jpg' )

# for img in images:
#     print(img)