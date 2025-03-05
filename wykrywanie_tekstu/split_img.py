import torch
from torch.autograd import Variable

import cv2
import numpy as np
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .imgproc import cvt2HeatmapImg, resize_aspect_ratio, normalizeMeanVariance, loadImage

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

    def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)   
        x = Variable(x.unsqueeze(0))               
        if cuda:
            x = x.cuda()

        with torch.no_grad():
            y, feature = net(x)

        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()
            
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = cvt2HeatmapImg(render_img)

        return boxes, polys, ret_score_text
    
    refine_net = None
   
    image = np.array(image)

    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net)
    sorted_polys = np.zeros_like(polys)
    i = 0
    while i < len(polys):
        start_i = i
        tmp_polys = []
        current_min_y = polys[i][0][1] 
        current_half_line_hight = (polys[i][2][1] - current_min_y) / 4
        tmp_polys.append(polys[i])
        i += 1
        while i < len(polys) and polys[i][0][1] - current_half_line_hight < current_min_y < polys[i][2][1] + current_half_line_hight:
            tmp_polys.append(polys[i])
            i += 1
        tmp_polys = sorted(tmp_polys, key=lambda x: x[0][0])
        sorted_polys[start_i:i] = tmp_polys
        
    

    images = []
    for poly in sorted_polys:
        min_x = int(poly[0][0])
        max_x = int(poly[2][0])
        min_y = int(poly[0][1])
        max_y = int(poly[2][1])
        
        new_img = np.array(image[min_y:max_y, min_x:max_x])
        
        images.append(new_img)
        
    return images
