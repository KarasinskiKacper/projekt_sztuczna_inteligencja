from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


def model_mikolaj_c(img):
    class ReEnhancedOCRNet(nn.Module):
        def __init__(self, num_chars):
            super().__init__()
            self.leakyReLU = nn.LeakyReLU(0.1)
            self.dropout = nn.Dropout(0.3)

            self.init_conv = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=1, padding=1),  
                nn.BatchNorm2d(8),
                self.leakyReLU
            )
            
            self.block1 = self._make_res_block(8, 16, stride=(2,1))    
            self.block2 = self._make_res_block(16, 32, stride=(2,1))   
            self.block3 = self._make_res_block(32, 64, stride=(2,2))   
            
            self.final_conv = nn.Sequential(
                nn.Conv2d(64, 64, (1,3), stride=1, padding=(0,1)),
                nn.BatchNorm2d(64),
                self.leakyReLU,
            )

            self.gru = nn.GRU(64, 64, bidirectional=True, num_layers=1, batch_first=True)
            self.attention = nn.MultiheadAttention(128, 4, dropout=0.3) 
            self.classifier = nn.Linear(128, num_chars + 1) 

        def _make_res_block(self, in_ch, out_ch, stride):
            return nn.Sequential(
                ResSepConv(in_ch, out_ch, stride)
            )

        def forward(self, x):
            x = self.init_conv(x)       
            x = self.block1(x)          
            x = self.block2(x)          
            x = self.block3(x)          
            
            x = self.final_conv(x)       
            x = x.mean(dim=2)            
            x = x.permute(0,2,1)         
            
            x, _ = self.gru(x)        
            x = x.permute(1,0,2)         
            x, _ = self.attention(x,x,x) 
            x = x.permute(1,0,2)         
            
            x = self.classifier(x)       
            return F.log_softmax(x, 2)

    class ResSepConv(nn.Module):
        def __init__(self, in_ch, out_ch, stride):
            super().__init__()
            self.leakyReLU = nn.LeakyReLU(0.1)
            
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride),
                    nn.BatchNorm2d(out_ch)
                )

        def forward(self, x):
            return self.leakyReLU(self.conv(x) + self.shortcut(x))
    
    def preprocess_image(image) -> torch.Tensor:
        img = Image.fromarray(image).convert('L')

        transform = T.Compose([
            T.Resize((32, 256)), 
            T.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0) 
        return img_tensor

    def decode_predictions(predictions: torch.Tensor, char_list: list) -> str:
        decoded = []
        blank_index = len(char_list)
        prev_char = blank_index
        for idx in predictions:
            if idx != prev_char and idx != blank_index:
                decoded.append(char_list[idx])
            prev_char = idx
        return ''.join(decoded)

    input_tensor = preprocess_image(img)
 
    model_data = torch.load('./models/mikolaj_c/best.pth', map_location=torch.device('cpu'), weights_only=False)
    model = ReEnhancedOCRNet(num_chars=len(model_data['config']['char_list']))
    model.load_state_dict(model_data['state_dict'])
    model.eval() 


    with torch.no_grad():
        output = model(input_tensor) 
        predictions = output.argmax(-1).squeeze(0)

    char_list = model_data['config']['char_list']
    predicted_text = decode_predictions(predictions, char_list)
    return predicted_text