from PIL import Image
import math

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


def model_pawel_d(img):
    class ChannelAttention(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels//8, 1),
                nn.SiLU(),
                nn.Conv2d(channels//8, channels, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return x * self.se(x)

    class ResSepConv(nn.Module):
        def __init__(self, in_ch, out_ch, stride):
            super().__init__()
            self.activation = nn.SiLU()
            
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, groups=1, bias=False),
                nn.BatchNorm2d(out_ch),
                ChannelAttention(out_ch),
                nn.Dropout2d(0.1),
                self.activation
            )
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )

        def forward(self, x):
            return self.conv(x) + self.shortcut(x)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return x

    class OCRModel(nn.Module):
        def __init__(self, num_chars):
            super().__init__()
            self.activation = nn.SiLU()
            
            self.init_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32, bias=False),
                nn.Conv2d(32, 64, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU()
            )
            
            self.block1 = self._make_res_block(64, 64, stride=(2,1))    # (16,256)
            self.block2 = self._make_res_block(64, 128, stride=(2,1))   # (8,256)
            self.block3 = self._make_res_block(128, 256, stride=(2,2))  # (4,128)
            self.block4 = self._make_res_block(256, 512, stride=(2,2))  # (2,64)
            
            self.final_conv = nn.Sequential(
                nn.Conv2d(512, 512, (2,3), stride=1, padding=(0,1), groups=512, bias=False),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.SiLU(),
                nn.Dropout(0.3)
            )

            self.pos_encoder = PositionalEncoding(512)
            self.ln = nn.LayerNorm(512)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512, nhead=8, dim_feedforward=2048,
                dropout=0.3, activation='gelu', batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            
            self.classifier = nn.Linear(512, num_chars + 1)
            self._init_weights()
            
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.xavier_normal_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            nn.init.constant_(param.data, 0)
                            n = param.size(0)
                            param.data[n//4:n//2].fill_(1.0)
                elif isinstance(m, nn.TransformerEncoderLayer):
                    for p in m.parameters():
                        if p.dim() > 1:
                            nn.init.xavier_uniform_(p)

        def _make_res_block(self, in_ch, out_ch, stride):
            return nn.Sequential(
                ResSepConv(in_ch, out_ch, stride),
                ResSepConv(out_ch, out_ch, 1)
            )

        def forward(self, x):
            x = self.init_conv(x)               # (B,64,32,256)
            x = self.block1(x)                  # (B,64,16,256)
            x = self.block2(x)                  # (B,128,8,256)
            x = self.block3(x)                  # (B,256,4,128)
            x = self.block4(x)                  # (B,512,2,64)
            
            x = self.final_conv(x)              # (B,512,1,64)
            x = x.squeeze(2).permute(0,2,1)     # (B,64,512)
            x = self.pos_encoder(x)             
            x = self.ln(x)
            
            x = self.transformer(x)             # (B,64,512)
            x = self.classifier(x)              # (B,64,num_chars+1)
            return F.log_softmax(x, dim=2)

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
 
    model_data = torch.load('./models/pawel_d/state_last.pth', map_location=torch.device('cpu'), weights_only=False)
    
    model = OCRModel(num_chars=len(model_data['char_list']))
    model.load_state_dict(model_data['state_dict'])
    model.eval() 


    with torch.no_grad():
        output = model(input_tensor) 
        predictions = output.argmax(-1).squeeze(0)

    char_list = model_data['char_list']
    predicted_text = decode_predictions(predictions, char_list)
    return predicted_text