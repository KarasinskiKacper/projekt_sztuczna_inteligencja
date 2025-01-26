from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


def model_pawel_h(img):
    # class ReEnhancedOCRNet(nn.Module):
    #     def __init__(self, num_chars):
    #         super().__init__()
    #         self.leakyReLU = nn.LeakyReLU(0.1)
    #         self.dropout = nn.Dropout(0.3)

    #         # Smaller Initial Feature Extraction
    #         self.init_conv = nn.Sequential(
    #             nn.Conv2d(1, 8, 3, stride=1, padding=1),  # Reduced from 16 to 8 channels
    #             nn.BatchNorm2d(8),
    #             self.leakyReLU
    #         )
            
    #         # Simplified Downsampling Blocks with adjusted input channels
    #         self.block1 = self._make_res_block(8, 16, stride=(2,1))    # H=16
    #         self.block2 = self._make_res_block(16, 32, stride=(2,1))   # H=8
    #         self.block3 = self._make_res_block(32, 64, stride=(2,2))   # H=4
            
    #         # Final processing
    #         self.final_conv = nn.Sequential(
    #             nn.Conv2d(64, 64, (1,3), stride=1, padding=(0,1)),
    #             nn.BatchNorm2d(64),
    #             self.leakyReLU,
    #         )

    #         # Reduced Sequence Modeling with GRU
    #         self.gru = nn.GRU(64, 64, bidirectional=True, num_layers=1, batch_first=True)  # GRU instead of LSTM
    #         self.attention = nn.MultiheadAttention(128, 4, dropout=0.3)  # Fewer attention heads
    #         self.classifier = nn.Linear(128, num_chars + 1)  # Adjusted input size

    #     def _make_res_block(self, in_ch, out_ch, stride):
    #         return nn.Sequential(
    #             ResSepConv(in_ch, out_ch, stride)
    #         )

    #     def forward(self, x):
    #         # Feature extraction
    #         x = self.init_conv(x)        # (B,8,32,256)
    #         x = self.block1(x)           # (B,16,16,256)
    #         x = self.block2(x)           # (B,32,8,256)
    #         x = self.block3(x)           # (B,64,4,128)
            
    #         # Final processing
    #         x = self.final_conv(x)       # (B,64,3,128)
    #         x = x.mean(dim=2)            # (B,64,128)
    #         x = x.permute(0,2,1)         # (B,128,64)
            
    #         # Sequence modeling with GRU
    #         x, _ = self.gru(x)           # (B,128,128) - GRU output
    #         x = x.permute(1,0,2)         # (128,B,128)
    #         x, _ = self.attention(x,x,x) # (128,B,128)
    #         x = x.permute(1,0,2)         # (B,128,128)
            
    #         # Classification
    #         x = self.classifier(x)       # (B,128,num_chars+1)
    #         return F.log_softmax(x, 2)

    # class ResSepConv(nn.Module):
    #     def __init__(self, in_ch, out_ch, stride):
    #         super().__init__()
    #         self.leakyReLU = nn.LeakyReLU(0.1)
            
    #         self.conv = nn.Sequential(
    #             nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch),
    #             nn.Conv2d(in_ch, out_ch, 1),
    #             nn.BatchNorm2d(out_ch)
    #         )
            
    #         self.shortcut = nn.Sequential()
    #         if stride != 1 or in_ch != out_ch:
    #             self.shortcut = nn.Sequential(
    #                 nn.Conv2d(in_ch, out_ch, 1, stride),
    #                 nn.BatchNorm2d(out_ch)
    #             )

    #     def forward(self, x):
    #         return self.leakyReLU(self.conv(x) + self.shortcut(x))
    class ReEnhancedOCRNet(nn.Module):
        def __init__(self, num_chars):
            super().__init__()
            self.leakyReLU = nn.LeakyReLU(0.1)
            self.dropout = nn.Dropout(0.2)  # Reduced dropout

            # Smaller Initial Feature Extraction
            self.init_conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=1, padding=1),  # Reduced channels
                nn.BatchNorm2d(16),
                self.leakyReLU
            )
            
            # Simplified Downsampling Blocks
            self.block1 = self._make_res_block(16, 16, stride=(2,1))   # H=16
            self.block2 = self._make_res_block(16, 32, stride=(2,1))   # H=8
            self.block3 = self._make_res_block(32, 64, stride=(2,2))   # H=4 (removed 1 block)
            
            # Final processing
            self.final_conv = nn.Sequential(
                nn.Conv2d(64, 64, (1,3), stride=1, padding=(0,1)),  # Smaller kernel
                nn.BatchNorm2d(64),
                self.leakyReLU,
            )

            # Reduced Sequence Modeling
            self.lstm = nn.LSTM(64, 128, bidirectional=True, num_layers=2, batch_first=True)  # Fewer layers/units
            self.attention = nn.MultiheadAttention(256, 4, dropout=0.2)  # Fewer heads
            self.classifier = nn.Linear(256, num_chars + 1)

        def _make_res_block(self, in_ch, out_ch, stride):
            return nn.Sequential(  # Single ResSepConv per block
                ResSepConv(in_ch, out_ch, stride)
            )

        def forward(self, x):
            # Feature extraction
            x = self.init_conv(x)        # (B,16,32,256)
            x = self.block1(x)           # (B,16,16,256)
            x = self.block2(x)           # (B,32,8,256)
            x = self.block3(x)           # (B,64,4,128)
            
            # Final processing
            x = self.final_conv(x)       # (B,64,3,128)
            x = x.mean(dim=2)            # (B,64,128) - Adaptive pooling
            x = x.permute(0,2,1)         # (B,128,64)
            
            # Sequence modeling
            x, _ = self.lstm(x)          # (B,128,256)
            x = x.permute(1,0,2)         # (128,B,256)
            x, _ = self.attention(x,x,x) # (128,B,256)
            x = x.permute(1,0,2)         # (B,128,256)
            
            # Classification
            x = self.classifier(x)       # (B,128,num_chars+1)
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
 
    model_data = torch.load('./models/pawel_h/best.pth', map_location=torch.device('cpu'), weights_only=False)
    model = ReEnhancedOCRNet(num_chars=len(model_data['config']['char_list']))
    model.load_state_dict(model_data['state_dict'])
    model.eval() 


    with torch.no_grad():
        output = model(input_tensor) 
        predictions = output.argmax(-1).squeeze(0)

    char_list = model_data['config']['char_list']
    predicted_text = decode_predictions(predictions, char_list)
    return predicted_text