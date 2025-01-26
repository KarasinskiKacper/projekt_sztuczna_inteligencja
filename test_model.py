from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


def test_model(img):
    
    class NeuralNetworkShortcuts(nn.Module):
        def __init__(self, num_chars):
            super().__init__()
            self.leakyReLU = nn.LeakyReLU(negative_slope = 0.1)
            self.dropout = nn.Dropout(0.2)

            self.cnn_11 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
            )
            self.cnn_12 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
            )
            self.cnn_1shortcut = nn.Conv2d(1, 16, kernel_size=1, stride=2)


            self.cnn_21 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
            )
            self.cnn_22 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
            )
            self.cnn_2shortcut = nn.Conv2d(16, 16, kernel_size=1, stride=2)


            self.cnn_3 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Dropout(0.2),
            )


            self.cnn_41 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
            )
            self.cnn_42 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
            )
            self.cnn_4shortcut = nn.Conv2d(16, 32, kernel_size=1, stride=2)


            self.cnn_5 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Dropout(0.2),
            )


            self.cnn_61 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
            )
            self.cnn_62 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
            )
            self.cnn_6shortcut = nn.Conv2d(32, 64, kernel_size=1, stride=2)


            self.cnn_7 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Dropout(0.2),
            )
            # self.cnn_8 = nn.Sequential(
            #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.LeakyReLU(negative_slope = 0.1),
            #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.LeakyReLU(negative_slope = 0.1),
            #     nn.Dropout(0.2),
            # )
            # self.cnn_9 = nn.Sequential(
            #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.LeakyReLU(negative_slope = 0.1),
            #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.LeakyReLU(negative_slope = 0.1),
            #     nn.Dropout(0.2),
            # )

            self.lstm = nn.LSTM(64, 128, bidirectional=True, num_layers=1, batch_first=True)
            self.classifier = nn.Linear(256, num_chars + 1)

        def forward(self, x):
            #x = x.permute(0, 3, 1, 2)
            skip = x
            out = self.leakyReLU(self.cnn_11(x))
            out = self.cnn_12(out)
            out += self.cnn_1shortcut(skip)
            out = self.leakyReLU(out)
            out = self.dropout(out)

            skip = out
            out = self.leakyReLU(self.cnn_21(out))
            out = self.cnn_22(out)
            out += self.cnn_2shortcut(skip)
            out = self.leakyReLU(out)
            out = self.dropout(out)

            out = self.cnn_3(out)

            skip = out
            out = self.leakyReLU(self.cnn_41(out))
            out = self.cnn_42(out)
            out += self.cnn_4shortcut(skip)
            out = self.leakyReLU(out)
            out = self.dropout(out)

            out = self.cnn_5(out)

            skip = out
            out = self.leakyReLU(self.cnn_61(out))
            out = self.cnn_62(out)
            out += self.cnn_6shortcut(skip)
            out = self.leakyReLU(out)
            out = self.dropout(out)

            out = self.cnn_7(out)

            # out = self.cnn_8(out) #cnn_8

            # out = self.cnn_9(out) #cnn_9


            out = out.reshape(out.size(0), -1, out.size(1))
            out, _ = self.lstm(out)
            out = self.dropout(out)

            out = self.classifier(out)
            out = F.log_softmax(out, 2)
            # print(out.size())

            return out

    def preprocess_image(image) -> torch.Tensor:
        img = Image.fromarray(image).convert('L')

        transform = T.Compose([
            T.Resize((32, 256)), 
            T.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension: (1, 1, H, W)
        return img_tensor

    def decode_predictions(predictions: torch.Tensor, char_list: list) -> str:
        # Remove repeated characters and blank tokens (blank index = len(char_list))
        decoded = []
        blank_index = len(char_list)
        prev_char = blank_index
        for idx in predictions:
            if idx != prev_char and idx != blank_index:
                decoded.append(char_list[idx])
            prev_char = idx
        return ''.join(decoded)

    input_tensor = preprocess_image(img)
    # Image.fromarray(input_tensor.squeeze().numpy()*255).show()
 
    model_data = torch.load('./models/test/checkpoint6.pth', map_location=torch.device('cpu'), weights_only=False)
    model = NeuralNetworkShortcuts(num_chars=len(model_data['config']['char_list']))
    model.load_state_dict(model_data['state_dict'])
    model.eval() 

    # summary(model, input_size=(1, 1, 32, 256))

    with torch.no_grad():
        output = model(input_tensor)  # Shape: (batch_size, seq_length, num_chars + 1)
        predictions = output.argmax(-1).squeeze(0)

    char_list = model_data['config']['char_list']
    predicted_text = decode_predictions(predictions, char_list)
    # print(f"Prediction: {predicted_text}")
    return predicted_text