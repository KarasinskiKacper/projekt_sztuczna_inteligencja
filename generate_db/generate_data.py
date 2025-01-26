import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2
import os


class PolishWordsDataset(Dataset):
    def __init__(self, dir):
        self.samples= list()
        self.max_len = None

        excluded_signs = ['ä', 'ô', 'í', 'é', 'š', 'ø', 'è', 'ö', 'î', 'ë', 'ü', 'à', 'ê', 'ş', 'ç', 'č', 'ū', 'ú', 'û', 'á', 'â', 'ñ', 'ù', 'ō', 'Ä', 'Ô', 'Í', 'É', 'Š', 'Ø', 'È', 'Ö', 'Î', 'Ë', 'Ü', 'À', 'Ê', 'Ş', 'Ç', 'Č', 'Ū', 'Ú', 'Û', 'Á', 'Â', 'Ñ', 'Ù', 'Ō']

        tmp_set = set()
        with open(dir, 'r', encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                sentences = line.split(",")
                for sentence in sentences:
                    words = sentence.split(" ")
                    for word in words:
                        if not any(sign in word for sign in excluded_signs) and len(word) < 21:                            
                            tmp_set.add(word.strip())

        self.samples = list(tmp_set)

        char_set = set()
        for sample in self.samples:
            for char in sample:
                char_set.add(char)

        chars = list(char_set)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.char_to_idx['<blank>'] = 0
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
            

    def __len__(self):
        if self.max_len is None:
            return len(self.samples)
        else:
            return self.max_len
        
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        else:
            item = self.__getitem__(self.index)
            self.index += 1
            return item
        
    def create_image(self, text):
        fonts = [
                "digital/Inter/Inter-Italic-VariableFont_opsz,wght.ttf",
                "digital/Inter/Inter-VariableFont_opsz,wght.ttf",
                "digital/Montserrat/Montserrat-VariableFont_wght.ttf",
                "digital/Montserrat/Montserrat-Italic-VariableFont_wght.ttf",
                "digital/Noto_Sans/NotoSans-VariableFont_wdth,wght.ttf",
                "digital/Noto_Sans/NotoSans-Italic-VariableFont_wdth,wght.ttf",
                "digital/Open_Sans/OpenSans-VariableFont_wdth,wght.ttf",
                "digital/Open_Sans/OpenSans-Italic-VariableFont_wdth,wght.ttf",
                "digital/Oswald/Oswald-VariableFont_wght.ttf",
                "digital/Roboto/Roboto-VariableFont_wdth,wght.ttf",
                "digital/Roboto/Roboto-Italic-VariableFont_wdth,wght.ttf",
                "digital/Roboto_Mono/RobotoMono-VariableFont_wght.ttf",
                "digital/Roboto_Mono/RobotoMono-Italic-VariableFont_wght.ttf",
                ]
        
        font = ImageFont.truetype(os.path.join("./fonts", np.random.choice(fonts)),size=np.random.randint(16,20))
        font.set_variation_by_name(np.random.choice(font.get_variation_names()))

        bbox = font.getbbox(text)
        color = np.random.randint(200,255)
        if color >= 127:
            inverse = np.random.randint(0,25)
        else:
            inverse = np.random.randint(225,255)
  
        
        # img = Image.new('L', (bbox[2] + margin[0] + margin[2], bbox[3] - bbox[1] + margin[1] + margin[3]), color=color)
        img = Image.new('L', (300,60), color=color)
        draw = ImageDraw.Draw(img)
        
        # draw.text((margin[0], margin[1]-bbox[1]), text, font=font, fill=inverse)
        draw.text(((300  - (bbox[2]))/2, (60 - (bbox[3] - bbox[1]))/2), text, font=font, fill=inverse)
        
        transform = v2.Compose([
            v2.RandomRotation(3),
            v2.CenterCrop((32,256)),
            # v2.ColorJitter(contrast = (1.5,1.5))
        ])

        img = transform(img)
        
        return img
        

    def __getitem__(self, index):
        text = self.samples[index]
        image = self.create_image(text)

        # target = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

        return image, text
       
    
    def shuffle(self):
        np.random.shuffle(self.samples)

    def limit(self, value):
        self.max_len = value
        
        
dataset = PolishWordsDataset("./odm.txt")
dataset.shuffle()
# labels = ""
# print(len(dataset))


for i, (img, label) in enumerate(dataset):
    # labels += f"{label}\n"
    if i%10_000 == 0:
        print(f"\r{i/1_000_000}", end="")
        if i >1_000_000:
            break
    with open("C:/Users/karas/programowanie/uczenie maszynowe/digital_data/labels.txt", 'a', encoding="utf-8") as file:
        file.write(f"{label}\n")
    img.save(f"C:/Users/karas/programowanie/uczenie maszynowe/digital_data/{i}.png")
    
    
    
# print(dataset.samples[0])
# print(len([i for i in dataset.samples if len(i)<21]))
    # 5% 13min 13%
# with open("./Tutorials/08_handwriting_recognition_torch/data/labels.txt", 'w', encoding="utf-8") as file:
#     file.write(labels)
