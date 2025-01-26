import os
import lmdb
from PIL import Image
from tqdm import tqdm
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from text_renderer.render import Render
from text_renderer.config import (
    RenderCfg,
    GeneratorCfg,
    FixedTextColorCfg,
    Effects,
    NormPerspectiveTransformCfg
)
from text_renderer.corpus import WordCorpus, WordCorpusCfg
from text_renderer.effect import DropoutRand

# ============== CONFIGURATION ==============
TEXT_PATHS = [Path("./words.txt")]
BG_DIR = Path("./backgrounds")
FONT_DIR = Path("./fonts")
CHARS_FILE = Path("./chars.txt")
LMDB_FULL_PATH = os.path.normpath("./dataset.lmdb")
MAP_SIZE =   64*2**30  # 64GB 
BATCH_SIZE = 500 
# ===========================================

def create_lmdb_directory():
    lmdb_dir = os.path.dirname(LMDB_FULL_PATH)
    
    if not os.path.exists(lmdb_dir):
        print(f"Creating directory: {lmdb_dir}")
        try:
            os.makedirs(lmdb_dir, exist_ok=True)
            if not os.path.isdir(lmdb_dir):
                raise NotADirectoryError(f"Failed to create directory: {lmdb_dir}")
            print("Directory successfully created")
        except Exception as e:
            print(f"Directory creation failed: {str(e)}")
            sys.exit(1)

def initialize_lmdb():
    return lmdb.open(LMDB_FULL_PATH,
                    map_size=MAP_SIZE,
                    max_dbs=3,
                    meminit=False,
                    map_async=True,
                    writemap=True,
                    lock=False,
                    create=True,
                    subdir=True,
                    )


def generate_dataset():
    env = initialize_lmdb()
    
    image_db = env.open_db(b'images')
    label_db = env.open_db(b'labels')
    meta_db = env.open_db(b'meta')

    print("\nGenerating images...")

    charset = []
    with open(CHARS_FILE, "r", encoding="utf8") as f:
        charset = [line.strip() for line in f]
    
    with env.begin(write=True) as txn:
        txn.put(b'charset', ''.join(charset).encode(), db=meta_db)

    config = GeneratorCfg(
        num_image=5_000_000,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            height=64,
            corpus=WordCorpus(
                WordCorpusCfg(
                    text_paths=TEXT_PATHS,
                    font_dir=FONT_DIR,
                    font_size=(16,64),
                    chars_file=CHARS_FILE,
                    filter_by_chars=False,
                    filter_font=True,  
                    num_word=(1, 2), 
                )
            ),
            corpus_effects=Effects([DropoutRand(0.2, (0.05,0.2))]),  
            perspective_transform=NormPerspectiveTransformCfg(scale=0.6),
            text_color_cfg=FixedTextColorCfg(),
            gray=True
        )
    )
    render = Render(config.render_cfg)


    txn = env.begin(write=True) 
    for idx in tqdm(range(config.num_image), desc="Progress"):
        image, label = render()

        image = cv2.convertScaleAbs(image,alpha=1.1)
        if np.random.randint(0,2):
            image = cv2.bitwise_not(image)
        image = cv2.blur(image, np.random.randint(1,10,2), cv2.BORDER_DEFAULT)  
        image = cv2.resize(image, (256,32))
            
        txn.put(f"img_{idx}".encode(), Image.fromarray(image).tobytes(), db=image_db)
        txn.put(f"label_{idx}".encode(), label.encode(), db=label_db)
        txn.put(f"meta_{idx}".encode(), "256,32".encode(), db=meta_db)

        if idx % BATCH_SIZE == 0 and idx > 0:
            txn.commit()
            env.sync()
            txn = env.begin(write=True)


    txn.commit()
    env.close()
    return

if __name__ == "__main__":
    try:
        print(f"Target LMDB path: {LMDB_FULL_PATH}")
        create_lmdb_directory()
        generate_dataset()        
        print(f"LMDB created at:\n{LMDB_FULL_PATH}")

    except Exception as e:
        print(e)
        sys.exit(1)