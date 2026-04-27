# Chars74K-OCR
An OCR Implementation using Chars74K Dataset

### Dataset Download Link: [Chars74K-Digital-English-Font](https://www.kaggle.com/datasets/supreethrao/chars74kdigitalenglishfont)

---

# Usage:
## Train

Download the Chars74K EnglishFnt*.7z dataset and place it in the project root. Then run: `python train.py`

It extracts the dataset automatically, trains for 30 epochs, and saves the best weights to chars74k.pth. You can tweak things:

`python train.py --epochs 50 --batch-size 128 --lr 5e-4 --model-out my_model.pth`

To skip training and just evaluate an existing model:
`python train.py --eval-only`

## Run OCR
Point it at an image and it'll extract and print the text: `python ocr.py image.png`

Save an annotated copy showing the detected character boxes: `python ocr.py image.png --save-vis annotated.png`

If your image is a single pre-cropped character: `python ocr.py char.png --single char`

Use a different model file with `--model path/to/weights.pth`
