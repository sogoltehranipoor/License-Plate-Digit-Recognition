# License-Plate-Digit-Recognition
Detects a license plate in an image with classic computer vision (OpenCV) and reads its digits with a small CNN trained on the MNIST dataset (TensorFlow / Keras). Built and run in Google Colab.

![Pipeline](images/pipeline.png)

## How it works

1. **Train** a CNN on MNIST (60,000 handwritten digits), with light augmentation (rotation, translation, zoom) so it tolerates different digit styles.
2. **Find the plate** in the input image using morphological Blackhat/Tophat filtering, a Sobel gradient and contour filtering by aspect ratio.
3. **Split the digits** by thresholding the plate (Otsu) and keeping only the tallest contours, which drops small text, screws and borders.
4. **Classify each digit**: every crop is padded to a square, resized to 28x28 (MNIST format) and passed to the CNN.
5. **Output** the digits, left to right, as the plate number.

## Result

![Result](images/result.png)

- MNIST test accuracy: `XX.X%` (fill in from your notebook output)
- Example plate: `30-0160` -> predicted: `XXXXXX`

## Tech stack

Python, TensorFlow / Keras, OpenCV, NumPy, Matplotlib, Google Colab.

## Getting started

1. Open `license_plate_mnist.ipynb` in Google Colab.
2. Run the training cell (MNIST is downloaded automatically by Keras).
3. Run the last cell and upload a photo of a plate or a car.

Or run locally:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
tensorflow
opencv-python
numpy
matplotlib
```

## Limitations

- The model only knows the digits 0-9 as written in MNIST. Printed plate fonts differ from handwriting, so thin digits like `1` can be misread. Check the per-digit confidence printed by the notebook.
- Letters and non-Latin digits (for example Persian or Thai numerals) are not supported and may be read as wrong digits.
- Plate detection with fixed OpenCV rules works best on clear, front-facing photos. Angled, dark or blurry images may fail.

## Future work

- Fine-tune on printed-digit data (fonts, or a dataset of plate digits) instead of only MNIST.
- Replace the OpenCV plate finder with an object detector such as YOLO.
- Support letters and other numeral systems.

## Author

[Your name] - [LinkedIn link]
