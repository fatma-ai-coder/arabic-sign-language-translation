# Arabic Sign Language Translation

Real-time translation of Arabic Sign Language (ArSL) letters into Arabic text, using a webcam, a hand detector and a CNN classifier.

![Sign to text demo](videos/sign-to-text-gif.gif)

A research project at **Dar Al-Hekma University**, presented at the **CCTS conference** (Computing and Cybersecurity Technologies Symposium).

**Paper:** *Using Machine Learning to Transcribe Visual Arabic Sign Language*, by Anas Al-Tirawi, Fatma Alamoudi, Rahaf Al-Shabrawi and Judi Al-Baghajati. [Read the PDF](docs/arabic_sign_language_paper.pdf)

## How it works

1. **Hand detection:** an SSD model (TensorFlow frozen graph) finds the hand in each webcam frame.
2. **Letter classification:** the cropped hand is resized to 64x64 and passed to a CNN trained on 32 ArSL letter classes.
3. **Arabic output:** predicted letters are joined into words and drawn on screen with correct right-to-left shaping (`arabic-reshaper` + `python-bidi`).

## Dataset

ArASL2018 (ArASL_Database_54K): about 54,000 images of 32 Arabic sign language letters. Training notebook: [`train_model/train_char_model.ipynb`](train_model/train_char_model.ipynb)

## Run it

```bash
pip install -r requirements.txt
python simple_test.py              # quick webcam test of the letter classifier
python ASL_detection_landmark.py   # full pipeline: hand detection + classification
```

Needs Python 3.8 (the pinned TensorFlow 2.4 does not install on newer versions) and a webcam.

## Tech

Python, TensorFlow / Keras, OpenCV, dlib, NumPy, Pandas

## Next steps

- Word and sentence level recognition with sequence models for moving signs
- Transfer learning on larger datasets
- A mobile app for daily use

## Credits

The base code was adapted from [Pavly Halim](https://github.com/pavlyhalim/Arabic-Sign-Language) under the MIT License, then restructured and extended for this research.

## License

[MIT](LICENSE)
