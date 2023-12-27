# Sound Stamp

A deep learning project for music tagging to categorize tracks by genre, mood, and more. 

The model is based on the 5-layer fully convolutional neural network (FCN) from [Choi et al. (2016) Automatic tagging using deep convolutional neural networks](https://arxiv.org/abs/1606.00298). It is trained on the top-50 tags from the MagnaTagATune data set, which contains 25.863 tracks.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/hoverslam/sound-stamp
```

2. Navigate to the directory:

```bash
cd sound-stamp
```

3. Set up a virtual environment:

```bash
# Create a virtual environment
python -3.11 -m venv .venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Activate the virtual environment (Unix or MacOS)
source .venv/bin/activate
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```


## Usage

**Train a music tagger:**

```Python
python -m scripts.train -f "music_tagger"
```
```
-m , --model        Name of the file to save the model. Defaults to 'music_tagger'.
```

**Evaluate the performance on Area under the ROC Curve:**

```Python
python -m scripts.evaluate -f "music_tagger"
```
```
-m , --model        File name of the trained model. Defaults to 'music_tagger'.
```

**Tag an audio track:**

```Python
python -m scripts.tag -i "<path to audio file>" -f "music_tagger" -t 0.5
```
```
-i, --input         File path to the audio file.
-m, --model         File name of the trained model. Defaults to 'music_tagger.pt'.
-t, --threshold     Probability threshold for tag prediction. Defaults to 0.5.
```

## License

The code in this project is licensed under the [MIT License](LICENSE.txt).


## Acknowledgements

The project is based on the course [Intelligent Audio and Music Analysis](https://tiss.tuwien.ac.at/course/educationDetails.xhtml?dswid=3058&dsrid=963&courseNr=194039&semester=2023W&locale=en) held at the TU Wien.
