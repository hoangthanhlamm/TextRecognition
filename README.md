# Image Text Recognition

Recognite text from image with CRNN network.

## Install requirements

```
pip install -r requirements.txt
```
## Dataset

Full dataset used is [Synthetic Word Dataset](https://www.robots.ox.ac.uk/~vgg/data/text/)  
Dataset used for project [here](https://drive.google.com/drive/folders/1A7vDf8_ZeohltSQuHcHAKHFkk1ZPHqHE?usp=sharing).

## Model's weights

Model's weights for prediction and evaluation [model.weights.hdf5](https://drive.google.com/file/d/1KvzHgnXAOHQfrU9QkMo-K4yMhTP6P5et/view?usp=sharing)

### Download data and model's weights

```
cd text-recognition/
python lib/preprocess/download_data.py
```

### Extract data and model's weights

```
tar -xzvf data.tar.gz
rm -rf data.tar.gz
```

### Create checkpoints folder 

```
mkdir Data/checkpoints/
```

## Run application

```
python main.py
```

Application run on <u>http://localhost:8096 </u>

## Api

1. Training model
    - URL: /train
    - Method: GET
    - Usage: Training model and save weights

2. Evaluation 
    - URL: /evaluation
    - Method: POST
    - Body:
        - **filename**: Path to file contain image's paths and labels
        - **paths**: List of image's paths
        - **labels**: List of image's labels
    - Usage: Evaluate accuracy and letter accuracy of model with data described in filename or (paths, labels).
    - Return: Accuracy, letter accuracy

3. Prediction
    - URL: /prediction
    - Method: POST
    - Body:
        - **img**: Path to image for prediction
    - Usage: Predict text in new image
    - Return: Text predicted
