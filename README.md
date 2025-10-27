# SMS Spam Classifier Project

## Project – Model Engineering

This project implements a machine learning pipeline for classifying SMS messages as spam or ham. It includes data preprocessing, model training with hyperparameter search, MLflow logging, and a FastAPI application for serving predictions.

The pipeline includes:

* `TextCleaner` for preprocessing,
* `TfidfVectorizer` or `CountVectorizer` for vectorization,
* `Multinomial Naive Bayes` classifier.

* Test datasets are automatically logged and used if no messages are provided for predictions to be tested through the API. See below.

## Project Structure
```
.
├── app/                  # FastAPI application
├── config.py             # Configuration variables
├── data/                 # Raw data (SMSSpamCollection.csv)
├── notebooks/            # EDA, preprocessing, model training
├── src/                  # Core scripts: data prep, cleaning, training, prediction, utils
├── tests/                # Optional unit tests
├── requirements.txt      # Required packages
├── README.md
└── main.py               # Entry point to run API

```
## Setup Instructions

1. Clone the repository:
```
git clone <repository_url>
cd <project_folder>
```
2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate       
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. the data can be downloaded from the course module into the `/data` library, or from kaggle with:  
```
#!/bin/bash
mkdir -p ./data
curl -L -o ./data/sms-spam-collection-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset
unzip ./data/sms-spam-collection-dataset.zip -d ./data
```
> you will need to have your Kaggle Profile API set-up for this option

## Running the API
1. Start the FastAPI server:
```
python main.py
```
2. Access the API documentation:

    Open your browser and go to:
```arduino
http://127.0.0.1:8000/docs
```
## API Usage
### Predict Endpoint
* Endpoint: /predict (POST)
* Input format: JSON object with optional messages list and risk_level

**Important:**
* The messages field must be a list of strings.
* You cannot send an empty string or leave a placeholder like "message": "".
* To use the default test dataset, simply omit the messages field.

**Example JSON:**
```json
{
    "risk_level": "medium",
    "messages": ["Congratulations! You won a free ticket.", "Hello, are we meeting today?"]
}
```
**Response:**
* Returns number of messages processed and the first 5 predictions.
* Each message contains: text, prob (spam probability), class_ (ham/spam), tag (confidence level).

### Train Endpoint
* Endpoint: /train (POST)
* Input: Optional experiment name as raw string JSON:
```json
"experiment_name": "experiment_NB_V2"
```

* Description: Triggers model training using the raw dataset, logs experiments to MLflow, and registers the best model.

**Response:**
```json
{
    "status": "Training completed",
    "experiment_name": "experiment_NB_V2",
    "best_run_id": "<MLflow_run_id>"
}
```
