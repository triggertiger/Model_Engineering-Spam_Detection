from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from typing import List, Optional
from src.predict_pipeline import Predictor, Message
from src.model_trainer import ModelTrainer

app = FastAPI(title="Spam Filter API", version="1.0")

predictor = Predictor()

class PredictInput(BaseModel):
    risk_level: str = Query("low", enum=["low", "medium", "high"])
    messages: Optional[List[str]]


class MessageOutput(BaseModel):
    text: str
    prob: float
    tag: str
    class_: str

@app.post("/predict")#, response_model=list[Message])
def predict(input_data: PredictInput):

    messages_to_predict = input_data.messages if input_data.messages else predictor.test_data['text'].tolist()
    results = predictor.predict_messages(messages_to_predict, threshold=input_data.risk_level, low_risk_level=0.9)

    # convert dataclass objects to dict
    results_dict = [r.__dict__ for r in results]
    print(results[:4])
    return {
        "n_messages": len(results_dict),
        "examples": results_dict[:5]  # first 5 examples
    }


@app.post("/train")
def train_model(experiment_name: Optional[str] = Body(None)):
    """
    Trigger training of the spam detection model.
    Optional: provide experiment_name to label this run.
    """
    trainer = ModelTrainer()
    trainer.train_experiments(experiment_name=experiment_name)
    trainer.train_best_model()
    return {"status": "Training completed", "experiment_name": trainer.experiment_name, "best_run_id": trainer.best_run_id}

