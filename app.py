from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from transformers import T5Tokenizer ,T5ForConditionalGeneration
import time


# uvicorn app:app --reload  --host 0.0.0.0 to run server
# to build app:  flutter build web --base-href "/static/"


# base_path = './../model/t5-small-gcp'
base_path = './checkpoint_720000'
trained_model = T5ForConditionalGeneration.from_pretrained(base_path + "/model")
tokenizer = T5Tokenizer.from_pretrained('./../model/t5-small-gcp'  +  '/tokenizer')
def infer(text):
    stt_time = time.time()
    inputs = tokenizer.encode( text,return_tensors='pt',max_length=256, padding='max_length',truncation=True    )
    corrected_ids = trained_model.generate( inputs, max_length=256, num_beams=2, early_stopping=True )

    corrected_sentence = tokenizer.decode( corrected_ids[0],skip_special_tokens = True )
    print("Took " , time.time() - stt_time , " secs for inference")
    return corrected_sentence


 
app = FastAPI()



templates = Jinja2Templates(directory="templates")


@app.get("/pred/")
def main(gesture: str):
    return {"pred": infer(gesture)}

app.mount("/static", StaticFiles(directory="build"), name="build")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 
