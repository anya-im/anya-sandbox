import os
import logging
import argparse
import uvicorn
import romkan
from anyasand.converter import Converter
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger("anya-web")

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-m', '--model_path', help='dnn model path', default="./anya.mdl")
arg_parser.add_argument('-d', '--db_path', help='dictionary DB path', default="./anya-dic.db")
arg_parser.add_argument('-p', '--port_num', help='port number', default=8080)
args = arg_parser.parse_args()
converter = Converter(args.model_path, args.db_path)

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.dirname(__file__)+"/web/static"), name="static")
templates = Jinja2Templates(directory=os.path.dirname(__file__)+"/web/templates")


@app.get("/", response_class=HTMLResponse)
async def anya_sand_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request
    })


@app.get('/convert', status_code=200)
async def convert(text: str):
    kana_text = romkan.to_hiragana(text)
    out_text = converter.convert(kana_text)
    return {"convText": out_text}


def main():
    uvicorn.run(app, host="0.0.0.0", port=args.port_num)


if __name__ == "__main__":
    main()
