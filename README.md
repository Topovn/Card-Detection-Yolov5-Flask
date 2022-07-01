# Yolov5 object detection model deployment using flask
https://github.com/ultralytics/yolov5 
https://pytorch.org/hub/ultralytics_yolov5/
https://flask.palletsprojects.com/en/1.1.x/

## Web app
Simple app consisting of a form where you can upload an image, and see the inference result of the model in the browser. 
Run:
`$python webapp.py`
then visit http://localhost:5000/ in your browser

## Run & Develop locally
Run locally and dev (use cmd):
* `python -m venv venv`
* `conda activate venv`
* `(venv) $ pip install -r requirements.txt`
* `(venv) $ python webapp.py`


