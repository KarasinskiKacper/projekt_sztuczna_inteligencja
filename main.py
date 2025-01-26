from flask import Flask, render_template, request, make_response
from PIL import Image
import os

from wykrywanie_tekstu.load_net import load_net
from wykrywanie_tekstu.split_img import split_img

from test_model import test_model
from model_kacper_k import model_kacper_k
from model_mikolaj_c import model_mikolaj_c
from model_pawel_h import model_pawel_h

split_img_model = load_net()

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the Flask app and specify the template folder
app = Flask(__name__, template_folder = current_dir)

@app.route('/', methods=['GET',"POST"])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        print(request.form["model"])
        
        full_img = Image.open(request.files['image'])
        imgs = split_img(split_img_model, full_img)
        
        res_text = ''
        for img in imgs:
            if request.form["model"] == "1":
                res_text += model_kacper_k(img) + ' '
                print(res_text)
            elif request.form["model"] == "2":
                res_text += model_pawel_h(img) + ' '
                print(res_text)
            elif request.form["model"] == "3":
                pass
                # res_text += model_mikolaj_c(img) + ' '
                # print(res_text)
            elif request.form["model"] == "4":
                res_text += model_mikolaj_c(img) + ' '
                print(res_text)
                
        
        response = make_response(res_text)
        # response = make_response(res_text)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Access-Control-Allow-Origin'] = '*'
        
        return response

if __name__ == '__main__':
    app.run(debug=True)