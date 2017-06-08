from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
import os
from predictor import predictors

UPLOAD_FOLDER = '/Users/vipinrai/Developer/Python/venvs/keras_env/keras_scripts/static/uploads'

app = Flask(__name__)
Bootstrap(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data = request.form
        f=request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    file_src='static/uploads/'+filename
    age=int(data['age'])
    if age<50:
        risk='Low'
    else:
        risk='High'
    normal,alz = predictors(file_src)
    new_data={'age':age,'src':file_src,'risk':risk,'alz':alz, 'norm':normal}
    return render_template('upload.html',data=new_data)

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/models')
def models():
    return render_template('models.html')

if __name__=='__main__':
    app.run(debug=True)