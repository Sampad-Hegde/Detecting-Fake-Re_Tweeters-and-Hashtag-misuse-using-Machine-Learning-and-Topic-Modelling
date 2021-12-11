from flask import Flask, request, jsonify
from Driver import get_all_data, extract_data_from_url, prepare_data_to_models
import warnings
import threading
from sys import path as syspath
from os import path as osPath, getcwd

warnings.filterwarnings("ignore")

syspath.insert(1, (osPath.dirname(getcwd()).replace('\\', '\\')) + '\\4_Model')
# noinspection PyUnresolvedReferences
from all_models_for_api import get_all_models, get_all_predictions
# noinspection PyUnresolvedReferences
from Combined_Models import CombinedClassifierModel
# noinspection PyUnresolvedReferences
from Temporal_models import TemporalClassifierModel
# noinspection PyUnresolvedReferences
from LDA import LDAClassifierModel

models = None

app = Flask(__name__, template_folder='template')

index_html = """

<html>

<title>
    Fake Re-tweeter Detector
</title>

<style>
    #body 
    {
        background-color: rgb(203, 230, 240); 
        align-items: center;
    }

    #main_box
    {
        margin: auto;
        width: 60%;
        border: 3px ;
        padding: 10px;
    }

    #url_textbox 
    {
        margin: auto;
        width: 75%;
        height: 5%;
        margin-left: 70px;
        
    }

    #submit_btn
    {
        margin: auto;
        width: 40%;
        vertical-align:middle;
        margin-left: 200px;
        height: 5%;
    }

    p{
        margin: auto;
        font-family: cursive;
        font-size:xx-large;
    }
    table, th, td 
    {
        padding: 5px;
        margin: auto;
        text-align: center;
        border: 1px solid black;
        border-collapse: collapse;
    }
    #res_table
    {
        margin: auto;
        margin-left: 220px;
        width: 60%;
        height: 5%;
    }
</style>



<body id="body">

<div id="main_box">

        <div>
            <h2><p>paste the url of tweet  :</p></h2>
        </br>
            <input type="text" id="url_textbox" placeholder="Paste the URL">

        </div>
    </br></br></br>
        <div>
            <button type="button" id="submit_btn">Submit</button>
        </div>
    </div>

    <div id="res_table"></div>

</body>
<script>

    var sub_but  = document.getElementById("submit_btn");
    var url_text = document.getElementById("url_textbox");
    var model_name = ["Hawkes Process and KNN", "Hawkes Process and SVM", "Hawkes Process and Naive Bayes", "Hawkes Process and FC NN", "LDA Topic Vecs and KNN", "LDA Topic Vecs and SVM", "LDA Topic Vecs and Naive Bayes", "LDA Topic Vecs and FC NN", "Bag of Words and LSTM NN", "Bag of Words and Bi-LSTM NN", "LDA + Hawkes and KNN", "LDA + Hawkes and SVM", "LDA + Hawkes and Naive Bayes", "LDA + Hawkes and FC NN"];

    sub_but.addEventListener("click", SendURLData);
    
    function SendURLData()
    {
        document.getElementById('res_table').innerHTML = '';
        var data = {url:url_text.value};
        
        var results = null;

        let headers = new Headers();

        headers.append('Content-Type', 'application/json');
        headers.append('Accept', 'application/json');

        headers.append('Access-Control-Allow-Origin', 'http://192.168.1.5:5000');
        headers.append('Access-Control-Allow-Credentials', 'true');

        headers.append('GET', 'POST', 'OPTIONS');


        fetch("http://192.168.1.5:5000/url", 
        {
            
            method: "POST", 
            body: JSON.stringify(data),
            timeout: 6000,
            headers: headers
        }
        ).then(response => console.log(response.json()
        .then(data => create_table(data['results']))));


    }
    
    
     function create_table(arr)
    {
       
        var table = document.createElement('table'), tr, td, row, cell;
        
        tr = document.createElement('tr');
        th = document.createElement('th');
        tr.appendChild(th);
        th.innerHTML = "Models";
        th = document.createElement('th');
        tr.appendChild(th);
        th.innerHTML = "Predicted";

        table.appendChild(tr);
        
        
        for (row = 0; row < 14; row++) 
        {
            tr = document.createElement('tr');
            for (cell = 0; cell < 2; cell++) 
            {
                if(cell == 0)
                {
                    td = document.createElement('td');
                    tr.appendChild(td);
                    td.innerHTML = model_name[row];
                }
                else
                {
                    td = document.createElement('td');
                    tr.appendChild(td);
                    td.innerHTML = arr[row];
                }
                
            }
            table.appendChild(tr);
        }
        document.getElementById('res_table').appendChild(table);
    }

</script>

</html>

"""


@app.route("/")
def mian_page():
    return index_html


@app.route("/url", methods=['POST'])
def pass_url_get_predictions():
    global models
    url = request.get_json(force=True)['url']




    user_df, tweets_df = extract_data_from_url(url)

    datas = prepare_data_to_models(user_df, tweets_df)
    preds = get_all_predictions(datas, models)

    res = []
    for i in preds:
        if i == 0:
            res.append('Real')
        else:
            res.append('Fake')
    return jsonify(results=res)


def run():
    # noinspection PyUnresolvedReferences
    from webapp import app
    app.run(debug=True)


with app.app_context():
    get_all_data()
    models = get_all_models()
    print("ALL models are trained : ")
