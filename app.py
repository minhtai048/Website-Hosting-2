# https://youtu.be/l3QVYnMD128
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 
Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.
"""


import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)
#Load the trained model and encoder. (Pickle file)
model = pickle.load(open('models/lung_cancer_model.pkl', 'rb'))
encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output
@app.route('/predict',methods=['POST']) #Do not add two method may make app crashed
def predict():
    #input section
    genderinput = request.form.get('genderinput')
    smokinginput = request.form.get('smokinginput')
    yellowinput = request.form.get('yellowinput')
    anxietyinput = request.form.get('anxietyinput')
    peerinput = request.form.get('peerinput')
    chronicinput = request.form.get('chronicinput')
    fatigueinput = request.form.get('fatigueinput')
    allergyinput = request.form.get('allergyinput')
    wheezinginput = request.form.get('wheezinginput')
    alcoholinput = request.form.get('alcoholinput')
    coughinginput = request.form.get('coughinginput')
    swallowinginput = request.form.get('swallowinginput')
    chestinput = request.form.get('chestinput')
    gender_display = genderinput
    
    #feature transformation section
    features = pd.DataFrame({"GENDER":[genderinput], "SMOKING":[smokinginput], "YELLOW_FINGERS":[yellowinput], "ANXIETY":[anxietyinput],
                             "PEER_PRESSURE":[peerinput], "CHRONIC DISEASE":[chronicinput], "FATIGUE":[fatigueinput], 
                             "ALLERGY":[allergyinput], "WHEEZING":[wheezinginput], "ALCOHOL CONSUMING":[alcoholinput],
                             "COUGHING":[coughinginput], "SWALLOWING DIFFICULTY":[swallowinginput], "CHEST PAIN":[chestinput]})
    features = features.replace('YES', 1)
    features = features.replace('NO', 0)
    features['GENDER'] = encoder.transform(features['GENDER'])
    prediction = model.predict(features)
    if(prediction == 0):
        output = 'NO'
    else:
        output = 'YES'

    #final section -> send data back to front page
    if(prediction == 0):
        return render_template('index.html', gender=gender_display, smoking=smokinginput, yellow=yellowinput, anxiety=anxietyinput,
                           peer=peerinput, chronic=chronicinput, fatigue=fatigueinput, allergy=allergyinput, wheezing=wheezinginput,
                           alcohol=alcoholinput, coughing=coughinginput, swallowing=swallowinginput, chest=chestinput,
                           prediction_text='you seem to do not have lung cancer as model predicted {}'.format(output))
    else:
        return render_template('index.html', gender=gender_display, smoking=smokinginput, yellow=yellowinput, anxiety=anxietyinput,
                           peer=peerinput, chronic=chronicinput, fatigue=fatigueinput, allergy=allergyinput, wheezing=wheezinginput,
                           alcohol=alcoholinput, coughing=coughinginput, swallowing=swallowinginput, chest=chestinput,
                           prediction_text='you are likely to have lung cancer as model predicted {}'.format(output))

#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()