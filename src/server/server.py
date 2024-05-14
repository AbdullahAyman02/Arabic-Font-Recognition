import numpy as np 
import pandas as pd
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from fastapi import Request
import skimage as ski
from PreProcessing import PreProcessing
from ExtractFeatures import extract_features, normalize , generate_kernels
import time
import skimage as ski
import numpy as np
from sklearn.preprocessing import StandardScaler



kernels = generate_kernels()

x_data = pd.read_csv("./features_1000_glcm.csv")
x_data = x_data.drop(columns=['Unnamed: 0'])
# unique_counts = dataset.nunique()
# columns_to_drop = unique_counts[unique_counts == 1].index
# x_data = dataset.drop(columns=columns_to_drop)
x_data = x_data.iloc[:, :-1]
scaler = StandardScaler()
scaler.fit(x_data)


columns = x_data.columns
print("coloumns " , columns)

print("x_data" , x_data)

print("shape" ,x_data.shape)


model = pickle.load(open("svm3.pkl","rb"))
app = FastAPI() 
# img 
# {time :  result: }
@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    try : 
        print("image is received")
        image = ski.io.imread(file.file)
        print("image is read " , image ) 

        start_time = time.time() 
        print("start_time " , start_time) 
        image_processed = PreProcessing(image)
        

        print("img_processed" , image_processed) 
        features = extract_features(image_processed,kernels)

        print("features before reshape" , features)
        print("feature shape " , features.shape)
        print ("feature type " , type(features))
        features = features.reshape(1,-1)
        print("feature : after reshape " , features)

        df = pd.DataFrame(features, columns=columns)
        features = scaler.transform(df)

        print("feature type " , type(features))
        print("features after normalization" , features)

        result = model.predict(features)
        print("result is predicted" , result)

        end_time = time.time()
        print("end time " , end_time)   
        time_taken = end_time - start_time

        print("time taken to predict the image is ", end_time - start_time)
        print("result is ", result) 

        return JSONResponse(content={"result":str(result) , "time" : time_taken},status_code=200)
    except Exception as e:
        return JSONResponse(content={"error":str(e)},status_code=500)

