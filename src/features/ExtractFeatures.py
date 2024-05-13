from GaborFilter.GaborFilter import gabor_filter
from EnclosingArea.EnclosingArea import enclosing_area
import numpy as np
from pandas import DataFrame as pdDataFrame
import pandas as pd

def extract_features(image, *, labelled=False, label_class=None):
    
    # Extracting features
    columns = ["area_ratio"]
    for j in range(1 , 17):
        columns.append("gabor_mean_" + str(j))
        columns.append("gabor_std_" + str(j))
        columns.append("gabor_energy_" + str(j))

    if labelled:
        columns.append("font_type")

    features = pdDataFrame(columns=columns)

    area = enclosing_area(image)
    gabor = gabor_filter(image)

    if labelled:
        features = pd.concat([features , pd.DataFrame([np.concatenate([[area] , gabor , [label_class]])] , columns = columns)])
    else:
        features = pd.concat([features , pd.DataFrame([np.concatenate([[area] , gabor])] , columns = columns)])
    return features
