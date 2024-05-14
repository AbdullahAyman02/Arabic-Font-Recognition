import requests

# Open the image file in binary mode
with open('../data/raw/fonts-dataset/IBM Plex Sans Arabic/5.jpeg', "rb") as file:
    # Send a POST request to the server
    response = requests.post("http://4.233.223.91/predict/", files={"file": file})

# Print the response
print(response.json())