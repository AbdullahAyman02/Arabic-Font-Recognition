import requests

# Open the image file in binary mode
# NOTE: This is an absolute path, change it depending on your computer


with open('../data/train/IBM Plex Sans Arabic/0.jpeg', "rb") as file:
# with open('../data/train/Lemonada/0.jpeg', "rb") as file:
# with open('../data/train/Marhey/0.jpeg', "rb") as file:
# with open('../data/train/Scheherazade New/0.jpeg', "rb") as file:
    # Send a POST request to the server
    response = requests.post("http://4.233.223.91/predict/", files={"file": file})
    # response = requests.post("http://127.0.0.1:8000/predict/", files={"file": file})
    
# Print the response
print(response.json())
