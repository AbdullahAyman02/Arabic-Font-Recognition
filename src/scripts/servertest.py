import requests

# Open the image file in binary mode
# NOTE: This is an absolute path, change it depending on your computer
with open('../data/raw/fonts-dataset/IBM Plex Sans Arabic/5.jpeg', "rb") as file:
    # Send a POST request to the server
    response = requests.post(
        "http://localhost:8000/predict/", files={"file": file})

# Print the response
print(response.json())
