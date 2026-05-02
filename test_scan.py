import requests
import numpy as np

# Example: create a features array
# Replace 100 with the number of features your model uses
features = np.zeros(215).tolist()

# Send POST request to Flask API
resp = requests.post("http://107.22.107.26:5000/scan", json={"features": features})
print(resp.json())

# Print response
print(resp.json())