import requests
import numpy as np
import json
import cv2

img = cv2.imread("test_img.jpg", 0)
img = img/255.0
img = np.expand_dims(img, 0)
img = np.expand_dims(img, -1)

# Make a request to serving container
headers = {'content-type': 'application/json'}

data = json.dumps({
    'signature_name': 'serving_default',
    'instances': img.tolist()
})

json_res = requests.post('http://0.0.0.0:8501/v1/models/vggmodel:predict', data=data, headers=headers)
json_res = json.loads(json_res.text)
result_img = json_res['predictions'][0]
for idx, class_prob in enumerate(result_img):
    print("Class {}: {}%".format(idx, round(class_prob*100, 4)))