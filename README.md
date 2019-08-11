# TF Keras + TensorFlow Serving + Docker
Using TF Keras to build model, training, and serving with TF Serving + Docker

### Install dependencies
```
pip3 install -r requirements.txt
```

### Install docker
#### Official document: https://docs.docker.com/ 
#### After instal successfuly, run command:
```
docker pull tensorflow/serving:latest
```

### Training and save model in servable SavedModel format
```
python3 vgg_mnist.py
```

### Deploy docker serving model - Ubuntu
```
sh run.sh
```

### Demo request via HTTP protocol
```
python3 request_api_demo.py
```
