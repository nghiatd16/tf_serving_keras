DIR_PATH=$(pwd)

docker run -p 8501:8501 -p 8500:8500 \
           --mount type=bind,source=$DIR_PATH/vggmodel,target=/models/vggmodel \
           -e MODEL_NAME=vggmodel --name vggmodel-tfserving -t tensorflow/serving
