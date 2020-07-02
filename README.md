# machine_learning
### Machine learning algorithms and experiments with Python :robot:

Inspriation for the code in this repository comes from many places, including but not limited to:
1) Machine Learning in Python - Bowles
2) Machine Learning with Scikit-learn and Tensorflow - Geron
3) Deep Learning with Python - Chollet
4) Python Data Science Handbook - VanderPlas
5) Machine Learning with Python - Muller & Guido

I highly recommend all of these books.

### Practice notebooks

I have included template notebooks for various machine learning projects with code blocks omitted, along with the fully functional versions for comparison. I find these very useful for practicing frameworks, and committing various techniques to memory. 

### Jupyter notebook with Tensorflow via Docker container 
##### Requirements:
###### 1) Nvidia GPU and up-to-date driver installed

##### Install Docker:

`wget -qO- https://get.docker.com/ | sh`

##### Add non-root user to Unix group:

`sudo usermod -aG docker your-name`

##### Install Nvidia-Docker support:

`distribution=$(. /etc/os-release;echo $ID$VERSION_ID)`

`curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -`

`curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list`

`sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit`

`sudo systemctl restart docker`

##### Pull desired Docker image:

`docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter`

##### Build & Run Docker container (first time):

`docker run --gpus all -it -p 8888:8888 -p 6006:6006 -v /$(pwd)/tensorflow:/notebooks --name tf tensorflow/tensorflow:latest-gpu-py3-jupyter`

###### *This command sets up network instructions, creates a local directory to keep saved notebook data, builds the container and saves it as "tf" on the host machine.*

##### Escape to Host terminal

`ctrl pq`

###### *-it flags specify an interactive terminal session*

##### Stop Docker container:

`docker container stop tf`

##### Start Docker container:

`docker start -i tf`

###### Open browser window to URL returned at cmd line to access the Jupyter notebook!
