FROM tensorflow/tensorflow:latest-py3
RUN apt-get update
RUN apt-get -y install git libopencv-dev cmake-qt-gui sudo wget

RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN rm get-pip.py

RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose /openpose
WORKDIR /openpose

RUN wget -c "https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb"
RUN sudo dpkg --install cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
RUN sudo apt-get update
RUN sudo apt-get -y install cuda

RUN sudo ubuntu/install_cudnn.sh
RUN sudo ubuntu/install_cmake.sh

RUN mkdir build
RUN cd build && cmake -D GPU_MODE=CPU_ONLY ..
#RUN cd build && cmake ..
RUN cd build && make -j `nproc`
RUN cd build && sudo make install


CMD /bin/bash
