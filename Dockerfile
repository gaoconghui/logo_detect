FROM ubuntu:14.04

#3.4.3
ENV PYTHON_VERSION 2.7
ENV NUM_CORES 4

# Install OpenCV 3.0
RUN perl -p -i -e "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list
RUN apt-get -y update
RUN apt-get -y install python$PYTHON_VERSION-dev wget unzip \
                       build-essential cmake git pkg-config libatlas-base-dev gfortran \
                       libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
                       libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install numpy matplotlib

RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O opencv3.zip && \
    unzip -q opencv3.zip && mv /opencv-3.1.0 /opencv
RUN wget https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip -O opencv_contrib3.zip && \
    unzip -q opencv_contrib3.zip && mv /opencv_contrib-3.1.0 /opencv_contrib
RUN mkdir /opencv/build
WORKDIR /opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make -j$NUM_CORES
RUN make install
RUN ldconfig

ADD ./logo_detect /root/logo_detect/logo_detect
ADD ./start.py /root/logo_detect/
ADD ./requirement.txt /root/logo_detect/
ADD ./start.ini /root/logo_detect/
RUN pip install -r /root/logo_detect/requirement.txt

# Define default command.
CMD ["uwsgi","/root/logo_detect/start.ini"]