FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
 
# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

RUN git clone https://github.com/NVlabs/nvdiffrast /tmp/pip/
RUN cp /tmp/pip/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install --upgrade pip
RUN cd /tmp/pip && pip install .

COPY . .
RUN pip install -r requirements.txt

#copy 3DMM modules from Deep3DFaceRecon_pytorch
RUN git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch && git clone https://github.com/deepinsight/insightface.git
RUN cp -r insightface/recognition/arcface_torch/ Deep3DFaceRecon_pytorch/models/
RUN cp -rf 3DMM/* Deep3DFaceRecon_pytorch Deep3DFaceRecon_pytorch/
RUN mv Deep3DFaceRecon_pytorch model/

# delete useless files
RUN rm -rf 3DMM insightface docker-examples tutorials NVIDIA_Deep_Learning_Container_License.pdf /workspace/model/Deep3DFaceRecon_pytorch/util/__init__.py