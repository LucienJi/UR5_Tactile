#!/bin/bash
# 获取最新镜像 ID，优先使用标签为 latest-amd64 的镜像
rosnode kill /husky_node

REPO="ripl/ur5-tactile"

# 检查是否提供了容器名称参数
if [ -z "$1" ]; then
  CONTAINER_NAME="tactile"
else
  CONTAINER_NAME="$1"
fi

# 尝试使用指定标签获取镜像 ID
IMAGE_ID=$(docker images --filter=reference="${REPO}:latest-amd64" --format '{{.ID}}')

# 如果未找到，则根据创建时间排序选择最新的镜像
if [ -z "$IMAGE_ID" ]; then
  IMAGE_ID=$(docker images "${REPO}" --format "{{.CreatedAt}}\t{{.ID}}" | sort -r | head -n1 | awk '{print $2}')
fi

if [ -z "$IMAGE_ID" ]; then
  echo "No image found for ${REPO}"
  exit 1
fi

echo "Using image ID: $IMAGE_ID"

# 使用 docker run 启动容器，将本地目录和设备挂载到容器中
docker run --rm --name "$CONTAINER_NAME" \
  --gpus all \
  --net=host \
  --device /dev/ttyACM0:/dev/ttyACM0 \
  -v "$(pwd)":/code/src/ur5-tactile \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env=DISPLAY=$DISPLAY \
  --env="ROS_MASTER_URI=http://192.168.131.11:11311" \
  --env="ROS_IP=192.168.131.245" \
  --env="ROS_HOSTNAME=192.168.131.245" \
  -it $IMAGE_ID bash -c "\
    echo 'export ROS_MASTER_URI=http://192.168.131.11:11311' >> ~/.bashrc; \
    echo 'export ROS_IP=192.168.131.245' >> ~/.bashrc; \
    echo 'export ROS_HOSTNAME=192.168.131.245' >> ~/.bashrc; \
    source ~/.bashrc; \
    exec bash"




