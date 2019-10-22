#!/bin/bash
# Example to show create from existing disk
# =================================================
# export IMAGE_FAMILY="tf-latest-gpu"
export ZONE="europe-west1-d"
export INSTANCE_NAME="t114-k80-dev"
export INSTANCE_TYPE="n1-highmem-2"
export DISK_NAME="tensorflow-114-vm"


# gcloud compute instances create $INSTANCE_NAME \
#   --machine-type=$INSTANCE_TYPE \
#   --zone=$ZONE \
#   --image-family=$IMAGE_FAMILY \
#   --image-project=deeplearning-platform-release \
#   --no-boot-disk-auto-delete \
#   --maintenance-policy=TERMINATE \
#   --preemptible

# Create from disk
gcloud compute instances create $INSTANCE_NAME \
  --machine-type=$INSTANCE_TYPE \
  --zone=$ZONE \
  --disk="name=${DISK_NAME},boot=yes,mode=rw" \
  --maintenance-policy=TERMINATE
  # --preemptible
