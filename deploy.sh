#!/bin/bash

# gcloud compute scp --zone=europe-west1-b *.py tf2-exp-vm-vm:~/./
# gcloud compute scp --project dd2424-239818 --zone europe-west4-a --recurse *.py tensorflow-1-vm:~/
gcloud compute scp --zone=europe-west4-b *.py tf-t4-preemp2:/home/chrislin
