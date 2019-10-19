1. List instances
```
gcloud compute instances list
```

2. describle instance
```
gcloud compute instances describe tf-dd2412-proj-exp-p100 --format="(scheduling.preemptible)" --zone=europe-west1-b
```

3. delete instance
```
gcloud compute instances delete tf-dd2412-proj-exp-p100 --zone=europe-west1-b
```

4. List deep learning realted img.
```bash
gcloud compute images list --project deeplearning-platform-release
#
gcloud compute images list --project deeplearning-platform-release --no-standard-images
```

5. Start instance
```
gcloud compute instances start <vm-name>
```

6. Stop instance
```
gcloud compute instances stop <vm-name>
```

7. SSH
```
gcloud compute ssh --zone=<zone> <vm-name>
```