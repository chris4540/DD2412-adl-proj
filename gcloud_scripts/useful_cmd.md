0. Check default project
```bash
gcloud config get-value project
```

1. List projects
```bash
gcloud projects list
```

1. Change default project
```bash
gcloud config set project <PROJECT ID>
```

1. List instances
```bash
gcloud compute instances list
```

2. describle instance
```bash
gcloud compute instances describe tf-dd2412-proj-exp-p100 --format="(scheduling.preemptible)" --zone=europe-west1-b
```

3. delete instance
```bash
gcloud compute instances delete tf-dd2412-proj-exp-p100 --zone=europe-west1-b
```

4. List deep learning realted img.
```bash
gcloud compute images list --project deeplearning-platform-release
#
gcloud compute images list --project deeplearning-platform-release --no-standard-images
```

5. Start instance
```bash
gcloud compute instances start <vm-name>
```

6. Stop instance
```bash
gcloud compute instances stop <vm-name>
```

7. SSH
```bash
gcloud compute ssh --zone=<zone> <vm-name>
```