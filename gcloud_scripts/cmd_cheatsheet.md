- Login
```bash
gcloud auth login
```
- Check default project
```bash
gcloud config get-value project
```

- List projects
```bash
gcloud projects list
```

- Change default project
```bash
gcloud config set project <PROJECT ID>
```

- List instances
```bash
gcloud compute instances list
```

- describle instance
```bash
gcloud compute instances describe tf-dd2412-proj-exp-p100 --format="(scheduling.preemptible)" --zone=europe-west1-b
```

- delete instance
```bash
gcloud compute instances delete tf-dd2412-proj-exp-p100 --zone=europe-west1-b
```

- List deep learning realted img.
```bash
gcloud compute images list --project deeplearning-platform-release
#
gcloud compute images list --project deeplearning-platform-release --no-standard-images
```

- Start instance
```bash
gcloud compute instances start <vm-name>
```

- Stop instance
```bash
gcloud compute instances stop <vm-name>
```

- SSH
```bash
gcloud compute ssh --zone=<zone> <vm-name>
```
-------------------------------------------------------------------
## Copying from / to google cloud storage

- List the bluckets
```bash
gsutil ls gs://dd2412-proj-exp-data/exp1
```

- Copy folder
```bash
gsutil cp -r wrn-16-2-seed10 gs://dd2412-proj-exp-data/exp1
```

- Rsync directory
```bash
gsutil rsync -r exp/exp1 gs://dd2412-proj-exp-data/exp1
```

- Rsync file
```bash
gsutil rsync data gs://mybucket/data
```
