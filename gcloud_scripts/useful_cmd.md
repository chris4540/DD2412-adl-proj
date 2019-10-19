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