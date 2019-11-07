# Advance Deep Learning Reproducibility Project
This is the course DD2412 final project aiming for reproduce

## Reproducibility

### Targets
- Reimplementation of all methods from scratch and reproducing, empirically
  analyzing, discussing (nearly) all results.
- Use different model teacher and student pair model to run key results.
  E.g. VGG-16, VGG-19.
- [Bonus point] Successful reimplementation in another deep learning framework.

## Method implemented
- [x] Reimplement WRN 
- [ ] Toy model Experiment. (Figure 1)  
- [x] Zero-shot transfer with adversarial training method  
- [x] Knowledge distillation and attention transfer (KD only has soft target)  
- [ ]

## Key results
- [ ] TODO 

## Reproducibility Checklist
[ReproducibilityChecklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)

## Report
Example reports on ReScience

https://rescience.github.io/read/

Selected paper:
[Re h-detach: Modifying the LSTM gradient towards better optimization](https://zenodo.org/record/3162114/files/article.pdf)

## Paper-only vs author implementation
1. [Small issue] Paper use L2-norm while their impl. uses (L2-norm)**2
2. [Big issue] Cannot find cosine annealing setting
3. [Small issue] Structure difference between Paul Micaelli and Sergey Zagoruyko

## Platform
Google cloud VM
```bash
image-name: tf-1-14-cu100-20191004
image-family: tf-1-14-cu100
image-project: deeplearning-platform-release
accelerator: type=nvidia-tesla-p100,count=1
```
---------------------------------------------------
## Experiment details
### Experiment 1
- Reproduce the result in table 1. 
- Iterate each step for 3 times with seed

#### Checklist
1. [ ] Train teacher from scratch
2. [ ] Train student from scratch
3. [ ] Samples 200 images per class
4. [ ] Zero-shot learning


---------------------------------------------------
## Technical details
### Software Requirement
Tensorflow v1.14; Capable to v1.15 or v2.0 but no testing
Python v3.4 or above

##### Cosine anneualing
[Small intro on blog](
https://towardsdatascience.com/https-medium-com-reina-wang-tw-stochastic-gradient-descent-with-restarts-5f511975163)

##### No bias in conv layer
From [Resnet author Kaiming He](https://github.com/a-martyn/resnet/blob/master/resnet.py)

##### See how to fix the random number seed accoss keras, tensorflow, and numpy
https://machinelearningmastery.com/reproducible-results-neural-networks-keras/

##### Save model in tf
https://www.tensorflow.org/guide/keras/save_and_serialize

##### Autodiff in tf
https://www.tensorflow.org/tutorials/customization/autodiff

##### New method for developing
https://www.tensorflow.org/guide/keras/custom_layers_and_models

##### PEP-8: Style Guide for Python Code
https://www.python.org/dev/peps/pep-0008/
-----------------------------------------------------------------
## Missing info
1. batch size
2. Consine anneuling info
3. How to train teacher with high acc
