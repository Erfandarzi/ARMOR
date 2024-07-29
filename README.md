# ARMOR:Robust Federated Learning and Adversarial Attacks

We formalize the notion of differential model robustness (DMR) under the federated learning (FL) context, and explore how can DMR be realized in concrete FL protocols based on deep neural networks (NNs).
We develop the differential model distribution (DMD) technique,
which distribute different NN models by noise-aided adversarial training.
This is a proof-of-concept implementation of our differential model distribution (DMD) technique.


## Experimental Tracking Platform
To report real-time result to wandb.com, please change wandb ID to your own. \
wandb login {YOUR_WANDB_API_KEY}

## Requirements
* Python 3.6
* Torch 1.10.1
* Numpy 1.19.5

## Experiment results

| Dataset | DMD Method | Noise  | Acc (%) | AATR (%) | ASR (%) 
| ------- | ---------- | ------ | ------- | -------- | ------- 
| MNIST | noise_only | 0.130 | 73.41 | 57.97 | 49.29
| MNIST | no_noise_adv | - | 92.65 | 81.46 | 54.10
| MNIST | **noise_adv** | **0.140** | **86.32** | **23.04** | **16.53**
| CIFAR-10 | noise_only | 0.045 | 51.16 | 68.48 | 52.22
| CIFAR-10 | no_noise_adv | - | 69.65 | 74.67 | 41.27
| CIFAR-10 | **noise_adv** | **0.040** | **56.26** | **29.23** | **17.20**

## Experiment Scripts

* To generate a set of noise-aided trained client models
(i.e., differential perturbation only technique), 
which is the basis of our following experiments, run
``` 
## MNIST
sh run_fed_train.sh 0 mnist 0.14 50
``` 
``` 
## CIFAR-10
sh run_fed_train.sh 0 cifar 0.04 50
``` 

* To test noise-aided differential adversarial training technique, run
``` 
## noise_adv
sh run_noise_adv.sh 0 mnist 0.14 50
``` 

* To test differential adversarial training only technique, run
``` 
## no_noise_adv
sh run_no_noise_adv.sh 0 mnist 0.14 50
``` 

* To test differential perturbation only technique, run
``` 
## noise_only
sh run_noise_only.sh 0 mnist 0.14 50
``` 

*  To analyze ATR and calculate AATR of above techniques, run
``` 
## noise_adv
sh run_process_attack_list_ATR.sh mnist 0.14 noise_adv 50
``` 
``` 
## no_noise_adv
sh run_process_attack_list_ATR.sh mnist 0.14 no_noise_adv 50
``` 
``` 
## noise_only
sh run_process_attack_list_ATR.sh mnist 0.14 noise_only 50
``` 

*  To analyze ATR, AATR and ASR of above techniques under different settings,
modify and run ```process_ATR.py``` and ```process_ASR.py``` to your need sequentially.

## Reference
[1] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[2] Papernot, Nicolas, et al. "Technical report on the cleverhans v2. 1.0 adversarial examples library." arXiv preprint arXiv:1610.00768 (2016).

[3] Wei, Kang, et al. "Federated learning with differential privacy: Algorithms and performance analysis." IEEE Transactions on Information Forensics and Security 15 (2020): 3454-3469.
