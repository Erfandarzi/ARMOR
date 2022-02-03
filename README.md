# ARMOR: Differential Model Distribution for Robust Federated Learning Against Adversarial Attacks

In this work, we formalize the concept of differential model robustness,
a new property for ensuring model security in federated learning (FL) systems. 
We point out that for most conventional FL frameworks, 
all clients receive the same global model at the end of each communication round.
Our observation here is that, if there exists a malicious client who
is able to generate powerful adversarial samples against one model,
the attack will be immediately transferable to all of the other benign clients.

To the best of our knowledge, we are the first to
define the notion of differential model robustness against white-box adversarial 
attacks to address the attack transferability concern in FL systems.
In addition, we propose a differential model distribution 
technique that perturbs and enhances the global model in a differential way and
sends each (potentially malicious) client different local models
that is derived from the global. In this way, we
are able to prevent adversarial samples generated on one model
from transferring to other models while retaining model utility. 
To better measure the attack and defense performance, we also propose new notions
such as average adversarial transfer rate (AATR) and differential model robustness.
Through extensive experiments on the MNIST
and CIFAR-10 datasets, we demonstrate that the ARMOR can significantly reduce both 
the ASR and AATR across different FL settings.
For example, the ASR and AATR are reduced by nearly 5/6 (100\% to 16.53\%) 
and 3/4 (100\% to 23.04\%) over the MNIST dataset, respectively, 
and by nearly 5/6 (100\% to 17.20\%) and 2/3 (100\% to 29.23\%) over the CIFAR-10 dataset, 
respectively, in a 50-client FL system.


## Experimental Tracking Platform
To report real-time result to wandb.com, please change wandb ID to your own. \
wandb login {YOUR_WANDB_API_KEY}

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
