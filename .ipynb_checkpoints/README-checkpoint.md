# Domain Adaptation via Rebalanced Sub-domain Alignment (DARSA)

The repository contains Pytorch implementations for **DARSA**. 
 
 ## Structure of the Repository
The structure of this repository is given below:
- `data`: This directory contains all the datasets and data-related classes/functions used for experiments in the DARSA paper.
- `utils`: This directory contains all the utility classes/functions used for experiments in the DARSA paper.
- `models`: This directory contains all the network structures used for experiments in the DARSA paper.
- `experiments`: This directory contains all the scripts needed for replicating the experiments (both our method and benchmark methods) in the DARSA paper. <br />
Within the directory of each experiment: <br />
  - `DARSA_train_source.py` and `DARSA.py`: Our method (more details provided in Instructions)
  - `CDAN.py`: Conditional adversarial domain adaptation [1], served as a benchmark in DARSA paper
  - `DANN_train_source.py` and `DANN.py`: Domain-adversarial training of neural networks [2], served as a benchmark in DARSA paper
  - `wdgrl.py`: Wasserstein distance guided representation learning for domain adaptation [3], served as a benchmark in DARSA paper
  - `DSN.py`: Domain separation networks [4], served as a benchmark in DARSA paper
  - `adda.py`: Adversarial discriminative domain adaptation [5], served as a benchmark in DARSA paper
  - `pixelda.py`: Unsupervised pixel-level domain adaptation with generative adversarial networks [6], served as a benchmark in DARSA paper
  - `DRANet_trainer.py`, `DRANet_train.py` and `DRANet_test.py`: Dranet: Disentangling representation and adaptation networks for unsupervised cross-domain adaptation [7], served as a benchmark in DARSA paper
  - `source_only.py`: Train model on the source data and apply the model directly to the target data.

 ## Instructions
1. Download the [BSDS500 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) and extract it to a directory. Change the `path_to_BSDS_data` variable in `util/sconfig.py` to this directory.
 
2. In a Python 3.8.8 environment, train a model on the source dataset with
```
$ python ./experiments/*/DARSA_train_source.py
```
3. Load the pretrained network and run the algorithm, for example:
```
$ python ./experiments/*/DARSA.py
```

Note: The MNIST, BSDS500, USPS, and SVHN datasets are publicly available. The Tail Suspension Test (TST) dataset [8] is available by request from the authors of [8].

## Acknowledgment

Our implementation for benchmark methods are mainly based on (https://github.com/jvanvugt/pytorch-domain-adaptation, https://github.com/fungtion/DSN, https://github.com/thuml/CDAN, https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pixelda/pixelda.py, https://github.com/Seung-Hun-Lee/DRANet, https://github.com/thudzj/CAT (tensorflow)). Thanks for their authors.

## References
[1] Long, Mingsheng, et al. "Conditional adversarial domain adaptation." Advances in neural information processing systems 31 (2018). <br />
[2] Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The journal of machine learning research 17.1 (2016): 2096-2030. <br />
[3] Shen, Jian, et al. "Wasserstein distance guided representation learning for domain adaptation." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018. <br />
[4] Bousmalis, Konstantinos, et al. "Domain separation networks." Advances in neural information processing systems 29 (2016). <br />
[5] Tzeng, Eric, et al. "Adversarial discriminative domain adaptation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. <br />
[6] Bousmalis, Konstantinos, et al. "Unsupervised pixel-level domain adaptation with generative adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. <br />
[7] Lee, Seunghun, Sunghyun Cho, and Sunghoon Im. "Dranet: Disentangling representation and adaptation networks for unsupervised cross-domain adaptation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021. <br />
[8] Gallagher, Neil, et al. "Cross-spectral factor analysis." Advances in neural information processing systems 30 (2017). <br />