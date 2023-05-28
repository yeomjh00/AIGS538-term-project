# AIGS538-term-project 2023S
AIGS538-term-project

Member:
 - JaeHu Yeom, CSE
 - SeonAh Yoo, CSE

# Quick Guide
## Requirements
In fedlearn.yaml file, we export our conda environment. You can create the same environment by using the following command: `conda env create -f fedlearn.yaml`

## Instructions
`python main.py --aug_type=___ --functions=__`

options for aug_type = ( default=None, cutmix, saliencymix, original )

* default: no augmentation
* cutmix: [cutmix](https://arxiv.org/abs/1905.04899) augmentation. 
* saliencymix: [cut mix with saliency map](https://arxiv.org/abs/2006.01791) augmentation
* original: our suggested augmentation

options for functions = [train, test, attack, default=None]

* train: train model with augmentation option
* test: test model with augmentation option, metric=cross-entropy
* attack: from trained model, attack model for stealing images
* None: do all of above

More options(arguments) are described in args.py file. Please refer it!


# Brief Introduction to our project!

Our goal from this project is to improve privacy protection in federated learning system by data augmentation. To protect clients from gradient inversion attack, some systems are used to adopt data augmentation to increase batch size. 

However up-to-date augmentation methods are likely to use important, salient patch of data which may be vulnerable to privacy leekage by gradient inversion attack.

Thus, we propose uneven and weakly linear augmentation method to increase non-linearity, and global linearity to prevent gradient inversion attack.