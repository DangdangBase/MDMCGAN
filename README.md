# Multi-Discriminator Multi-Conditional Generative Adversarial Network (MDMCGAN)
### Baseline and Comparison target
- [Pytorch Conditional WGAN with Gradient Penalty](https://github.com/gcucurull/cond-wgan-gp)

### How to run
```
python mdmcgan.py
```
It will generate the log file 'mdmcgan_result.csv' which contains workload and loss of generater G and discriminator D


### Dependencies
- python==3.8.18
- torch==2.1.2
- torchvision==0.16.2
- and others in requirements.txt