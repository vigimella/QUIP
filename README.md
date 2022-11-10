# Quantum Neural Network

This repository enables the execution of a Deep Learning algorithm using a Quantum Neural Network.

## Usage
Create a shell file by entering the following parameters: dataset name, ephocs, batch size, learning rate and threshold.
```bash
#!/bin/bash

python3 main.py -d dataset_name -e epochs_name -b batch_size -r learning_rate -t threshold
```
After the .sh file is created, run the following command

```bash
docker-compose up --build
```
## Authors

- [@Vigimella](https://www.github.com/vigimella)
- [@FrancescoMercaldo](https://github.com/FrancescoMercaldo)
- [@Dack1010](https://github.com/Djack1010)

## Contributing

The authors would like to thank the 'Trust, Security and Privacy' research group within the [Institute of Informatics and Telematics](https://www.iit.cnr.it/) (CNR - Pisa, Italy), that support their researches.

In this code we built a Quantum Neural Network (QNN). It is similar to the approach used in [Farhi et al](https://arxiv.org/pdf/1802.06002.pdf)

In addition we were inspired by [MNIST classification](https://colab.research.google.com/github/tensorflow/quantum/blob/master/docs/tutorials/mnist.ipynb#scrollTo=udLObUVeGfTs)