# WSDS - Word Sense Disambiguation via Siamese Network
This repo contains the code for the second homework of the NLP 2023 course at Sapienza University of Rome.

I implemented a Siamese network trained via contrastive learning, to tackle the Word Sense Disambiguation problem.

I also implemented the paper [BEM](https://github.com/facebookresearch/wsd-biencoders) from Meta, and confronted the performances.

A detailed report can be found in [PDF](https://github.com/Andreus00/WSDS/blob/main/nlp2023_hw2_v2.pdf)


## Requirements

* Ubuntu distribution
  * Either 20.04 or the current LTS (22.04) are perfectly fine.
  * If you do not have it installed, please use a virtual machine (or install it as your secondary OS). Plenty of tutorials online for this part.
* [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community.

## Notes

Unless otherwise stated, all commands here are expected to be run from the root directory of this project.

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client
  
### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding.
For those who might be unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependencies to run. We will be using conda to create the environment for this client.

```bash
conda create -n nlp2023-hw2 python=3.9
conda activate nlp2023-hw2
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp2023-hw2
bash test.sh data/coarse-grained/test_coarse_grained.json
```

