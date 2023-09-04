> **Disclaimer:** The coding exercises and all other contents for the Computer Vision course are the intellectual property of [Prof. Dr. Bernhard Egger](https://www.lgdv.tf.fau.de/person/bernhard-egger/), [Prof. Dr.-Ing. habil. Andreas Maier](https://lme.tf.fau.de/person/maier/), and [Prof. Dr. Tim Weyrich](https://www.phil.fau.de/fakultaet/gremien-kommissionen/univisid/42196854/) at [FAU Erlangen-Nürnberg](https://www.fau.eu/). Please be aware that copying content from here holds you accountable.

# FAU - Computer Vision Exercises (SS23)

Welcome to the Computer Vision repository for the Summer'23 semester at [Friedrich-Alexander University Erlangen-Nürnberg](https://www.fau.eu/). This repository contains programming exercises of the Computer Vision course, taught by [Prof. Dr. Bernhard Egger](https://www.lgdv.tf.fau.de/person/bernhard-egger/), [Prof. Dr.-Ing. habil. Andreas Maier](https://lme.tf.fau.de/person/maier/), and [Prof. Dr. Tim Weyrich](https://www.phil.fau.de/fakultaet/gremien-kommissionen/univisid/42196854/) at FAU Erlangen-Nürnberg.

## Environment Setup

Follow these steps to set up the environment:

1. Install [Python 3.x](https://www.python.org/).
2. Download and install [Anaconda](https://www.anaconda.com/download) Distribution on your machine.
3. Create and activate a conda environment (optional: choose a specific python version, in my case I used python=3.9).
```bash
conda create -n <env_name> python=<version>
conda activate <env_name>
```
4. Install the required packages from the respective channels.
```bash
conda install -c conda-forge opencv numpy pytest line_profiler
```
5. To run test scripts, use the following commands.
```bash
python3 bin/run_ex0.py
```
6. To execute test scripts, run the following commands.
```bash
pytest test/test_ex0.py
```

Feel free to explore the intriguing world of Computer Vision through these exercises!