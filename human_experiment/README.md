# Human experiment

This folder contains PsychoPy code for replicating the human experiment described in the main paper.

## Installation

The code was tested with PsychoPy version 2024.2.5 on macOS Sonoma 14.7.4.

### Official installer

PsychoPy developers provide an official installer, but it did not work for me.

### Manual installation

First install python3.10 (if it is not already in the system)

Download the source from https://www.python.org/downloads/release/python-31016/ and run the following:

```
./configure
make
make test
sudo make install
```

Then install PsychoPy:

```
pip3 install psychopy==2024.2.5
pip3 install psychopy-visionscience
```

## Running the experiment

This command will run the experiment for one subject:

```
python3 human_experiment/run_experiment.py <subject_id>
```

Subject_id is a numeric id assigned to the subject, e.g. `python3 human_experiment/run_experiment.py 10`

### Setup

The script expects a monitor (change settings in line 40 of the experiment script), a keyboard, and a mouse.

### Data

The script uses `data/exp_samples.xlsx`, which contains 200 samples for 100 participants generated from the SNAP data.

### Steps of the experiment

1. Prompts to enter gender and age information are shown (can be skipped);

2. Instructions are shown;

3. Practice session for 2 questions: one open-ended, one multi-choice;

4. Main session consisting of 4 blocks.

### Saving the results

The script will save progress in the cache automatically, so if the experiment is interrupted, restarting the script will not load previously seen images.

Gender/age information is saved as a text file with subject's id, e.g. `data/subject_1.txt`.

The answers are saved in excel files for each subject, e.g. `data/subject_1.xlsx`.

Practice session results are not saved.