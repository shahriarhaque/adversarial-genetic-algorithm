# Genetic Algorithm to produce adversarial text sequences for binary text classifiers

## The Test Subject: A Binary Phishing Classifier

Please refer to [this link](https://github.com/shahriarhaque/themis#train-themis-model) for instructions on how to build the phishing classifier and its associated tokenizers.

## Fitness Function Setup

  1) Edit `simutils.py` and modify the following variables to point to the model and tokenizers obtained from the phishing classifier repository: `MODEL_DIRECTORY`, `MODEL_FILE`, `TOKEN_HW_FILE`, `TOKEN_HC_FILE`, `TOKEN_BW_FILE`, and `TOKEN_BC_FILE`.
  2) Modify `ga-reverse-engineer.py` by commenting either one of the following lines in the `evalOneMax` function. This decides whether the algorithm will evolve a highly phishy sentence or a minimally phishy sentence.
  
    # return sum(individual),
    return (1.0 - simutils.fitness(individual)),  
  
## Generate adversarial sentence using the GA.
  1) Run `ga-reverse-engineer.py`
  
## Inject adversarial sentence into a known phishing test-case
  1) Modify `sample-phish.txt` with the text of a known phishing email.
  2) Inject the adversarial output of the previous step anywhere in the file.
  3) Run `simutils.py` and observe the impact of the adversarial injection.
