# Research Assignment - Computer Vision Lab, TU Delft

## What is the repository about?
Code repository for the Research Assignment (RA) I carried out together with the Computer Vision Lab in TU Delft from the beginning of September to mid-November. 
The RA had the following objectives:
- A first introductory literature review whose focus will be on the different training and encoding strategies, as well as the different neuron models deployed in state-of-the-art SNNs.
- The implementation and training of a Spiking Neural Network using the library SNNTorch for a simple Computer Vision task.
- An analysis of the previous implemented model focused on exploring the dependencies between the neuron model (e.g. Leaky vs non-Leaky neurons) and the network’s performance and behaviour.

## How to use the scripts locally?
First of all clone the repository locally:
```bash
$ git clone git@github.com:LucaFrat/Intern_SNN.git
```

Create virtualenv through your command line:
```bash 
# make sure to use python >= 3.8
$ virtualenv -p python3 venv
# activate venv
$ source venv/bin/activate
```

Install the snnTorch library, which is the library used to model the Spiking Neural Network utilised during the RA:
```bash
$ pip install snntorch
```

Then open the  `main_Fashion.py`  file in you preferred code editor, and decide whether you want to run the main function corresponding to the Betas experiment (in this case run the 'main_Fashion_betas' function) or the main function corresponding to the Slopes experiment (in this case run the 'main_Fashion_surr_coeff' function).

Before running the chosen script assure that the folder you want to save the files in is correctly specified in the code (if you change the name of the folders you want to save the metrics from the run, remember to change them in the "Visualise" file you are gonna use later on).

Once the script has completed its execution, move to the corresponding .ipynb file to visualize the results.