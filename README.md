# tcrLM: A large language model for predicting T cell receptor-antigen binding specificity 
tcrLM is an advanced computational model that predicts the binding specificity of antigens to TCR molecules. By using a large language model as the model encoder, tcrLM provides a comprehensive evaluation of antigen immunogenicity, which is crucial for the development of effective immunotherapies

## Key features
- **Large language model**: Leveraged the powerful semantic understanding capabilities of large language models.
- **Virtual adversarial training**: Enhances model robustness by training on perturbed data.
- **Superior performance**: Outperforms existing methods on pTCR prediction tasks on multiple datasets.

For inquiries or collaborations, please contact: hliu@njtech.edu.cn

## System requirements
- **Linux version**: 4.18.0-193 (Centos confirmed)
- **GPU**: NVIDIA GeForce RTX 4090 (or compatible GPUs)
- **CUDA Version**: 12.4
- **Python**: 3.10
- **PyTorch**: 2.2.1 (model implementation)

## Installation guide
>1. Clone the tcrLM repository

` git clone https://github.com/hliulab/tcrLM.git`

>2. Enter tcrLM project folder

` cd tcrLM/`

>3. Set up the Python environment and install the required packages
   
` pip install -r requirements.txt `

## Instructions for use
The training data for pTCR bindings is stored in the <kbd>data</kbd> folder. Both the training and testing scripts are included in the <kbd>source</kbd> folder.The source code of tcrLM model is included in the <kbd>models</kbd> folder.The trained models are stored in the <kbd>trained_model</kbd> folder.

### Input data format
For pretraining, the input data should be a CSV file with two columns: `sequence`, which represents the input sequence, and `label`, which indicates the type of sequence. For fine-tuning, the data should also be in CSV format, with two columns: `tcr`, representing the TCR CDR3 sequence, and `peptide`, representing the peptide sequence.

### Model training
After ensuring that the required environment and dependencies are installed, you can execute the following commands for fine-tuning and pretraining.
>You can run the following command to pre-train the model
`cd source`

`deepspeed pretrain.py --deepspeed_config ./deepspeed.json`

>You can run the following command to fine-tune the model

`cd source`

`deepspeed fine_tune.py --deepspeed_config ./deepspeed.json`

### Model testing
Given your fine-tuned model or our trained model (saved in trained_model folder), you can evaluate it on our provided demo test set using the following test scripts.

`cd source`

`python TCR_test.py`


### Hyperparameter adjustment
If transfer the model using your custom dataset, you may need to adjust the hyperparameters within the Python scripts. Hyperparameters include learning rate, batch size, number of epochs, and other model-specific parameters.

Note: Ensure that the file paths and script names provided in the commands match those in your project directory. The source/ directory and script names like TCR_test.py are placeholders and should be replaced with the actual paths and filenames used in your implementation.

### Customizing output
To customize the output results, users can modify the parameters within each script. Detailed comments within the code provide descriptions and guidance for parameter adjustments.

## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/tcrLM).

---

Please replace the placeholder links and information with actual data when the repository is available. Ensure that the instructions are clear and that the repository contains the `requirements.txt` file with all necessary dependencies listed.
