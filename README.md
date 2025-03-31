# tcrLM: A lightweight language model for predicting T cell receptor-antigen binding specificity 
tcrLM introduces a lightweight BERT-based masked language model pretrained on the largest TCR CDR3 sequence dataset (100M+), incorporating virtual adversarial training to enhance robustness, and achieves state-of-the-art performance in TCR-antigen binding prediction while demonstrating clinical relevance through correlations with immunotherapy outcomes.

## Key features
- **Large language model**: : tcrLM introduces a BERT-based masked language model pretrained on over 100 million distinct TCR CDR3 sequences.
- **Virtual adversarial training**: tcrLM incorporates virtual adversarial training and novel positional encoding (RoPE).
- **Superior performance**: tcrLM achieves superior generalizability across diverse test sets, outperforms existing methods and larger protein language models.

For inquiries or collaborations, please contact: hliu@njtech.edu.cn

## System requirements
- **Linux version**: 4.18.0-193 (Centos confirmed)
- **GPU**: NVIDIA GeForce RTX 4090 (or compatible GPUs)
- **CUDA Version**: 12.4
- **Python**: 3.10
- **PyTorch**: 2.4.1 (model implementation)

## Installation guide
>1. Clone the tcrLM repository

`git clone https://github.com/hliulab/tcrLM.git`

>2. Enter tcrLM project folder

`cd tcrLM/`

>3. Create a conda environment

`conda create -n tcrLM python=3.10 `

>4. Activate the conda environment

`conda activate tcrLM `

>5. Install environments
`conda env update -n tcrLM -f environments.yaml`

>6. Install PyTorch

`pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118 `

## Instructions for use
The training data for pTCR bindings is stored in the <kbd>data</kbd> folder. Both the training and testing scripts are included in the <kbd>source</kbd> folder.The source code of tcrLM model is included in the <kbd>models</kbd> folder.The trained models are stored in the <kbd>trained_model</kbd> folder.

### Input data format
For pretraining, the input data should be a CSV file with two columns: `sequence`, which represents the input sequence, and `label`, which indicates the type of sequence. For fine-tuning, the data should also be in CSV format, with two columns: `tcr` representing the TCR CDR3 sequence, and `peptide` representing the peptide sequence.

### Model training
After ensuring that the required environment and dependencies are installed, you can execute the following commands for fine-tuning and pretraining.
>Running following command to pre-train the model
`cd source`

`python pretrain.py`

>Running the following command to fine-tune the model

`cd source`

`python fine_tune.py`

### Model testing
Given your fine-tuned model or our trained model (saved in trained_model folder), you can evaluate it on our provided demo test set using the following test scripts.

`cd source`

`python TCR_test.py`

### Testing with Docker
You can also run our code using Docker, which has been verified for compatibility in a Windows environment.
Start the Docker container:
```bash
docker pull diaoxing359/tcrlm
docker run -it tcrlm:latest /bin/bash
```

​Inside the container, execute:
```bash
conda activate tcrLM
python TCR_test.py
```


### Hyperparameter adjustment
If transfer the model using your custom dataset, you may need to adjust the hyperparameters within the Python scripts. Hyperparameters include learning rate, batch size, number of epochs, and other model-specific parameters.

Note: Ensure that the file paths and script names provided in the commands match those in your project directory. The source/ directory and script names like TCR_test.py are placeholders and should be replaced with the actual paths and filenames used in your implementation.

### Customizing output
To customize the output results, users can modify the parameters within each script. Detailed comments within the code provide descriptions and guidance for parameter adjustments.

## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/tcrLM).

---

Please replace the placeholder links and information with actual data when the repository is available. Ensure that the instructions are clear and that the repository contains the `environments.yaml` file with all necessary dependencies listed.
