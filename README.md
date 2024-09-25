# **Deep Learning for Protein Subcellular Localization**

This project applies deep learning models to predict the **subcellular localization** of proteins based on their sequence data. Understanding where a protein is localized within a cell is essential for understanding its function and role in cellular processes. This project aims to leverage various neural network architectures, such as **Convolutional Neural Networks (CNN)** and **Residual Networks (ResNet)**, to perform accurate protein localization predictions.

## **Project Structure**
The project is implemented in a Jupyter notebook:

- `Projet_Localization.ipynb` - Main notebook that contains the data preprocessing, model development, training, and evaluation steps.

## **Objective**
The primary goal of this project is to predict **protein subcellular localization** using different deep learning models. This information is crucial for understanding protein functions in various cell compartments, such as the nucleus, cytoplasm, mitochondria, etc.

## **Approach**
We experimented with different neural network architectures to improve predictive performance:
- **CNN (Convolutional Neural Networks)**: Capture local patterns within protein sequences.
- **ResNet (Residual Networks)**: Use residual connections to build deeper models and avoid issues like vanishing gradients.
- **CNN + LSTM**: Explore hybrid architectures that combine CNN’s feature extraction capability with LSTM’s ability to capture sequential dependencies in protein data.

## **Data Preprocessing**
- Protein sequences are converted to a numerical format suitable for deep learning models.
- **One-hot encoding** is applied to the labels to classify the subcellular locations.
- **Masking** is used to handle variable-length sequences, ensuring the model correctly processes padded inputs.

## **Model Architecture**
The project explores several deep learning models:
- **CNN Models**: Convolutional layers followed by pooling and dense layers.
- **ResNet Models**: Stack of residual blocks for deeper architecture, which improves learning and gradient flow.
- **CNN + LSTM**: Combination of convolutional layers followed by LSTM layers to capture both spatial and sequential information from protein sequences.

## **Evaluation**
The models are evaluated using **K-fold cross-validation** to assess their generalization performance. The key metrics used for evaluation are:
- **Accuracy**: Measures the correctness of localization predictions.
- **Loss**: Evaluates how well the model is learning during training.

## **Results**
Here is a summary of the results from our models (more details can be found in the notebook):
- **Accuracy**: The final models achieved an accuracy ranging from ~76% to ~83% on validation data, depending on the architecture used.
- **Loss**: The models showed good convergence during training, with final loss values ranging between 0.64 and 0.92.

## **Installation & Usage**
To run this project, follow these steps:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/theo-serralta/Subcellular_localization.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ./Subcellular_localization
    ```
3. Open the Jupyter notebook:
    ```bash
    jupyter notebook Projet_Localization.ipynb
    ```

## **Requirements**
Make sure the following dependencies are installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Scikeras (for Keras and scikit-learn integration)

## **Conclusion**
This project demonstrates the application of deep learning techniques, specifically **CNN** and **ResNet**, to predict protein subcellular localization with high accuracy. The results highlight the potential of deep learning in bioinformatics tasks like protein localization, which is crucial for better understanding biological functions and processes at the molecular level.

## **Future Work**
Future improvements could include:
- Experimenting with other deep learning architectures like **Transformers**.
- Incorporating protein sequence embeddings from pre-trained models like **ESM** or **ProtBERT**.
- Expanding the dataset to improve generalization.

---

## **Author**
Theo Serralta

---
## **Reference**
Almagro Armenteros, J. J., Sønderby, C. K., Sønderby, S. K., Nielsen, H., & Winther, O. (2017). DeepLoc: prediction of protein subcellular localization using deep learning. Bioinformatics (Oxford, England), 33(21), 3387–3395. https://doi.org/10.1093/bioinformatics/btx431
---
