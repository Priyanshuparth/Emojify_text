# Emojify Text

![GitHub repo size](https://img.shields.io/github/repo-size/Priyanshuparth/Emojify_text)
![GitHub contributors](https://img.shields.io/github/contributors/Priyanshuparth/Emojify_text)
![GitHub stars](https://img.shields.io/github/stars/Priyanshuparth/Emojify_text?style=social)
![GitHub forks](https://img.shields.io/github/forks/Priyanshuparth/Emojify_text?style=social)
[![Instagram Follow](https://img.shields.io/badge/Instagram-%23E4405F.svg?logo=Instagram&logoColor=white)](https://instagram.com/priyanshuparth) 
[![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/priyanshuparth) 

Emojify Text is a web application built using Flask and Keras that predicts and appends relevant emojis to the input text. It utilizes a Long Short-Term Memory (LSTM) neural network trained on a dataset of text samples paired with emoji labels.

## Activities

- Preprocessing and tokenization of text data.
- Training an LSTM neural network model.
- Building a web interface using Flask.
- Predicting emojis for user-input text.

## Technologies Used

- Python
- Flask
- Keras
- TensorFlow
- Pandas
- NumPy

## Prerequisites

- Python 3.x
- pip (Python package manager)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Priyanshuparth/Emojify_text.git
   ```
2. Install the required Python packages:
   
    ```bash
    pip install -r requirements.txt
    ```
3. Download the pre-trained GloVe word embeddings:
   
    - Download glove.6B.100d.txt from here
    - Place the file in the project directory.


## Usage

1. Run the Flask application:
    ```bash
    python app.py
    ```
2. Open a web browser and go to http://localhost:5000 to access the application.
3. Enter your text in the input field and submit. The application will predict and display the corresponding emoji.

## Model

The LSTM model architecture consists of an embedding layer followed by two LSTM layers and a dense layer with softmax activation. The model is trained using categorical cross-entropy loss and optimized using the Adam optimizer.

## Dataset

The dataset used for training consists of text samples paired with emoji labels. The dataset is preprocessed and tokenized before training the model.

## Acknowledgements

The GloVe word embeddings used in this project were trained by the Stanford NLP Group.

## Contributors

- [Priyanshu Parth](https://github.com/Priyanshuparth)
- [Abhijeet Shankar](https://github.com/abhijeet-shankar)

## License

This project is licensed under the [MIT License](LICENSE).

## Outputs 

![Screenshot 2024-04-14 175739](https://github.com/Priyanshuparth/Emojify_text/assets/73892924/c71f0420-ee18-4966-b1f9-8fa0a7f6d0a4)
![Screenshot 2024-04-14 175748](https://github.com/Priyanshuparth/Emojify_text/assets/73892924/7182d539-16d2-49b5-8e88-da1e05e49993)
