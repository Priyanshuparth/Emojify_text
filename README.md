# Emojify Text

![GitHub repo size](https://img.shields.io/github/repo-size/Priyanshuparth/Emojify_text)
![GitHub contributors](https://img.shields.io/github/contributors/Priyanshuparth/Emojify_text)
![GitHub stars](https://img.shields.io/github/stars/Priyanshuparth/Emojify_text?style=social)
![GitHub forks](https://img.shields.io/github/forks/Priyanshuparth/Emojify_text?style=social)
[![Instagram Follow](https://img.shields.io/badge/Instagram-%23E4405F.svg?logo=Instagram&logoColor=white)](https://instagram.com/priyanshuparth) 
[![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/priyanshuparth) 

Emojify Text is a web application built using Flask and Keras that predicts and appends relevant emojis to the input text. It utilizes a Long Short-Term Memory (LSTM) neural network trained on a dataset of text samples paired with emoji labels.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Priyanshuparth/Emojify_text.git
   ```
3. Install the required Python packages:
   
    ```bash
    pip install -r requirements.txt
    ```
5. Download the pre-trained GloVe word embeddings:
   
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
