# Deep Learning for Phishing Site Detection

## Overview
This project is part of Stanford's CS230 course and focuses on using deep learning to detect phishing websites. Phishing detection is crucial to online security, and in this project, we aim to develop a model that can classify websites as legitimate or phishing based on various features, such as URLs, HTML structure, and images.

## Project Structure


## Requirements
To set up the project environment, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```


### Dependencies
* psycopg2 – PostgreSQL database adapter for Python (optional, if you plan to store phishing site data in PostgreSQL)
* boto3 – AWS SDK for Python (for S3 or other AWS services, if applicable)
* h5py – Library for interacting with HDF5 datasets
* numpy – Fundamental package for scientific computing with Python
* Pillow – Image processing library used for rendering and image manipulation
* requests – For making HTTP requests, used in scraping website data
* beautifulsoup4 – For parsing HTML and extracting useful information from website content


## Project Objectives
Goal: Develop a deep learning model to classify websites as phishing or legitimate based on features extracted from the website URL, HTML content, and images.
Dataset: The dataset will consist of legitimate and phishing site data collected from open-source repositories and through web scraping techniques.
Modeling Approach: We will use a combination of convolutional neural networks (CNNs) for image-based analysis and recurrent neural networks (RNNs) or transformers for text-based analysis.

## Dataset
Sources: Datasets for phishing and legitimate websites will be collected from various sources like PhishTank, and additional data may be scraped from the web using the requests and BeautifulSoup libraries.
Preprocessing: The URLs, HTML structure, and rendered images will undergo various preprocessing steps. This includes cleaning HTML content, tokenizing URLs, and resizing images for input to deep learning models.
Model Architecture
Input Features:

URL analysis: Tokenized URLs to capture patterns commonly associated with phishing sites.
HTML structure: Parsed HTML tags and structure, extracting key elements that distinguish phishing pages from legitimate ones.
Image analysis: Screenshots of web pages analyzed through a CNN to detect visual clues that may indicate phishing.
Model Details: TBD (To be filled in as the project progresses)

## Training
Training Data: A mix of phishing and legitimate website data.
Loss Function: Cross-entropy loss for binary classification.
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, and AUC-ROC for model performance.
Results
The results of the trained model will be evaluated on a test set to assess its accuracy and ability to generalize to unseen data.

## Usage
Once trained, the model can be used to classify websites into phishing or legitimate. Here's an example of how to run the model:

```bash
python src/detect_phishing.py --input <website_url> 
```


## Future Work
* Enhance the dataset by including more examples of phishing sites.
* Explore advanced deep learning architectures, such as transformers, for improved performance.
* Investigate transfer learning techniques using pre-trained models on website screenshots.
* Deploy the model as a web service using AWS Lambda or a similar platform.
