# Steam Reviews Sentiment Analysis

## Introduction
This Steam Reviews Sentiment Analysis project leverages advanced Natural Language Processing (NLP) and supervised machine learning techniques to differentiate between positive and negative Steam reviews. Using a "noisy" dataset with over 6.41 million video game reviews from Steam, it employs a variety of Python libraries for data cleaning and preprocessing, visualization, and machine learning/NLP practices for accurate sentiment classification.

## Features
- **Data Processing and Cleaning:** Utilizes a subset of over 6.41 million Steam reviews for manageable computation, focusing on reviews that were recommended by other users to ensure reliability.
- **Exploratory Data Analysis (EDA):** Analyzes sentiment distribution across reviews and identifies common words in positive and negative feedback using separate word clouds.
- **NLP-based Sentiment Classification:** Employs text preprocessing methods and TF-IDF vectorization for converting user reviews into a format suitable for machine learning models.
- **Machine Learning Models:** Applies Logistic Regression and Multinomial Naive Bayes models, optimized through hyperparameter tuning, to classify review sentiments accurately.
- **Model Ensembling with Hyperparameter Optimization:** After evaluating base models, a stacking classifier, optimized with tuned hyperparameters, is used to enhance prediction accuracy, leveraging the strengths of individual models for superior performance.

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- wordcloud
- scikit-learn
- NLTK
- SciPy

## Model Metrics
Upon the completion of the final ensemble model's hyperparameter optimization, the following metrics were recorded and approximated to evaluate the performance of the sentiment analysis:
- **Accuracy**: 84.85%
- **Precision:** 86.86%
- **Recall:** 93.77%
- **F1 Score**: 90.19%


## Contributing
Contributions to enhance the project's capabilities and efficiency are welcome. Please feel free to submit issues, fork the repository, and submit pull requests.

## License
This project is under the MIT License. See the LICENSE file for details.

## Acknowledgments
- The data used in this project comes from the following Kaggle dataset: [Steam Reviews](https://www.kaggle.com/datasets/andrewmvd/steam-reviews), originally containing 6,417,106 rows. 
