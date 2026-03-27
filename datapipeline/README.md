# Data Pipeline

## Data Preprocessing

In this section, we describe the processes used to prepare the data for the lightweight deepfake face detection model. Data preprocessing is crucial for improving the quality of the input used in the model.

### Steps Involved:
1. **Data Cleaning**: Removing outliers and irrelevant information.
2. **Normalization**: Scaling data to a standard range.
3. **Augmentation**: Creating variations of the dataset to increase its size and diversity.

## Pipeline Scripts

The data pipeline consists of several scripts responsible for different stages:
- **Data Extraction**: Scripts to fetch data from various sources.
- **Data Transformation**: Scripts to format data into the desired structure.
- **Data Loading**: Scripts to load the preprocessed data into the model.

These scripts are designed to be modular and reusable, ensuring flexibility in the data pipeline.