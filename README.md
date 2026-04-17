# Hybrid Movie Recommendation System

This repository contains a **Hybrid Movie Recommendation System** that leverages both Content-Based Filtering and Collaborative Filtering. The graphical interface is built with **Streamlit** to generate personalized movie recommendations based on a user's taste and similar users' behaviours.

## Objective
The primary objective of this project is to overcome the limitations of using a straightforward recommendation engine (like only Content-Based) by integrating Collaborative predictions. By doing so, the engine:
1. Finds movies mathematically similar to what the user provides based on movie attributes (actors, genres, keywords).
2. Weights and refines those recommendations using Singular Value Decomposition (SVD) to predict how much the given user would actually rate those movies based on historical data.

This minimizes the "cold start" and "filter bubble" issues inherent to respective architectures.

## Architecture & How It Works

* **Content-Based Filtering (CBF):** Built using the TMDB 5k dataset. We calculate similarities between movies based on properties like the overview (plot), genres, cast, and crew. This textual representation undergoes text processing and Vectorization through `TfidfVectorizer`. Using `cosine_similarity`, the model retrieves a base set of mathematically similar movies.
* **Collaborative Filtering (CF):** Built using the MovieLens dataset. It creates a User-Item rating matrix. Utilizing `scipy.sparse.linalg.svds`, we decompose the matrix to learn latent factors and predict full, approximated rating scores for unseen movies. 
* **Hybrid Score Formulation:** Both systems output a normalized score, which this app uniformly blends to bring up movies that are structurally similar AND probabilistically appealing to the selected underlying user index.

## Project Structure

* **`download_data.py`**: A python script that securely downloads the necessary *TMDB 5000 Movie Dataset* via `kagglehub` and the *MovieLens Latest Small Dataset* externally into a centralized `data/` directory.
* **`train_model.py`**: The crux of the analytical engine. Reads the downloaded .csv files, cleans/collapses metadata lists, calculates TF-IDF and SVD, then serializes the structures down into `.pkl` (pickle) binaries for speedy inference tasks.
* **`app.py`**: A front-end Streamlit web application. It ingests user parameters (selected movie & user ID block), unpacks the pre-trained `.pkl` files efficiently using built-in Streamlit caching, calculates real-time Hybrid recommendations vectors, and displays the suggestions interactively.

## Setup & Installation

### 1. Prerequisites 
Ensure you have Python 3.8+ installed on your system.
You will need to install the following Python packages. You can install them manually via pip or save them to a `requirements.txt`:
```bash
pip install pandas numpy scikit-learn scipy streamlit kagglehub requests
```

### 2. Fetch the Data
Run the following script to automatically pull down all necessary datasets into the `data/` folder.
```bash
python download_data.py
```

### 3. Generate Models
Run the training script to clean the data, generate mathematical relationships, and dump the binary output models in the data folder.
```bash
python train_model.py
```
This might take a couple of minutes depending on your CPU, as matrix transformations are computationally expensive.

### 4. Run the Interface
Launch the Streamlit front-end application to interact with your tuned model!
```bash
streamlit run app.py
```

This will launch a web browser pointed to `http://localhost:8501`.
