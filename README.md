# recommender-system-for-e-commerce
A Python recommender system for the Olist e-commerce dataset, featuring both Collaborative Filtering (SVD) and Content-Based (TF-IDF) models.


# 🛍️ Olist E-commerce Recommender System

This project is a comprehensive recommender system built for the **Brazilian E-Commerce Public Dataset by Olist**. It implements and compares two distinct recommendation models to suggest products to users:

1.  **Collaborative Filtering:** To provide personalized recommendations based on past user behavior ("People who like what you like, also liked this...").
2.  **Content-Based Filtering:** To find similar items based on their product details ("Because you viewed this item, you might also like...").

---

## 🎯 Project Goals

* Load and merge a complex, multi-file e-commerce dataset into a usable format.
* Implement a **Collaborative Filtering** model using **SVD (Singular Value Decomposition)** to predict user ratings.
* Implement a **Content-Based Filtering** model using **TF-IDF and Cosine Similarity** to find similar products.
* Demonstrate how each model works with different test-case scenarios, such as:
    * Recommending items to a "Power User" with many reviews.
    * Showing the "Cold Start Problem" with a new user.
    * Finding similar items from a specific product category.

---

## 💾 Dataset

This project uses the **Brazilian E-Commerce Public Dataset by Olist**. This is a large, real-world dataset containing information on 100,000 orders from 2016 to 2018.

* **Source:** [Kaggle: Brazilian E-commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
* **Key Files Used:**
    * `olist_customers_dataset.csv`
    * `olist_orders_dataset.csv`
    * `olist_order_items_dataset.csv`
    * `olist_order_reviews_dataset.csv`
    * `olist_products_dataset.csv`
    * `product_category_name_translation.csv`

---

## 🚀 Getting Started: Installation & Setup

You can run this project in a local Jupyter environment or in Google Colab.

### 1. Clone the Repository

```

git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

```

### 2. Install Dependencies
This project requires pandas, scikit-learn, and scikit-surprise.

⚠️ IMPORTANT COMPATIBILITY FIX: The scikit-surprise library is not yet compatible with NumPy 2.0+. You must install an older version of NumPy first to avoid errors.

Bash

# 1. Force-install the older NumPy version
pip install "numpy<2.0"

# 2. Install the other libraries
pip install scikit-surprise pandas scikit-learn

# 3. Run the Notebook
Open the Olist_E_commerce_Recommender_System.ipynb file in Google Colab or Jupyter Lab.

- Make sure your dataset CSV files are accessible. If using Colab, upload them or connect your Google Drive (as shown in the notebook).

- Run the cells in order from top to bottom.


### 🤖 Models Implemented
This notebook builds and tests two separate models.

# Model 1: Collaborative Filtering (SVD):

This model answers the question: "What products would this user personally like?"

- **Algorithm:** SVD (Singular Value Decomposition), a powerful Matrix Factorization technique.

- **Library:** scikit-surprise

**How it Works:**

  - We create a giant user-item-rating matrix from the data.
  
  - SVD "factorizes" this large, sparse matrix into two smaller, dense matrices (user-factors and item-factors). These factors represent the "hidden tastes" of users and "hidden attributes" of items.
  
  - By multiplying a user's factors by an item's factors, the model can predict the star rating a user would give to a product they've never even seen.
  
  - **Key Function:** get_top_n_recommendations_svd(user_id)



# Model 2: Content-Based Filtering (NLP):

**This model answers the question:** "What other products are similar to this one?"

**Algorithm:** TF-IDF (Term Frequency-Inverse Document Frequency) & Cosine Similarity.

**Library:** scikit-learn

**How it Works:**

  - This model ignores users completely and only looks at product content (in this case, their English-translated category names).
  
  - TF-IDF converts these text categories into a matrix of numerical vectors.
  
  - Cosine Similarity calculates the "angle" between every product vector. A small angle (similarity score close to 1.0) means the products are very similar.
  
  - The model finds the top 10 products with the highest similarity score.

**Key Function:** get_similar_products(product_id)


