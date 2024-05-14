# Simple Streamlit App

This repository contains a simple Streamlit app to demonstrate skin color classification.

## Getting Started

Follow these instructions to set up and run the Streamlit app on your local machine.

### Prerequisites

Make sure you have Python installed. You can download it from the [official Python website](https://www.python.org/).

### Installation

1. Clone the repository or download the `app.py` and `run_app.py` files.

    ```sh
    git clone https://github.com/your-username/simple-streamlit-app.git
    cd simple-streamlit-app
    ```

2. Create a virtual environment (optional but recommended).

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages.

    ```sh
    pip install requirements.txt
    ```

### Model Training 
1. Download data from [oily, dry and normal skin dataset](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset).

2. Train the model using model_training.py script, use the following command:
   ```sh
       python model_training.py
   ```
### Running the App 

To run the Streamlit app using the `run_app.py` script, use the following command:

```sh
streamlit run app.py
