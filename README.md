# HEART FAILURE PREDICTION

## Overview
->Main purpose of this application if to predict whether a heart disease treatment will fail which will further lead to death or not.

### Prerequisites
- Python 3.x
- Pip (Python package installer)
- AWS CLI
- AWS Elastic Beanstalk CLI

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/steve601/heartfailure.git
    cd heartfailure
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application locally**:
    ```sh
    python heart.py
    ```
Access the app at `http://127.0.0.1:5000`.

## Deployment to AWS Elastic Beanstalk

### Step-by-Step Deployment

1. **Initialize Elastic Beanstalk**:
    ```sh
    eb init
    ```

    Follow the prompts to configure your application. Choose the appropriate region and platform (Python).

2. **Create an Elastic Beanstalk environment**:
    ```sh
    eb create flask-env
    ```

3. **Deploy the application**:
    ```sh
    eb deploy
    ```

4. **Access the application**:
    After deployment, access your application using the URL provided by Elastic Beanstalk.

### Configuration File: `.ebextensions/python.config`

Ensure your configuration file is set up correctly:

```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: heart:app
