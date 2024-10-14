# My Chatbot Flask App

In this guide, you will learn how to set up and run the Flask app that powers the chatbot.

## Requirements

Before you begin, ensure you have the following installed on your system:

- Python 3.6 or higher
- pip (the Python package installer)
- Virtual environment (optional, but recommended)

## Getting Started

Follow these steps to set up the environment and run the Flask app for your chatbot.

### 1. Install required dependencies

Use `pip` to install the necessary dependencies, which are specified in the `requirements.txt` file:

```
pip install -r requirements.txt
```

### 2. Configure environment variables

Set the following environment variables required for running the Flask app:

- **Windows**:

  ```
  set FLASK_APP=flask_app.py
  set FLASK_ENV=development
  ```

- **macOS and Linux**:

  ```
  export FLASK_APP=flask_app.py
  export FLASK_ENV=development
  ```

### 3. Run the Flask app

Now that your environment is configured, navigate to the server folder and execute the following command to start the Flask development server:

```
python flask_app.py
```

### 4. Access the chatbot 

You should now be able to use the chatbot in the website. Enjoy!

---
