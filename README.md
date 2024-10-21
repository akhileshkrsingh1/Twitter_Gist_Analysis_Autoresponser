# Social Media Analysis Usecase

This project aims to analyze social media tweets for gist analysis, category classification, and response generation specifically for the Delhi Police's official Twitter account.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Logging](#logging)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7+
- MongoDB

### Clone the Repository

```bash
git clone https://github.com/shrey2003/Private_Project.git
cd Private_Project
```
### Install Dependencies
Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
## Project Structure
```
.
├── app
│   ├── api
│   │   ├── __init__.py
│   │   ├── routes.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── chains.py
│   │   ├── tweet.py
│   ├── scripts
│   │   ├── __init__.py
│   │   ├── init_db.py
│   │   ├── read_data.py
│   ├── templates
│   │   ├── __init__.py
│   │   ├── category_prompt.json
│   │   ├── priority_prompt.json
│   │   ├── response_prompt.json
│   │   ├── sentiment_prompt.json
│   ├── utils
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   ├── langchain_helpers.py
│   │   ├── translations.py
│   │   ├── config.py
│   │   ├── embeddings.py
│   ├── __init__.py
├── data
│   ├── crime_categories.txt
│   ├── priority_list.txt
│   ├── response_list.docx
├── ui
│   ├── __init__.py
├── .env
├── app.py
├── index.html
├── requirements.txt
└── README.md

```
## Usage
### Running the Application
Make sure MongoDB is running and then start the application:

``` bash
python app.py
```
### Uploading Tweet
You can upload tweets through the provided endpoint. The tweets will be analyzed and responses will be generated.

## Configuration
Configuration settings such as the MongoDB URI can be set in the app/config.py file.

### Example `config.py`
```python
MONGO_URI = "mongodb+srv://dummyuser:dummy@cluster0.qvmspwg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
```

## Logging
Logging is configured to capture errors and important information throughout the application. Logs can be viewed in the console.

### Example Log Configuration
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## API Endpoints
### GET `/tweets`
Retrieve all tweets from the database.

### GET `/draft/<tweet_id>`
Retrieve the draft response for a specific tweet.

### POST `/upload`
Upload a list of tweets for analysis.

#### Request Body:

```json
[
  {
    "tweet": "Example tweet content",
    "tweet_url": "http://example.com",
    "twitter_account": "example_user",
    "image_url": "http://example.com/image.jpg",
    "Number_of_retweet": 10,
    "number_of_followers": 100
  }
]
``` 
#### Response:

```json
{
  "message": "Tweets uploaded successfully."
}
```
### GET `/test`
Test route to check if the server is running.

#### Response:

```json
{
  "message": "Test route is working"
}
```
## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

1) Fork the repository 
2) Create a new branch (git checkout -b feature-branch)
3) Commit your changes (git commit -m 'Add new feature')
4) Push to the branch (git push origin feature-branch)
5) Create a new Pull Request
## License
This project is licensed under the MIT License. See the LICENSE file for details.
