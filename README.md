# Project Overview

In this project, ML Model with FastAPI is deployed with the help of Heroku.

Project steps:

- Train a ML Model with classification approach to prodict salary level of each individual based on the given features.
- Deploying a Model api with the help of Heroku and FastAPI.
- Use DevOps practices for continuous integration and delivery with the help of GitHub Action.

## Github Repository

[scalable model api repo](https://github.com/hsefati/scalable_ml)

## CI/CD Status Badge

[![Python CI](https://github.com/hsefati/scalable_ml/actions/workflows/ci.yml/badge.svg)](https://github.com/hsefati/scalable_ml/actions/workflows/ci.yml)

# Environment Set up

- Install poetry by following this [instruction](https://python-poetry.org/docs/#installation).

- Use `poetry install` to install all the necessary python packages

  - if virtual environment is not automatically created then:
    - `python -m venv .venv`
    - `source /.venv/bin/activate` and then `poetry install`

- Due to light weight of the model and data not external storage is used. However it is possible to add the support by following these steps:

  - In your CLI environment install the<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html" target="_blank"> AWS CLI tool</a>.
  - In the navigation bar in the Udacity classroom select **Open AWS Gateway** and then click **Open AWS Console**. You will not need the AWS Access Key ID or Secret Access Key provided here.
  - From the Services drop down select S3 and then click Create bucket.
  - Give your bucket a name, the rest of the options can remain at their default.

  To use your new S3 bucket from the AWS CLI you will need to create an IAM user with the appropriate permissions. The full instructions can be found <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console" target="_blank">here</a>, what follows is a paraphrasing:

  - Sign in to the IAM console <a href="https://console.aws.amazon.com/iam/" target="_blank">here</a> or from the Services drop down on the upper navigation bar.
  - In the left navigation bar select **Users**, then choose **Add user**.
  - Give the user a name and select **Programmatic access**.
  - In the permissions selector, search for S3 and give it **AmazonS3FullAccess**
  - Tags are optional and can be skipped.
  - After reviewing your choices, click create user.
  - Configure your AWS CLI to use the Access key ID and Secret Access key.

## Data

- Data is from [here](https://archive.ics.uci.edu/ml/datasets/census+income) which is already downloaded and saved as csv file under data folder.

## Model

- you can find the necessary information regarding the trained model on [model care](/app/model_card.md)

## Run and Test FastAPI locally

- Make sure `uvicorn & fastapi` are installed
- Go to app directory and run `uvicorn app.main:app --reload`
- After successfull run of the previous command, in your broswer go to `http://127.0.0.1:8000/`
- If you want to test the `GET` and `POST` methods then go to `http://127.0.0.1:8000/docs`
-

![docs](/screenshots/live_post.png)

## Deploy on Heroku

- Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).

- Create a new app and have it deployed from your GitHub repository.

  - Enable automatic deployments that only deploy if your continuous integration passes.
  - Hint: think about how paths will differ in your local environment vs. on Heroku.
  - Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
    ![docs](/screenshots/heroko_deployment_settings.png)

- you can test your deployed API on heroku with the following command:

  - `python scripts/test_api_post.py` FYI, you need to adjust `url`
    ![docs](/screenshots/live_post_api_call.png)
