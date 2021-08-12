# Master thesis

This is the repository for Julian FRÄBEL master thesis for Heriot Watt University on 'Temporally aware news delivery via summarization'.

This project uses Git LFS.

# Project structure

The [data](./data) folder contains the data and scripts that were used to train the rasa nlu model.

The [libs](./libs) folder contains a modified version of [this](https://github.com/ramakanth-pasunuru/QmdsCnnIr) repository made to work with the latest version of pytorch and cuda.
It is also used to create the summarization model. This code was **NOT** written by me I only adapted some parts.

The [models](./models) folder contains pre-trained models used for the abstractive multi-documents summarization part (the actual model and the tokenizer model).

The [src](./src) folder contains all the code used to run the bot / various models. Each sub module is named based on its usage.

# Install

Either use the Dockerfile to create a docker image or create a venv.

Regardless of how you run the project you need to copy [.env.example](./.env.example) to `.env` and fill it.

## .env values

BOT_API_TOKEN: the token given by [@BotFather](https://t.me/BotFather) on telegram after creating a bot (to know how to create a bot visit the [telegram documentation](https://core.telegram.org/bots/#3-how-do-i-create-a-bot)).

BOT_LOG_DIR: the directory were the bot logs will be saved. If you use a different value than the default remember to change the volume in the [docker-compose.yaml](./docker-compose.yaml) file.

BOT_ALLOWED_IDS: a comma separated list of uuid4 that are allowed in the bot login command. They will be used track each user and generate the csv files in the logs dirs.

GUARDIAN_API_KEY: The api key given by the guardian to interact with their api. [How to get a key](https://open-platform.theguardian.com/documentation/)

LOG_LEVEL: Uncomment one based on the level of logs you want.

MODEL_PATH: Leave by default except if you trained manually an other rasa model.

DUCKLING_URL: Only usefull if you use the local development environment as this value is overitten in the [docker-compose.yaml](./docker-compose.yaml) file. This should be the base url where duckling can be joined.

FAKE_DATE_NOW: An unix timestamp in milliseconds that force the bot to believe is the current time. Leave empty if you always want the bot to use current time.

## Docker

To launch in production mode run `docker compose up --build`.

This will start a duckling instance on port 8000 as well as the bot.
If you need duckling to use an other port update the environment variable DUCKLING_URL from the [docker-compose.yaml](./docker-compose.yaml) file.

## venv

If you want to run the project localy the version of python used in development was Python 3.8.3.

Create a venv using `python3 -m venv ./venv` and activate it by using `source ./venv/bin/activate`.
You can exit the venv by using `deactivate`.

Install the required dependencies using `pip install -r requirements.txt`
You will also have to download the spacy model used by the rasa pipeline using `python3 -m spacy download en_core_web_md`.

If running locally you will have to run manually a version of [duckling](https://github.com/facebook/duckling).
Do do so either build from source or use the [provided docker image](https://hub.docker.com/repository/docker/eldriann/duckling).
Regardless of how you run it fill the .env with the base url on how to reach the duckling instance (ex: `http://0.0.0.0:8000`).

You can then run the bot using `python3 ./src/main.py`.
