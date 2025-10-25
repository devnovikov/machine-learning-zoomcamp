## Homework

> Note: sometimes your answer doesn't match one of the options exactly. 
> That's fine. 
> Select the option that's closest to your solution.
> If it's exactly in between two options, select the higher value.

We recommend using python 3.12 or 3.13 in this homework.

In this homework, we're going to continue working with the lead scoring dataset. You don't need the dataset: we will provide the model for you.


## Question 1

* Install `uv`
* What's the version of uv you installed?
* Use `--version` to find out

uv --version
Answer: uv 0.9.5

## Initialize an empty uv project

You should create an empty folder for homework
and do it there. 


## Question 2

* Use uv to install Scikit-Learn version 1.6.1 
* What's the first hash for Scikit-Learn you get in the lock file?
* Include the entire string starting with sha256:, don't include quotes

uv add scikit-learn==1.6.1

sha256:b4fc2525eca2c69a59260f583c56a7557c6ccdf8deafdba6e060f94c1c59738e

## Models

We have prepared a pipeline with a dictionary vectorizer and a model.

It was trained (roughly) using this code:

```python
categorical = ['lead_source']
numeric = ['number_of_courses_viewed', 'annual_income']

df[categorical] = df[categorical].fillna('NA')
df[numeric] = df[numeric].fillna(0)

train_dict = df[categorical + numeric].to_dict(orient='records')

pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)

pipeline.fit(train_dict, y_train)
```

> **Note**: You don't need to train the model. This code is just for your reference.

And then saved with Pickle. Download it [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/cohorts/2025/05-deployment/pipeline_v1.bin).

With `wget`:

```bash
wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin
```


## Question 3

Let's use the model!

* Write a script for loading the pipeline with pickle
* Score this record:

```json
{
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
```

What's the probability that this lead will convert? 

```bash
uv run python predict_pickle.py
```

* 0.533

## Question 4

Now let's serve this model as a web service

* Install FastAPI
* Write FastAPI code for serving the model
* Now score this client using `requests`:

```python
url = "YOUR_URL"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
requests.post(url, json=client).json()
```

What's the probability that this client will get a subscription?

Answer: {'subscription_probability': 0.5340417283801275}

* 0.534


## Docker

Install [Docker](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/06-docker.md). 
We will use it for the next two questions.

For these questions, we prepared a base image: `agrigorev/zoomcamp-model:2025`. 
You'll need to use it (see Question 5 for an example).

This image is based on `3.13.5-slim-bookworm` and has
a pipeline with logistic regression (a different one)
as well a dictionary vectorizer inside. 

This is how the Dockerfile for this image looks like:

```docker 
FROM python:3.13.5-slim-bookworm
WORKDIR /code
COPY pipeline_v2.bin .
```

We already built it and then pushed it to [`agrigorev/zoomcamp-model:2025`](https://hub.docker.com/r/agrigorev/zoomcamp-model).

> **Note**: You don't need to build this docker image, it's just for your reference.


## Question 5

Download the base image `agrigorev/zoomcamp-model:2025`. You can easily make it by using [docker pull](https://docs.docker.com/engine/reference/commandline/pull/) command.

> docker pull agrigorev/zoomcamp-model:2025

> docker images agrigorev/zoomcamp-model:2025

```
REPOSITORY                 TAG       IMAGE ID       CREATED      SIZE
agrigorev/zoomcamp-model   2025      4a9ecc576ae9   4 days ago   121MB
```


So what's the size of this base image?

Answer: 121MB

* 121 MB

You can get this information when running `docker images` - it'll be in the "SIZE" column.


## Dockerfile

Now create your own `Dockerfile` based on the image we prepared.

It should start like that:

```docker
FROM agrigorev/zoomcamp-model:2025
# add your stuff here
```

Now complete it:

* Install all the dependencies from pyproject.toml
* Copy your FastAPI script
* Run it with uvicorn 

After that, you can build your docker image.

```
 docker buildx build --platform linux/amd64 -t mlzoomcamp2025_hw5 -f Dockerfile .
[+] Building 5.0s (13/13) FINISHED                                                                                                                                                        docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                                                                                      0.0s
 => => transferring dockerfile: 398B                                                                                                                                                                      0.0s
 => [internal] load metadata for ghcr.io/astral-sh/uv:latest                                                                                                                                              0.3s
 => [internal] load metadata for docker.io/agrigorev/zoomcamp-model:2025                                                                                                                                  0.0s
 => [internal] load .dockerignore                                                                                                                                                                         0.0s
 => => transferring context: 2B                                                                                                                                                                           0.0s
 => FROM ghcr.io/astral-sh/uv:latest@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7                                                                                              1.2s
 => => resolve ghcr.io/astral-sh/uv:latest@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7                                                                                        0.0s
 => => sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 2.19kB / 2.19kB                                                                                                            0.0s
 => => sha256:0f419824ea1810fe2a47af12aef8e8c39eabae8d19c728cd9612d18e17a98d17 669B / 669B                                                                                                                0.0s
 => => sha256:d113707eb731275ee0cdfbb9c115f020876888fa376857bee3f6cc5a471e2f2b 1.30kB / 1.30kB                                                                                                            0.0s
 => => sha256:0daf2b1a974db1a0f3bca91305aba6c54d333d82104cd404c9c58813b6dd4146 21.82MB / 21.82MB                                                                                                          1.0s
 => => sha256:99df4935c9790c120a2cd9be361f1d2717560ba4c68e6eb037a883bfac214c18 96B / 96B                                                                                                                  0.3s
 => => extracting sha256:0daf2b1a974db1a0f3bca91305aba6c54d333d82104cd404c9c58813b6dd4146                                                                                                                 0.2s
 => => extracting sha256:99df4935c9790c120a2cd9be361f1d2717560ba4c68e6eb037a883bfac214c18                                                                                                                 0.0s
 => [internal] load build context                                                                                                                                                                         0.0s
 => => transferring context: 1.48kB                                                                                                                                                                       0.0s
 => CACHED [stage-0 1/6] FROM docker.io/agrigorev/zoomcamp-model:2025                                                                                                                                     0.0s
 => [stage-0 2/6] COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/                                                                                                                                  0.0s
 => [stage-0 3/6] WORKDIR /app                                                                                                                                                                            0.0s
 => [stage-0 4/6] COPY pyproject.toml uv.lock .python-version ./                                                                                                                                          0.0s
 => [stage-0 5/6] RUN uv sync --locked                                                                                                                                                                    2.9s
 => [stage-0 6/6] COPY predict_fastapi.py pipeline_v1.bin ./                                                                                                                                              0.0s
 => exporting to image                                                                                                                                                                                    0.4s
 => => exporting layers                                                                                                                                                                                   0.4s
 => => writing image sha256:347adac2ad92df60be60fa32f97b2720f455f89f0614f8b14ed811ad1a4c2af7                                                                                                              0.0s
 => => naming to docker.io/library/mlzoomcamp2025_hw5                                                                                                                                                     0.0s

View build details: docker-desktop://dashboard/build/desktop-linux/desktop-linux/px1e2xt3k67w0r7lkuyo88e8p

docker run -it --rm -p 8000:8000 mlzoomcamp2025_hw5
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     192.165.15.16:55243 - "POST /predict HTTP/1.1" 200 OK


``` 


## Question 6

Let's run your docker container!

After running it, score this client once again:

```python
url = "YOUR_URL"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
requests.post(url, json=client).json()
```

What's the probability that this lead will convert?

{'subscription_probability': 0.9933071490756734}
* 0.99


## Submit the results

* Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw05
* If your answer doesn't match options exactly, select the closest one. If the answer is exactly in between two options, select the higher value.



## Publishing to Docker hub

This is just for reference, this is how we published an image to Docker hub.

`Dockerfile_base`: 

```dockerfile
FROM python:3.13.5-slim-bookworm
WORKDIR /code
COPY pipeline_v2.bin .
```

Publishing:

```bash
docker build -t mlzoomcamp2025_hw5 -f Dockerfile_base .
docker tag mlzoomcamp2025_hw5:latest agrigorev/zoomcamp-model:2025
docker push agrigorev/zoomcamp-model:2025
```


```bash
docker tag mlzoomcamp2025_hw5:latest alexeynovikov/mlzoomcamp2025_hw5:latest
docker push alexeynovikov/mlzoomcamp2025_hw5:latest
```