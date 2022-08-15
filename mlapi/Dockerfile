#Start with slim-buster
FROM python:3.10-slim-buster as venv

#Update apt and install curl functionality
RUN apt -qq update \
    && apt -qq install -y curl build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*

#Install and setup poetry
ENV POETRY_VERSION=1.1.13
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH /root/.local/bin:$PATH

#setup directory for poetry
WORKDIR /mlapi
COPY pyproject.toml poetry.lock ./

#run poetry to install requirements
RUN python -m venv --copies /mlapi/venv
RUN . /mlapi/venv/bin/activate\
    && poetry install --no-dev

#Create deployment with a slim version
FROM python:3.10-slim-buster as prod

#copy over venv from build
COPY --from=venv /mlapi/venv /mlapi/venv/
ENV PATH /mlapi/venv/bin:$PATH

WORKDIR /mlapi
COPY . ./

#healthcheck to make sure container is still running
HEALTHCHECK --start-period=10s --interval=30s CMD "curl http://localhost:8000/health"

#Start the listener service
CMD ["uvicorn", "mlapi.main:app", "--host", "0.0.0.0", "--port", "8000"]