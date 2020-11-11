# Sentiment Analysis Application

## Container

Building the docker file:
```
docker build -t "data-engineering-2-project" .
```

This will run the unit tests & the integration test.

Running the image:
```
docker run "data-engineering-2-project"
```

## Tests
Running the unit tests (inside the `app` folder):
```
python test_classifier.py
```

Running the integration tests (inside the `app` folder):
```
python test_integration.py
```