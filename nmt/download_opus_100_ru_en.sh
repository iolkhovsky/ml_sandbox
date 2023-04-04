#!/bin/bash

mkdir data

curl https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-train.en > data/train_en.txt
curl https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-train.ru > data/train_ru.txt
curl https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-test.en > data/test_en.txt
curl https://data.statmt.org/opus-100-corpus/v1.0/supervised/en-ru/opus.en-ru-test.ru > data/test_ru.txt

echo "Downloading completed"

