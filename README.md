# GR_COVID19_Mobility 🦠😷🚗 🏙️🚶🏽‍♂️
Deployment of ML models for forecasting Greek citizens' mobility and pandemic indicators during COVID19 crisis.

This repository contains the data, code, presentation and report of my diploma thesis for my BSc at University of Pireaus, Greece, supervised by Professor Yannis Theodoridis.
## Overview
The COVID-19 pandemic is one of the biggest challenges faced by modern societies as it extremely affected every aspect of human life. In the absence of prior knowledge about the virus and its effects, governments took unprecedented measures to restrict human mobility in order to hinder its spread in the general population and secure healthcare systems.

Focusing on the geographical area of Greece, the goal of this diploma thesis is to conduct bibliographic research regarding the impact of COVID-19 on human mobility and urban transport modes, as well as to develop machine learning models (MLP, RNN, LSTM, Random Forests, SVR, XGBoost, CNN, CNN-LSTM, GRU) using TensorFlow and Keras to predict citizens' mobility indexes derived from Google and Apple datasets, along with important pandemic indicators that represent the epidemiological advancement of COVID-19 in Greek society (e.g., number of deaths, cases) through real scenarios.

## Datasets
In this thesis, datasets capturing human mobility during the coronavirus pandemic were used, specifically the datasets from Apple and Google, as well as the iMEdD-Lab, Sandbird COVID-19 Greece, and COVID-19 Response Greece datasets, which are supported through appropriate GitHub repositories and provide useful information about COVID-19 cases, vaccinations, deaths, and intensive care cases in Greece. The database was developed in PostgreSQL and consists of ten tables, which are updated automatically via appropriate Python code. It is particularly important to mention that the data used and selected pertain only to the geographical area of Greece.

1. Apple COVID-19 Mobility Trends Report [Apple Inc.](https://covid19.apple.com/mobility)
2. Google COVID-19 Community Mobility Reports [Google LLC](https://www.google.com/covid19/mobility/)
3. iMEdD-Lab [iMEdD-Lab](https://github.com/iMEdD-Lab/open-data/tree/master/COVID-19)
4. Sanbird COVID-19 Greece [Sandbird](https://github.com/Sandbird/covid19-Greece)
5. COVID-19-Response Greece  [Covid-19 Response Greece](https://github.com/Covid-19-Response-Greece/covid19-data-greece)

## Research findings
The basic conclusion that resulted from the experimental study in real scenarios is that mobility is positively correlated with the contagion of the virus since the existence of the vaccine against COVID-19 and the number of cases, deaths, and people in intensive care units interact not only with each other but also with human mobility.

The time series of walking during the pandemic showed a higher prediction success rate in models, perhaps due to the smaller changes in the walking index compared to the driving time series (the existence of lockdowns imposed restrictions on traveling by car, but these were not as intensive as those on outdoor exercise and walking). In addition, the evolution of the virus in Greek society, based on various experiments conducted, appears to be more inextricably connected to human mobility in retail and recreation, public transport, and parks compared to groceries and pharmacies, workplaces, and residences.

## Acknowledgements
I would like to express my sincere gratitude to my supervising professor, Mr. Yannis Theodoridis, for his guidance, patience, and interest throughout the preparation of my thesis. In addition, I would like to thank the PhD candidate, Mr. Andreas Tritsarolis, for the support and help he offered me in overcoming difficulties and answering questions that arose during the writing stage. Finally, a heartfelt thanks goes to my family and friends for their support and understanding.
## Keywords
Machine learning, Data Analytics, Mobility data, COVID-19 pandemic, Timeseries forecasting, Urban data science
