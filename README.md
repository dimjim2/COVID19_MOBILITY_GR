# GR_COVID19_Mobility 🦠😷🚗 🏞️🚶🏽‍♂️
Deployment of ML models for forecasting Greek citizens' mobility and pandemic indicators during COVID19 crisis.

This repository contains the data, code, presentation and report of my diploma thesis for my BSc at University of Pireaus, Greece, supervised by Professor Yannis Theodoridis.
## Overview
The COVID-19 pandemic is one of the biggest challenges faced by modern societies as it extremely affected every aspect of human life. In the absence of prior knowledge about the virus and its effects, governments took unprecedented measures to restrict human mobility in order to hinder its spread in the general population and to secure health care systems.

Focusing on the geographical area of ​​Greece, the goal of this diploma thesis is to conduct bibliography research regarding the impact of COVID-19 on human mobility and urban transport modes, as well as the development of machine learning models (MLP, RNN, LSTM, Random, Forests, SVR, XGBoost,CNN, CNN-LSTM, GRU) using Tensorflow and Keras to predict the citizens' mobility indexes derived from Google and Apple datasets and important pandemic indicators that represent the epidemiological advancement of COVID19 in Greek society (eg, number of deaths, cases) through real scenarios.

## Datasets
In this thesis, datasets were used that capture human mobility during the coronavirus pandemic  and, more specifically, the datasets of Apple and Google and the datasets iMEdD-Lab, Sandbird COVID19-Greece and COVID-19 Response Greece, which are supported through appropriate GitHub repositories and provide useful information about COVID-19 cases, vaccinations, deaths and intensive care cases in Greece. The database was developed in PostgreSQL and consists of ten tables which are updated automatically via appropriate Python code. It is particularly important to mention that the data used and selected concern only the geographical area of Greece.

1. Apple COVID-19 Mobility Trends Report [Apple Inc.](https://covid19.apple.com/mobility)
2. Google COVID-19 Community Mobility Reports [Google LLC](https://www.google.com/covid19/mobility/)
3. iMEdD-Lab [iMEdD-Lab](https://github.com/iMEdD-Lab/open-data/tree/master/COVID-19)
4. Sanbird COVID-19 Greece [Sandbird](https://github.com/Sandbird/covid19-Greece)
5. COVID-19-Response Greece  [Covid-19 Response Greece](https://github.com/Covid-19-Response-Greece/covid19-data-greece)

## Research findings
The basic conclusion which resulted from the experimental study in real scenarios is that mobility is positively correlated with the contagion of the virus, since the existence of the vaccine against COVID- 19 and the number of cases, deaths and people in intensive care units interact both with each other but also with human mobility.

The time series of walking during the pandemic showed higher prediction success rate from models, which is perhaps due to the smaller changes that the walking index displays in comparison with the driving time series (the existence of the lockdowns imposed restrictions on travelling by car which were not as intensive as in cases of outdoor exercise and walking).  In addition, the evolution of the virus in Greek society through the various experiments that have been conducted, appears to be more inextricably connected to human mobility in retail and recreation, public transport and parks categories against groceries and pharmacies, workplaces and residences mobility residence.

## Keywords
Machine learning, Data Analytics, Mobility data, COVID-19 pandemic, Timeseries forecasting, Urban data science

   
