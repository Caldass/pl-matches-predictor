# Premier League Matches Predictor - Project Overview
- Deployed tool that predicts which team will win a match (Accuracy ~ 56.4%) in order to make profit out of betting markets.
- Had a profit of over 4.81% in a investment simulation using the built tool.
- Scraped more than 6000 matches data from https://www.oddsportal.com/ using BeatifulSoup and Selenium.
- Created 35+ features in order to help predict the match outcome.
- Built models and optimized Logistic Regression.
- Productionized the model using Flask and Heroku.

## Code and Resources Used
**Python version:** 3.9.5 <br />
**Packages:** Pandas, Numpy, Flask, Sklearn, Pickle, Seaborn, Matplotlib, BeatifulSoup, Selenium. <br />

## Web Scraping
Used BeautifulSoup and Selenium to scrape 6077 Premier League matches from https://www.oddsportal.com/. The data contained matches from the seasons 2005/2006 to 2020/2021. The dataset built after the Web scraping contained the following features:

- **season** - Match's season.
- **date** - Date of the match.
- **match_name** - "Home team vs Away team" string.
- **result** - Match result.
- **h_odd** - Home team winning odd.
- **d_odd** - Draw odd.
- **a_odd** - Away team winning odd.

## Data cleaning
Did a brief data cleaning and created main features in order to prepare the dataset for the next step:

- Transformed date into datetime.
- Changed odd columns to float.
- Got beginning year of a season.
- Split match name into home and away teams.
- Split scores into home and away score.
- Created feature with the match result.

## Feature Engineering
Created 35+ new features in order to help predict the match outcome. Most features had to be created for each team, therefore the prefix "ht" for home team and "at" for away team. Here are some of the created features:

- **_points** - Team's points in that season so far.
- **_l_wavg_points** - Team's weighted exponential average points in the last 3 games.
- **_rank** - Team's rank in that season so far.
- **_ls_rank** - Team's finishing rank in the previous season.
- **_days_ls_match** - Amount of days since last match.
- **_win_streak** - Team's current win streak.
- **ls_winner** - Winner of the last match between both teams.

There's a file called features.txt in the repo that explains each feature.

## EDA
In this step I did a brief data exploration in order to get a better understanding of it. Here are some highlights of the data exploration:

![alt text](https://github.com/Caldass/pl-matches-predictor/blob/master/img/distributions.png "Distributions of RFE features")

![alt text](https://github.com/Caldass/pl-matches-predictor/blob/master/img/pairplot.png "RFE features pairplot")

## Model Building
In this step I removed all the variables that wouldn't be available before a match such as "winner" and "home_score" and also split the data into training and test set with a test size of 20%. Here are the models used:

- **Logistic Regression** - Linear model expected to work due to the linear relationships between features.
- **Random Forest Classifier** - Tree based ensemble model expected to work well due to the sparsity of the data.
-  **Gradient Boost Classifier** - Just like Random Forest, a tree based ensemble model expected to work well due to the sparsity of the data.
- **K-nearest neighbors Classifier** - Non supervised learning algorithm where an object is classified by a plurality vote of its K nearest neighbors.

To evaluate the models I chose the accuracy score metric since our job is to predict the exact result, no matter what class it belongs to. Accuracy score is the total number of correct predictions divided by the total predictions.

### Model training results and feature selection
Given that Logistic Regression had the best training results it was the model chosen to continue the modeling phase:
 - **Logistic Regression** - Acc. = 56.4%
 - **Random Forest Classifier** - Acc. = 53.4%
 - **Gradient Boost Classifier** - Acc. = 54.8%
 - **K-nearest neighbors Classifier** - Acc. = 45.4%

Since there were more than 40 features in total, I performed a feature selection using the Recursive Feature Elimination (RFE) method. After applying RFE, it was possible to maintain the accuracy level with less than half of the number of original features:

![alt text](https://github.com/Caldass/pl-matches-predictor/blob/master/img/rfe.png "RFE")

After selecting 13 as the ideal number of features I tuned Logistic Regression. 

## Simulating Investment
I used the test data as a simulation of an investment in betting markets using the model's predictions and the matches' odds.

The simulation contained test data of 1216 matches. We had a fictional investment of $100 per match, making a total investment of $121,600.

Here's how the models performed:

![alt text](https://github.com/Caldass/pl-matches-predictor/blob/master/img/simulation.jpg "Simulation")

## Productionization
In this step I built a Flask API endpoint, which I deployed to a Heroku application. The application is available and hosted at https://pl-matches-predictor.herokuapp.com/predict. The input should be as follows:
- **h_odd** - Float number that represents the home team winning odd.
- **a_odd** - Float number that represents the away team winning odd.
- **d_odd** - Float number that represents the draw odd.
- **ht_rank** - Integer that represents the home team's current rank.
- **at_rank** - Integer that represents the away team's current rank.
- **ht_l_points** - Average home team points in the last 3 games.
- **at_l_points** - Average away team points in the last 3 games.
- **at_l_wavg_points** - Weighted Exponential Average away team points in the last 3 games.
- **at_l_wavg_goals** - Weighted Exponential Average away team goals in the last 3 games.
- **at_l_wavg_goals_sf** - Weighted Exponential Average away team goals suffered in the last 3 games.
- **at_win_streak** - Integer that represents the away team's current win streak.
- **ls_winner** - Last match winner between both teams ("HOME_TEAM, "AWAY_TEAM", "DRAW"). If there wasn't a last match don't fill this input.

The output of the request is a Data Frame containing a "prediction" column with the match predicted outcome.

Under _heroku/example/example.py_ there's an input example to help the understanding of the request and its output.

## Conclusion
Predicting soccer matches is not an easy task. Data Scientists have been trying to predict it and beat the betting odds efficiently for years. The results of this project were very satisfying considering it was possible to have an accuracy score better than just a random prediction or a "home team always wins" prediction:

![alt text](https://github.com/Caldass/pl-matches-predictor/blob/master/img/comparison.jpg "Comparison")

And more importantly, it was also possible to have a profit percentage on an investment simulation of approximately 4.81% using the model's prediction. 
