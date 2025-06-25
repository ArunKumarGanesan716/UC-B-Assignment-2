# What drives the price of a car?

![](images/kurt.jpeg)

**OVERVIEW**

In this application, you will explore a dataset from Kaggle. The original dataset contained information on 3 million used cars. The provided dataset contains information on 426K cars to ensure speed of processing.  Your goal is to understand what factors make a car more or less expensive.  As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

### CRISP-DM Framework

<center>
    <img src = images/crisp.png width = 50%/>
</center>


To frame the task, throughout our practical applications, we will refer back to a standard process in industry for data projects called CRISP-DM.  This process provides a framework for working through a data problem.  Your first step in this application will be to read through a brief overview of CRISP-DM [here](https://mo-pcco.s3.us-east-1.amazonaws.com/BH-PCMLAI/module_11/readings_starter.zip).  After reading the overview, answer the questions below.

### Business Understanding

From a business perspective, we are tasked with identifying key drivers for used car prices.  In the CRISP-DM overview, we are asked to convert this business framing to a data problem definition.  Using a few sentences, reframe the task as a data task with the appropriate technical vocabulary. 

### Business Context

Pre-owned vehicle dealership operates in a highly competitive and price-sensitive market. Success hinges on curating the right mix of inventory and implementing data-informed pricing strategies that not only attract prospective buyers but also maximize profit margins. 

The user car Price are influenced by many factors such as  -  Year, Brand, Type, Title status, Condition, Odometer, drive 

As a usercar dealersip I want to know 
- What do buyers value most in used cars?
- What features increase or decreses the resale Price?


#### Business Goals
- Understand key features that affect the price of used cars
- Price optimally to attract buyers
- Prioritize inventory strategy towards cars with high-value features
- Improve marketing by highlighting features consumers value most.

### Business Questions
- What car characteristics are most strongly associated with higher or lower prices?
- How does year affect the car price?
- Does brand, model, type affect pricing?
- Does Odometer reading have more impact than year?
- Does color, condition, title status impact pricing?

### Success Criteria
- Clear identification of top price-driving factors.
- Easy-to-interpret visualizations or models to support decisions.

### Data Evaluation
Total rows: 426880, Total columns: 17

# Percentage of missing Value by feature
    price           100.00
    year             99.72
    transmission     99.40
    fuel             99.29
    odometer         98.97
    model            98.76
    title_status     98.07
    manufacturer     95.87
    type             78.25
    paint_color      69.50
    drive            69.41
    condition        59.21
    cylinders        58.38
    size             28.23
    dtype: float64

The following columns have a lot of missing data and may skew the results
-              % Missing
- size          71.767476
- cylinders     41.622470
- condition     40.785232
- VIN           37.725356
- drive         30.586347
- paint_color   30.501078
- type          21.752717


### Data Preparation
Clean Data
Price Between 100 to 300000
Odometer between 0 to 1000000
and Car year from 1980

Also calculate the Age of the car as new column

Run the IQR Based Price outlier removal

    Max price: 3736928711
    Min price: 0
    Count of prices above 100000: 655
    Odometer - Max: 10000000.0
    Odometer - Min: 0.0
    Count of odometer values above 500000: 1386
    Original size: 426880
    After rule-based cleaning: 378855 (88.75%)
    After IQR cleaning: 371788 (87.09%)

Important Feature List

            Feature  Importance
    0       car_age    0.549036
    1     cylinders    0.235302
    2      odometer    0.110311
    3          fuel    0.048555
    4  transmission    0.031137
    5         drive    0.023063
    6  title_status    0.001323
    7          type    0.001273
    8  manufacturer    0.000000
    9     condition    0.000000

Drop duplicate records and remove NAN's and generate a cleaned up sample file
Removed 'id', 'VIN', 'size', 'region' features as they are not signifiant features

    Row count with any NaN values: 766
    Total rows after dropping NaN values: 71765




### Modeling

Model the following 

    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1, max_iter=1000),
    'Decision Tree': DecisionTreeRegressor(max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5)


                   Model  R2 Score          MAE           MSE         RMSE
    5  Gradient Boosting  0.792102  3054.304011  2.199625e+07  4690.015757
    4      Random Forest  0.777235  3208.617910  2.356922e+07  4854.813907
    3      Decision Tree  0.714175  3566.470430  3.024111e+07  5499.192077
    1   Ridge Regression  0.527147  5048.995169  5.002917e+07  7073.129825
    2   Lasso Regression  0.527147  5048.996899  5.002917e+07  7073.129946
    0  Linear Regression  0.527147  5049.000492  5.002919e+07  7073.131490


From the above the Gradient Boosting seems to be most effective modeling that can use used for this usecase with a R2 Score 0.792102 

### Evaluation
evaluate using gradiant boosting

            Feature  Mean Importance  Std Importance
    9       car_age         0.622457        0.006616
    2     cylinders         0.226671        0.004053
    4      odometer         0.182584        0.002392
    7         drive         0.062387        0.001296
    3          fuel         0.058306        0.001539
    8          type         0.031263        0.000910
    0  manufacturer         0.021758        0.000574
    1     condition         0.014927        0.000496
    5  title_status         0.012070        0.000814
    6  transmission         0.003061        0.000220


### Deployment

### Factors impacting Price
- Low Odometer => higher Price
- Lower car age => higher Price
- Car condition new => Higher Price
- Car has higher Cylinders => Higher Price
- 4WD  => Drives the price higher
- White color cars have a higher price and they seems to sell more white cars

### Factors affecting Sales
- Top Selling Manufactures -
	Ford, Chevrolet, Toyota, Honda, Nissan are top movers
- Gas cars sells disproportionally more than other fuel types
- Cars with Odometer less than 10 K sells more
- Car age less than 7 to 15 years sells more
- Clean title and excellent condition  sells faster
- 6 Cylinder and AWD seems to be preferred more 
- Sedans are top sellers followed by SUV and Trucks
- Automatic cars sells disproportionately high


But from all of the features The Car Age, Cylinder and Odometer plays a big factor -  Cars with Low Odometer, 6 cylinder and under 15 years will fetch higher price and will sell faster
 
