import uvicorn
import pandas as pd 
import joblib
import json
import mlflow
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile


description = """
# Welcome to Getaround API.\n
[GetAround](https://www.getaround.com/?wpsrc=Google+Organic+Search) is the Airbnb for cars. You can rent cars from any person for a few hours to a few days!\n
The goal of this API will give you the optimal rental rental price of a car per day.

## Preview
Where you can: 
* `/preview` a few rows of your dataset.

## Prediction
Where you can: 
* `/prediction` of daily rental price of a car with machine learning.

Check out documentation for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Introduction Endpoints",
        "description": "Introduction",
    },
    {
        "name": "Preview",
        "description": "Preview a few rows of your dataset",
    },
    {
        "name": "Numerical",
        "description": "Endpoints that deal with numerical data"
    },
    {
        "name": "Categorical",
        "description": "Endpoints that deal with categorical data",
    },
    {
        "name": "Prediction",
        "description": "Prediction of daily rental price of a car with machine learning"
    }
]


app = FastAPI(
    title="🚗 Getaround API",
    description=description,
    version="1.0",
    openapi_tags=tags_metadata
)

class GroupBy(BaseModel):
    column: str = "model_key"
    by_method: Literal["mean", "median", "max", "min", "sum", "count"] = "mean"

class FilterBy(BaseModel):
    column: str = "model_key"
    by_category: List[str]= ["Peugeot"]

class PredictionFeatures(BaseModel):
    model_key: str = "Citroën"
    mileage: Union[int, float] = 140411
    engine_power: Union[int, float] = 100
    fuel: str = "diesel"
    paint_color: str = "black"
    car_type: str = "convertible"
    private_parking_available: bool = True
    has_gps: bool = True
    has_air_conditioning: bool = True
    automatic_car: bool = True
    has_getaround_connect: bool = True
    has_speed_regulator: bool = True
    winter_tires: bool = True


# Endpoints

@app.get("/", tags = ["Introduction Endpoints"])
async def index():
    message = "Hello world ! This `/` is the most simple and default endpoint for the API. If you want to learn more, check out documentation of the api at `/docs`."
    return message

@app.get("/greetings", tags=["Introduction Endpoints"])
async def greetings(name: str="Mr (or Miss) Nobody"):
    """
    Say hi to anybody who's specifying their name as query parameter. 
    """
    greetings = {
        "Message": f"Hello {name} How are you today?"
    }
    return greetings

@app.get("/preview", tags=["Preview"])
async def random_data(n_rows: int = 5):
    """
    Display a number of rows of the dataset. Enter an integer in n_row.
    """
    data = pd.read_csv("get_around_pricing_project.csv", index_col=0)
    if n_rows > len(data):
        response = {"message : dataset has less row than n_row you entered."}
    else:
        sample = data.sample(n_rows)
        response = sample.to_json(orient='records')
    return response

@app.get("/column_names", tags=["Preview"])
async def column_names():
    """
    Display column names of the dataset.
    """
    data = pd.read_csv("get_around_pricing_project.csv", index_col=0)
    columns = {"column names :": list(data.columns)}
    return columns

@app.get("/unique-values", tags=["Preview"])
async def unique_values(column: str = "model_key"):
    """
    Get unique values from a given column.
    """
    data = pd.read_csv("get_around_pricing_project.csv", index_col=0)
    uniq_v = pd.Series(data[column].unique())
    return uniq_v.to_json()

@app.get("/quantile", tags=["Numerical"])
async def quantile(column: str = "mileage", percent: float = 0.1, top: bool = True):
    """
    Get a values of dataset according above or below a given quantile. 
    Columns possible values are:
    * `['mileage', 'engine_power', 'rental_price_per_day']`
    You can choose whether you want the top quantile or the bottom quantile by specify `top=True` or `top=False`. Default value is `top=True`.
    Accepted values for percentage is a float between `0.01` and `0.99`, default is `0.1`.
    """
    data = pd.read_csv("get_around_pricing_project.csv", index_col=0)
    if percent > 0.99 or percent <0.01:
        msg = "percentage value is not accepted"
        return msg
    else:
        if top:
            data = data[ data[column] > data[column].quantile(1-percent)]
        else:
            data = data[ data[column] < data[column].quantile(percent)]
        return data.to_json()

@app.post("/filter-by", tags=["Categorical"])
async def filter_by(filterBy: FilterBy):
    """
    Filter by one or more categories in a given column. Columns possible values are:
    * `['model_key', 'fuel', 'paint_color', 'car_type', 'private_parking_available', 'has_gps', 'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']`
    Check values within dataset to know what kind of `categories` you can filter by. You can use `/unique-values` path to check them out.
    `categories` must be `list` format.
    """
    data = pd.read_csv("get_around_pricing_project.csv", index_col=0)
    if filterBy.by_category != None:
        data = data[data[filterBy.column].isin(filterBy.by_category)]
        return data.to_json()
    else:
        msg = "Please chose a column to filter by"
        return msg

@app.post("/groupby", tags=["Categorical"])
async def group_by(groupBy: GroupBy):
    """
    Get data grouped by a given column. Accepted columns are:
    * `['model_key', 'fuel', 'paint_color', 'car_type', 'private_parking_available', 'has_gps', 'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']`
    You can use different method to group by method which are:
    * `mean`, `median`, `min`, `max`, `sum`, `count`.
    """
    data = pd.read_csv("get_around_pricing_project.csv", index_col=0)
    data_group = data.groupby(groupBy.column).agg(groupBy.by_method, numeric_only=True)
    return data_group.to_json()


@app.post("/prediction", tags = ["Prediction"])
async def predict(features: PredictionFeatures):
    """
    Prediction for single set of input variables. Possible input values in order are:\n\n
    model_key: str\n
    mileage: float\n
    engine_power: float\n
    fuel: str\n
    paint_color: str\n
    car_type: str\n
    private_parking_available: bool\n
    has_gps: bool\n
    has_air_conditioning: bool\n
    automatic_car: bool\n
    has_getaround_connect: bool\n
    has_speed_regulator: bool\n
    winter_tires: bool\n\n

    Endpoint will return a dictionnary like this:
    \n\n
    ```
    {'prediction': rental_price_per_day}
    ```
    \n\n
    You need to give this endpoint all columns values as a dictionnary, or a form data.
    Take care to fill boolean value with true and not True with capital letter.
    """

    # Read data 
    data = pd.DataFrame(dict(features), index=[0])
    # Load model
    ml_model = joblib.load("model.joblib")
    # Prediction
    prediction = ml_model.predict(data)
    print(prediction.tolist()[0])
    # Format response
    response ={"prediction": prediction.tolist()[0]}
    return response


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)





# commande pour tester :
# docker build . -t api-getaround-name
# docker run -p 4000:4000 -e PORT=4000 api-getaround-name

# image et app : getaround-fastapi-gqforjedha

# delpoy : 
# heroku login
# heroku container:login
# heroku create getaround-fastapi-gqforjedha
# docker buildx build . --platform linux/amd64 -t getaround-fastapi-gqforjedha
# docker tag getaround-fastapi-gqforjedha registry.heroku.com/getaround-fastapi-gqforjedha/web
# docker push registry.heroku.com/getaround-fastapi-gqforjedha/web
# heroku container:release web -a getaround-fastapi-gqforjedha
# heroku open -a getaround-fastapi-gqforjedha