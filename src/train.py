import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "data/yellow_tripdata_2025-01.parquet"
VAL_PATH = "data/yellow_tripdata_2025-02.parquet"
FEATURES = ["PULocationID", "DOLocationID", "trip_distance"]


def load(path):
    df = pd.read_parquet(path)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    return df[(df.duration >= 1) & (df.duration <= 60)]


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    train = load(TRAIN_PATH)
    val = load(VAL_PATH)
    X_val, y_val = val[FEATURES], val["duration"]

    lr = LinearRegression().fit(train[FEATURES], train["duration"])
    print("Linear regression RMSE:", rmse(y_val, lr.predict(X_val)))

    sample = train.sample(300_000, random_state=42)
    rf = RandomForestRegressor(
        n_estimators=50, max_depth=15, n_jobs=-1, random_state=42
    )
    rf.fit(sample[FEATURES], sample["duration"])
    print("Random forest RMSE:", rmse(y_val, rf.predict(X_val)))


if __name__ == "__main__":
    main()