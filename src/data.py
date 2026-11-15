from io import BytesIO
from typing import Literal

import boto3
import pandas as pd

AWS_PROFILE = "ml-project"
AWS_REGION = "eu-north-1"

PRIVATE_BUCKET = "ml-production-private-858525799461-eu-north-1-an"
PUBLIC_BUCKET = "ml-production-demo-858525799461-eu-north-1-an"

DATA_PREFIX = "datasets/house-prices/v1"


def _load_private_csv(s3_client, filename: str) -> pd.DataFrame:
    response = s3_client.get_object(
        Bucket=PRIVATE_BUCKET,
        Key=f"{DATA_PREFIX}/{filename}",
    )

    return pd.read_csv(BytesIO(response["Body"].read()))


def _load_public_csv(filename: str) -> pd.DataFrame:
    url = (
        f"https://{PUBLIC_BUCKET}.s3."
        f"{AWS_REGION}.amazonaws.com/"
        f"{DATA_PREFIX}/{filename}"
    )

    return pd.read_csv(url)


def load_house_prices(
    source: Literal["private", "public"] = "private",
) -> pd.DataFrame:
    """
    Load the labeled House Prices training dataset from AWS S3.
    """

    if source == "private":
        session = boto3.Session(
            profile_name=AWS_PROFILE,
            region_name=AWS_REGION,
        )
        s3_client = session.client("s3")

        return _load_private_csv(s3_client, "train.csv")

    elif source == "public":
        return _load_public_csv("train.csv")

    else:
        raise ValueError(
            f"Unknown data source: {source}. " "Choose 'private' or 'public'."
        )
