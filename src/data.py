from io import BytesIO
from typing import Literal

import boto3
import pandas as pd

from src.config import settings


def _load_private_csv(s3_client, filename: str) -> pd.DataFrame:
    response = s3_client.get_object(
        Bucket=settings.aws.private_bucket,
        Key=f"{settings.data.prefix}/{filename}",
    )

    return pd.read_csv(BytesIO(response["Body"].read()))


def _load_public_csv(filename: str) -> pd.DataFrame:
    url = (
        f"https://{settings.aws.public_bucket}.s3."
        f"{settings.aws.region}.amazonaws.com/"
        f"{settings.data.prefix}/{filename}"
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
            profile_name=settings.aws.profile,
            region_name=settings.aws.region,
        )

        s3_client = session.client("s3")

        return _load_private_csv(
            s3_client,
            settings.data.train_filename,
        )

    elif source == "public":
        return _load_public_csv(
            settings.data.train_filename,
        )

    else:
        raise ValueError(
            f"Unknown data source: {source}. " "Choose 'private' or 'public'."
        )
