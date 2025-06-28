import argparse
import json

import pandas as pd
import pydantic_core
from pydantic import BaseModel

from paperazzi.log import logger
from paperazzi.platforms.utils import get_platform
from paperazzi.structured_output.utils import get_structured_output, make_disk_store

PLATFORM_INSTRUCTOR = "_raw_response"
PLATFORM_VERTEXAI = "usage_metadata"


def main(argv: list = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cost_input",
        type=float,
        metavar="FLOAT",
        help=f"Cost per {1e6} input tokens",
    )
    parser.add_argument(
        "cost_output",
        type=float,
        metavar="FLOAT",
        help=f"Cost per {1e6} output tokens",
    )
    parser.add_argument(
        "--projection",
        type=int,
        default=1,
        metavar="INT",
        help=f"Number of projected queries",
    )
    options = parser.parse_args(argv)

    cost_input = options.cost_input / 1e6
    cost_output = options.cost_output / 1e6

    in_tokens = []
    out_tokens = []
    retries = []

    metadata = get_structured_output().METADATA

    disk_store = make_disk_store(metadata)

    for response in disk_store.iter_files(key="*"):
        try:
            *_, index = response.stem.split("_")

            json_data = json.loads(response.read_text())

            if PLATFORM_INSTRUCTOR in json_data:
                # Parse instructor response
                with response.open("rb") as f:
                    response = (
                        get_platform()
                        .ResponseSerializer(get_structured_output().Analysis)
                        .load(f)
                    )
                usage: BaseModel = response._raw_response.usage
                usage = usage.model_dump()
            elif PLATFORM_VERTEXAI in json_data:
                # Parse vertexai response
                with response.open("rb") as f:
                    response = (
                        get_platform()
                        .ParsedResponseSerializer(get_structured_output().Analysis)
                        .load(f)
                    )
                usage: BaseModel = response.usage_metadata
                usage = usage.model_dump()

            in_tokens.append(
                usage.get("prompt_tokens", None)
                or usage.get(
                    "input_tokens",
                    # vertexai
                    usage["prompt_token_count"],
                )
            )
            out_tokens.append(
                usage.get("completion_tokens", None)
                or usage.get(
                    "output_tokens",
                    # vertexai
                    usage["candidates_token_count"] + usage["thoughts_token_count"],
                )
            )

            if int(index) > 0:
                retries.append(response)

        except pydantic_core.ValidationError as e:
            logger.warning(
                f"Validation error for response {response}: {e}", exc_info=True
            )
            continue

    sum_input = sum(in_tokens) * cost_input
    sum_output = sum(out_tokens) * cost_output
    len_input = len(in_tokens) or 1

    data = {
        f"Total ({len(in_tokens)})": [sum_input + sum_output],
        f"{sum(in_tokens)} input token(s) @{options.cost_input:.2f}$/1M": [sum_input],
        f"{sum(out_tokens)} output token(s) @{options.cost_output:.2f}$/1M": [
            sum_output
        ],
        f"Average ({len(in_tokens)})": [(sum_input + sum_output) / len_input],
        f"Average w/o retries ({len(in_tokens) - len(retries)})": [
            (sum_input + sum_output) / (len_input - len(retries))
        ],
        f"{sum(in_tokens) / len_input:.2f} input token(s) @{options.cost_input:.2f}$/1M": [
            sum_input / len_input
        ],
        f"{sum(out_tokens) / len_input:.2f} output token(s) @{options.cost_output:.2f}$/1M": [
            sum_output / len_input
        ],
        **{
            f"Projection ({options.projection})": [
                options.projection * (sum_input + sum_output) / len_input
            ],
            f"Projection w/ retry ({options.projection * len(in_tokens) / (len_input - len(retries)):.2f})": [
                options.projection
                * len(in_tokens)
                / (len_input - len(retries))
                * (sum_input + sum_output)
                / len_input
            ],
        },
    }
    df = pd.DataFrame(data).round(3)

    string = "\n".join(
        f"{l} $" for l in df.transpose().to_string(header=False).splitlines()
    )
    print(string)


if __name__ == "__main__":
    main()
