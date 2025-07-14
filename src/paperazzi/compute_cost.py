import argparse
import json
import math
from pathlib import Path

import pandas as pd
import pydantic_core
from pydantic import BaseModel

from paperazzi.config import CFG
from paperazzi.log import logger
from paperazzi.platforms.utils import get_platform
from paperazzi.structured_output.utils import get_structured_output, make_disk_store

PLATFORM_INSTRUCTOR = "_raw_response"
PLATFORM_OPENAI = "usage"
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
    metadata.llm_model = CFG[CFG.platform.select].model

    disk_store = make_disk_store(metadata)

    for response in disk_store.iter_files(key="*"):
        try:
            index = Path(response.name.split("_")[-1]).stem

            response_json: dict = json.loads(response.read_text())

            try:
                with response.open("rb") as f:
                    get_platform().ResponseSerializer().load(f)
            except (TypeError, pydantic_core.ValidationError):
                with response.open("rb") as f:
                    (
                        get_platform()
                        .ParsedResponseSerializer(get_structured_output().Analysis)
                        .load(f)
                    )

            if PLATFORM_INSTRUCTOR in response_json:
                # Response data is located in the _raw_response key
                response_json = response_json[PLATFORM_INSTRUCTOR]

            if PLATFORM_VERTEXAI in response_json:
                # Parse vertexai response
                usage = response_json[PLATFORM_VERTEXAI]

            elif PLATFORM_OPENAI in response_json:
                # Parse openai response
                usage = response_json[PLATFORM_OPENAI]

            in_tokens.append(
                next(
                    filter(
                        lambda x: x >= 0,
                        [
                            # openai
                            usage.get(
                                "prompt_tokens", usage.get("input_tokens", -math.inf)
                            ),
                            # vertexai
                            usage.get("prompt_token_count", -math.inf),
                        ],
                    )
                )
            )
            out_tokens.append(
                next(
                    filter(
                        lambda x: x >= 0,
                        [
                            # openai
                            usage.get(
                                "completion_tokens",
                                usage.get("output_tokens", -math.inf),
                            ),
                            # vertexai
                            usage.get("candidates_token_count", -math.inf)
                            + usage.get("thoughts_token_count", -math.inf),
                        ],
                    )
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
