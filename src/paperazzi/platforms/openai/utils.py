import json
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from openai import OpenAI
from openai.types.responses.parsed_response import ParsedResponse
from openai.types.responses.response import Response
from pydantic import BaseModel

import paperazzi
from paperazzi.platforms.utils import Message
from paperazzi.structured_output.utils import Metadata

CODE = "OPN"


@dataclass
class ResponseSerializer:
    def dump(self, response: Response, file_obj: BinaryIO):
        model_dump = response.model_dump()
        model_dump["_metadata"] = response._metadata.model_dump()
        return file_obj.write(
            json.dumps(model_dump, indent=2, ensure_ascii=False).encode("utf-8")
        )

    def load(self, file_obj: BinaryIO) -> Response:
        return Response.model_validate_json(file_obj.read().decode("utf-8"))


@dataclass
class ParsedResponseSerializer(ResponseSerializer):
    content_type: type[ParsedResponse] = ParsedResponse

    def dump(self, response: Response, file_obj: BinaryIO):
        return super().dump(response, file_obj)

    def load(self, file_obj: BinaryIO) -> ParsedResponse:
        # filter out the "text" field to avoid the following error:
        # text.format.ResponseFormatText.type
        #   Input should be 'text' [type=literal_error, input_value='json_schema', input_type=str]
        #     For further information visit https://errors.pydantic.dev/2.11/v/literal_error
        # text.format.ResponseFormatTextJSONSchemaConfig.schema
        #   Field required [type=missing, input_value={'name': 'Categorization'...': None, 'strict': True}, input_type=dict]
        #     For further information visit https://errors.pydantic.dev/2.11/v/missing
        # text.format.ResponseFormatJSONObject.type
        #   Input should be 'json_object' [type=literal_error, input_value='json_schema', input_type=str]
        #     For further information visit https://errors.pydantic.dev/2.11/v/literal_error
        data = json.load(file_obj)
        data.pop("text", None)
        return ParsedResponse[self.content_type].model_validate(data)


def client(*args, **kwargs) -> OpenAI:
    return OpenAI(*args, **kwargs)


def prompt(
    client: OpenAI,
    messages: list[Message],
    model: str,
    structured_model: BaseModel = None,
    structured_metadata: Metadata = None,
    no_parse: bool = False,
    max_attempts: int = 1,
    check: Callable[[Any], bool] = lambda _: True,
    responses_kwargs: dict = {},
) -> Response:
    """Generate a prompt for a list of messages.

    Args:
        client: OpenAI client
        messages: List of messages
        model: Model to use
        structured_model: Structured model to use
        structured_metadata: Metadata to use
        no_parse: If True, do not parse the response
        max_attempts: Maximum number of attempts
        check: Function to check if the response is valid
        responses_kwargs: Keyword arguments for the responses.create|parse method
    """
    no_parse = no_parse or not structured_model
    attempt = 0

    response = None
    while attempt < max_attempts and (response is None or not check(response)):
        attempt += 1

        # Generate the response
        if no_parse:
            response = client.responses.create(
                input=[m.format_message() for m in messages],
                model=model,
                **responses_kwargs,
            )
        else:
            response = client.responses.parse(
                input=[m.format_message() for m in messages],
                model=model,
                text_format=structured_model,
                **responses_kwargs,
            )

    if structured_metadata and hasattr(response, "_metadata") and response._metadata:
        raise RuntimeError(
            f"Response already has _metadata. {paperazzi.__package__} needs this location to store its own metadata."
        )

    if structured_metadata:
        response._metadata = structured_metadata

    return response
