import json
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

import instructor
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses.response import Response
from pydantic import BaseModel

import paperazzi
from paperazzi.config import CFG
from paperazzi.platforms.openai.utils import (
    ResponseSerializer as OpenAIResponseSerializer,
)
from paperazzi.platforms.utils import Message
from paperazzi.structured_output.utils import Metadata

CODE = "INS"


@dataclass
class ResponseSerializer(OpenAIResponseSerializer):
    content_type: type[BaseModel]

    def dump(self, response: BaseModel, file_obj: BinaryIO):
        model_dump = response.model_dump()
        model_dump["_raw_response"] = response._raw_response.model_dump()
        model_dump["_metadata"] = response._metadata.model_dump()
        return file_obj.write(
            json.dumps(model_dump, indent=2, ensure_ascii=False).encode("utf-8")
        )

    def load(self, file_obj: BinaryIO) -> BaseModel:
        data = json.load(file_obj)
        raw_response = data.pop("_raw_response")
        metadata = data.pop("_metadata")
        loaded_model = self.content_type.model_validate(data)

        # Fix legacy / incomplete _raw_response
        raw_response["id"] = raw_response.pop("id", "")
        raw_response["choices"] = raw_response.pop("choices", [])
        raw_response["created"] = raw_response.pop("created", 0)
        raw_response["model"] = raw_response.pop("model", "")
        raw_response["object"] = raw_response.pop("object", "chat.completion")

        loaded_model._raw_response = ChatCompletion.model_validate(raw_response)
        loaded_model._metadata = Metadata.model_validate(metadata)
        return loaded_model


@dataclass
class ParsedResponseSerializer(ResponseSerializer):
    pass


def client(*args, **kwargs) -> instructor.Instructor:
    match CFG.platform.select:
        case "openai":
            return instructor.from_openai(
                OpenAI(*args, **kwargs), mode=instructor.Mode.TOOLS_STRICT
            )
        case _:
            raise ValueError(f"Unsupported Instructor platform: {CFG.platform.select}")


def prompt(
    client: instructor.Instructor,
    messages: list[Message],
    model: str,
    structured_model: BaseModel = None,
    structured_metadata: Metadata = None,
    max_attempts: int = 1,
    check: Callable[[Any], bool] = lambda _: True,
) -> Response:
    """Generate a prompt for a list of messages.

    Args:
        client: Instructor client
        messages: List of messages
        model: Model to use
        structured_model: Structured model to use
        structured_metadata: Metadata to use
        max_attempts: Maximum number of attempts
        check: Function to check if the response is valid
    """
    attempt = 0

    response = None
    while attempt < max_attempts and (response is None or not check(response)):
        attempt += 1

        # Generate the response
        response, raw_response = client.chat.completions.create_with_completion(
            messages=[m.format_message() for m in messages],
            model=model,
            response_model=structured_model,
            max_retries=1,
        )

    # assert check(response), f"Response is not valid: {response}"
    response._raw_response = getattr(response, "_raw_response", raw_response)

    if structured_metadata and hasattr(response, "_metadata") and response._metadata:
        raise RuntimeError(
            f"Response already has _metadata. {paperazzi.__package__} needs this location to store its own metadata."
        )

    if structured_metadata:
        response._metadata = structured_metadata

    return response
