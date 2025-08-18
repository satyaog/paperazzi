import json
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from google import genai
from google.genai import types
from pydantic import BaseModel

import paperazzi
from paperazzi.platforms.utils import Message
from paperazzi.structured_output.utils import Metadata

CODE = "VTX"


# TODO: make this a class that you need to instantiate (dump and load should take the
# self or cls argument)
@dataclass
class ResponseSerializer:
    def dump(response: types.GenerateContentResponse, file_obj: BinaryIO):
        model_dump = response.model_dump()
        model_dump["_metadata"] = response._metadata.model_dump()
        return file_obj.write(
            json.dumps(model_dump, indent=2, ensure_ascii=False).encode("utf-8")
        )

    def load(file_obj: BinaryIO) -> types.GenerateContentResponse:
        return types.GenerateContentResponse.model_validate_json(
            file_obj.read().decode("utf-8")
        )


@dataclass
class ParsedResponseSerializer(ResponseSerializer):
    content_type: type[BaseModel] = BaseModel

    def dump(self, response: types.GenerateContentResponse, file_obj: BinaryIO):
        return ResponseSerializer.dump(response, file_obj)

    def load(self, file_obj: BinaryIO) -> types.GenerateContentResponse:
        data = json.load(file_obj)
        metadata = data.pop("_metadata")
        response = types.GenerateContentResponse.model_validate(data)
        response.parsed = self.content_type.model_validate(data["parsed"])
        response._metadata = Metadata.model_validate(metadata)
        return response


def client(*args, **kwargs) -> genai.Client:
    return genai.Client(*args, **kwargs)


def prompt(
    client: genai.Client,
    messages: list[Message],
    model: str,
    structured_model: BaseModel = None,
    structured_metadata: Metadata = None,
    max_attempts: int = 1,
    check: Callable[[Any], bool] = lambda _: True,
) -> types.GenerateContentResponse:
    """Generate a prompt for a list of messages.

    Args:
        client: VertexAI client
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
        contents = []
        for message in [m.format_message() for m in messages]:
            if message["role"] == "application/pdf":
                contents.append(
                    types.Part.from_bytes(
                        data=message["content"].read_bytes(),
                        mime_type=message["role"],
                    )
                )
            else:
                contents.append(message["content"])

        config = None
        if structured_model:
            config = {
                "response_mime_type": "application/json",
                "response_schema": structured_model,
            }

        response = client.models.generate_content(
            contents=contents, model=model, config=config
        )

    if structured_metadata and hasattr(response, "_metadata") and response._metadata:
        raise RuntimeError(
            f"Response already has _metadata. {paperazzi.__package__} needs this location to store its own metadata."
        )

    if structured_metadata:
        response._metadata = structured_metadata

    return response
