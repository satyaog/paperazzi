import json
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

from mistralai import Mistral, OCRPageDimensions, OCRResponse

from paperazzi.platforms.utils import Message
from paperazzi.structured_output.utils import Metadata

CODE = "MST"


# TODO: make this a class that you need to instantiate (dump and load should take the
# self or cls argument)
@dataclass
class OCRResponseSerializer:
    def dump(response: OCRResponse, file_obj: BinaryIO):
        model_dump = response.model_dump()
        model_dump["_metadata"] = response._metadata.model_dump()
        return file_obj.write(
            json.dumps(model_dump, indent=2, ensure_ascii=False).encode("utf-8")
        )

    def load(file_obj: BinaryIO) -> OCRResponse:
        data = json.load(file_obj)
        metadata = data.pop("_metadata")

        # Fix legacy / incomplete response
        for index, page in enumerate(data["pages"]):
            page["index"] = page.pop("index", index)
            page["images"] = page.pop("images", [])
            page["dimensions"] = page.pop(
                "dimensions", OCRPageDimensions(dpi=0, height=0, width=0).model_dump()
            )
        data["model"] = data.pop("model", metadata["llm_model"])

        response = OCRResponse.model_validate(data)
        response._metadata = Metadata.model_validate(metadata)

        return response


def client(*args, **kwargs) -> Mistral:
    return Mistral(*args, **kwargs)


def ocr(
    client: Mistral,
    messages: list[Message],
    model: str,
    max_attempts: int = 1,
    check: Callable[[Any], bool] = lambda _: True,
) -> OCRResponse:
    """Generate an OCR response for a list of messages.

    Args:
        client: Mistral client
        messages: List of messages
        model: Model to use
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
                    {
                        "file_name": message["content"].name,
                        "content": message["content"].read_bytes(),
                    }
                )

        assert len(contents) == 1

        try:
            uploaded_pdf = client.files.upload(file=contents[0], purpose="ocr")
            signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
            response: OCRResponse = client.ocr.process(
                model=model,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
            )
        finally:
            if uploaded_pdf:
                client.files.delete(file_id=uploaded_pdf.id)

    return response
