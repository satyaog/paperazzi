"""Tests for paperazzi.compute_cost module."""

from itertools import product
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from paperazzi.compute_cost import main
from paperazzi.config import Config
from paperazzi.platforms.utils import iter_platforms
from paperazzi.structured_output.utils import Metadata


class EmptyModel(BaseModel):
    pass


class ValErrModel(BaseModel):
    ghost_field: str


@pytest.fixture(scope="function")
def mock_get_structured_output():
    with patch(
        "paperazzi.compute_cost.get_structured_output"
    ) as m_get_structured_output:
        m_structured_output = Mock()
        m_structured_output.METADATA = Metadata(model_id="model", model_version="0.0.0")
        m_structured_output.Analysis = EmptyModel
        m_get_structured_output.return_value = m_structured_output
        yield m_get_structured_output


class TestComputeCost:
    """Test compute cost functionality."""

    @pytest.mark.parametrize(
        ["argv", "platform", "instructor"],
        product(
            [["2.5", "10"], ["2.5", "10", "--projection", "10"]],
            # TODO: Add mistralai OCR usage parsing to compute cost
            filter(lambda x: x not in ["instructor", "mistralai"], iter_platforms()),
            ["1", ""],
        ),
    )
    def test_compute_cost(
        self,
        argv,
        platform,
        instructor,
        capsys,
        file_regression,
        mock_get_structured_output,
    ):
        """Test basic argument parsing."""
        with Config.push() as cfg:
            cfg.platform.select = platform
            cfg.platform.instructor = instructor

            main(argv)

            out, err = capsys.readouterr()
            file_regression.check("\n".join(out.splitlines() + err.splitlines()))

    def test_compute_cost_validation_error(
        self, caplog, capsys, file_regression, mock_get_structured_output
    ):
        """Test that validation errors are handled gracefully and logged."""
        mock_get_structured_output.return_value.Analysis = ValErrModel

        with Config.push() as cfg:
            cfg.platform.select = "openai"
            cfg.platform.instructor = "1"

            # Run main function - should handle validation errors gracefully
            main(["2.5", "10"])

        out, err = capsys.readouterr()
        assert len(caplog.records) == 2

        assert all(
            "validation error for ValErrModel" in record.message
            for record in caplog.records
        )
        file_regression.check("\n".join(out.splitlines() + err.splitlines()))
