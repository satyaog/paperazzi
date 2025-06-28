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


@pytest.fixture(scope="function")
def mock_get_structured_output():
    with patch(
        "paperazzi.compute_cost.get_structured_output"
    ) as m_get_structured_output:
        m_structured_output = Mock()
        m_structured_output.METADATA = Metadata(model_id="model", model_version="0.0.0")
        m_structured_output.Analysis = EmptyModel()
        m_get_structured_output.return_value = m_structured_output
        yield m_get_structured_output


class TestComputeCost:
    """Test compute cost functionality."""

    @pytest.mark.parametrize(
        ["argv", "platform", "instructor"],
        product(
            [["2.5", "10"], ["2.5", "10", "--projection", "10"]],
            filter(lambda x: x != "instructor", iter_platforms()),
            [True, False],
        ),
    )
    def test_compute_cost(
        self,
        argv,
        platform,
        instructor,
        file_regression,
        capsys,
        mock_get_structured_output,
    ):
        """Test basic argument parsing."""
        with Config.push() as cfg:
            cfg.platform.select = platform
            cfg.platform.instructor = instructor

            main(argv)

            out, err = capsys.readouterr()
            file_regression.check("\n".join(out.splitlines() + err.splitlines()))
