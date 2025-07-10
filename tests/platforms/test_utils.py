from paperazzi.platforms.utils import get_platform, iter_platforms


def test_all_platforms_have_unique_codes():
    """Test that all platforms have unique CODE constants."""
    platform_codes: dict[str, str] = {
        platform_name: get_platform(platform_name).CODE
        for platform_name in iter_platforms()
    }

    assert len(set(platform_codes.values())) == len(
        platform_codes
    ), "Platforms codes are not unique"

    # Ensure we found at least one platform with a CODE
    assert len(platform_codes) > 0, "No platform found"

    # Ensure all codes are uppercase
    assert all(
        code.isupper() for code in platform_codes.values()
    ), "Some or all platforms codes are not uppercase"

    # Ensure all codes are non-empty strings
    assert all(
        code.strip() for code in platform_codes.values()
    ), "Some or all platforms codes are empty"
