from app.extraction.entity_extractor import EntityExtractor


def test_email_not_extracted_as_upi_partial_domain():
    extractor = EntityExtractor()
    intel = extractor.extract_all(
        "Sure, you can reach me at rahul.support@fakebank.com for details."
    )

    assert "rahul.support@fakebank.com" in intel.email_addresses
    assert "rahul.support@fakebank" not in intel.upi_ids


def test_fake_upi_without_tld_is_extracted():
    extractor = EntityExtractor()
    intel = extractor.extract_all(
        "My UPI ID is scammer.fraud@fakebank. Send quickly."
    )

    assert "scammer.fraud@fakebank" in intel.upi_ids


def test_upi_and_email_can_coexist_without_cross_contamination():
    extractor = EntityExtractor()
    intel = extractor.extract_all(
        "UPI: scammer.fraud@fakebank and email: rahul.support@fakebank.com"
    )

    assert "scammer.fraud@fakebank" in intel.upi_ids
    assert "rahul.support@fakebank.com" in intel.email_addresses
    assert "rahul.support@fakebank" not in intel.upi_ids
