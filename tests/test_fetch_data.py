import lambda_fetch_data as fetch
import pytest

VALID_HEADER = (
    "SYSTEM_ID,PRINTED_ID,METER_CODE,METER_DATE,END_DATE,PAID_DURATION,"
    "TOTAL_DURATION,FREE_DURATION,AMOUNT,CURRENCY,PAYMENT_MEAN,CARD_NAME,"
    "ZONE_DESC,CIRCUIT_DESC,ADDRESS,PARK_CODE,PARK_NAME"
)


def csv_body(header=VALID_HEADER, rows=("1,P1,100,2026-09-01 08:00:00.000000",)):
    return ("\n".join([header, *rows]) + "\n").encode()


def test_a_normal_response_passes():
    fetch.validate_csv(csv_body())


def test_access_denied_with_a_200_status_is_rejected():
    """The reported failure: the API answers 200 and a plain-text error body."""
    with pytest.raises(RuntimeError, match="not a transaction CSV"):
        fetch.validate_csv(b"ACCESS DENIED")


def test_the_rejection_quotes_the_actual_body():
    """The message has to say what came back, or the cause is unguessable."""
    with pytest.raises(RuntimeError, match="ACCESS DENIED"):
        fetch.validate_csv(b"ACCESS DENIED\n")


@pytest.mark.parametrize(
    "body",
    [
        b"ACCESS DENIED",
        b"access denied",
        b"INVALID CREDENTIALS",
        b"<html><body>Session expired</body></html>",
        b'{"error":"unauthorized"}',
        b"METER_CODE,AMOUNT\n100,1.5\n",  # some columns, but not all three
    ],
)
def test_any_non_transaction_body_is_rejected(body):
    """The check is a whitelist on the expected contract, not a blacklist on known
    error strings, so wording the API has never used is still caught."""
    with pytest.raises(RuntimeError):
        fetch.validate_csv(body)


def test_an_empty_body_is_rejected():
    with pytest.raises(RuntimeError, match="empty body"):
        fetch.validate_csv(b"")

    with pytest.raises(RuntimeError, match="empty body"):
        fetch.validate_csv(b"   \n  ")


def test_a_header_without_data_rows_is_rejected():
    """A month with no rows would silently punch a hole in the feature table."""
    with pytest.raises(RuntimeError, match="no data rows"):
        fetch.validate_csv((VALID_HEADER + "\n").encode())


def test_a_byte_order_mark_does_not_break_the_first_column():
    body = b"\xef\xbb\xbf" + csv_body()

    fetch.validate_csv(body)


def test_quoted_and_reordered_headers_are_accepted():
    """Only presence matters, not position or quoting."""
    header = '"METER_DATE","SYSTEM_ID","METER_CODE"'
    fetch.validate_csv(csv_body(header=header, rows=('"2026-09-01",1,100',)))


def test_lowercase_headers_are_accepted():
    header = "system_id,meter_code,meter_date"
    fetch.validate_csv(csv_body(header=header, rows=("1,100,2026-09-01",)))


def test_expected_columns_match_the_glue_rename_map():
    """Regression guard: if the Glue job stops recognising a column, bronze and
    silver disagree and the failure surfaces minutes later inside Spark."""
    import re
    from pathlib import Path

    glue = Path(__file__).resolve().parents[1] / "glue_scripts" / "bronze_to_silver.py"
    raw_columns = set(re.findall(r'^\s*"([A-Z_]+)":', glue.read_text(), re.M))

    assert set(fetch.EXPECTED_HEADER_COLUMNS) <= raw_columns
