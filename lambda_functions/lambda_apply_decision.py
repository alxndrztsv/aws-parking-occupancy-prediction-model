"""
Apply the human decision about a freshly trained model set.

Exposed through a Lambda Function URL so the links in the model_decision email
reach it directly, with no API Gateway in between.

Authorisation is a single-use random token that select_model stored in pending.json.
Reading that object already implies enough bucket access to copy models directly, so
the token adds nothing against a bucket reader - its job is only to make the emailed
link unguessable to everyone else. That is why there is no HMAC secret to manage.
"""

import base64
import hmac
import json
import os
import urllib.parse
from datetime import UTC, datetime

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")

ARTIFACTS_BUCKET = os.environ["ARTIFACTS_BUCKET"]
HISTORICAL_PREFIX = "historical_models"
PENDING_KEY = f"{HISTORICAL_PREFIX}/pending.json"
REGISTRY_KEY = f"{HISTORICAL_PREFIX}/best_models.json"
PREV_REGISTRY_KEY = f"{HISTORICAL_PREFIX}/best_models.prev.json"

APPROVE = "approve"
REJECT = "reject"


class LinkError(Exception):
    """A user-facing reason the link cannot be acted on."""


def _now():
    return datetime.now(UTC)


def _parse_timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            return None
        return result
    except (TypeError, ValueError):
        return None


def _fmt_expiry(value):
    """Format the ISO timestamp into a clean, readable string."""
    dt = _parse_timestamp(value)
    if dt is None:
        return "?"
    # Outputs something like: "30 Nov 2026, 06:00 UTC"
    return dt.strftime("%d %b %Y, %H:%M UTC")


def load_pending():
    try:
        body = s3.get_object(Bucket=ARTIFACTS_BUCKET, Key=PENDING_KEY)["Body"].read()
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            raise LinkError("There is no pending decision at the moment.") from exc
        raise
    return json.loads(body.decode("utf-8"))


def save_pending(pending):
    s3.put_object(
        Bucket=ARTIFACTS_BUCKET,
        Key=PENDING_KEY,
        Body=json.dumps(pending, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def validate(pending, run_id, token):
    """Raise LinkError unless this exact link may still be acted on."""
    # An empty token on either side would make compare_digest("", "") succeed and
    # turn the link into an unauthenticated endpoint, so refuse before comparing.
    stored_token = str(pending.get("token") or "")
    if not stored_token or not token or not hmac.compare_digest(stored_token, token):
        # Deliberately vague: a wrong token must not reveal anything else.
        raise LinkError("This link is not valid.")

    status = pending.get("status", "pending")
    if status != "pending":
        decided_at = pending.get("decided_at", "unknown time")
        raise LinkError(f"This decision was already {status} at {decided_at}.")

    if pending.get("run_id") != run_id:
        # The link carries the older run; pending.json holds the newer one.
        raise LinkError(
            f"This link is for run {run_id}, which was superseded by run {pending.get('run_id')}."
        )

    expires_at = _parse_timestamp(pending.get("expires_at"))
    if expires_at is None:
        raise LinkError("This link has an invalid expiry time.")

    if _now() >= expires_at:
        raise LinkError(f"This link expired on {_fmt_expiry(pending.get('expires_at'))}.")


def apply_decision(pending, decision):
    """Promote the recommended models, or record that nothing was promoted."""
    horizons = pending.get("horizons", {})

    if decision == REJECT:
        # best_models.json is left exactly as it was: production keeps every
        # retained model, and the next run re-scores the same historicals.
        return []

    promoted = []
    for horizon, entry in sorted(horizons.items()):
        if entry.get("recommendation") != "replace":
            continue
        current = entry["current"]
        src_bucket, src_key = _split_s3_uri(current["artifact_s3_uri"])
        dst_key = f"{HISTORICAL_PREFIX}/{current['model']}_{horizon}.joblib"
        s3.copy_object(
            Bucket=ARTIFACTS_BUCKET,
            CopySource={"Bucket": src_bucket, "Key": src_key},
            Key=dst_key,
        )
        promoted.append((horizon, current["model"], dst_key))

    if promoted:
        _write_registry(pending)

    return promoted


def _write_registry(pending):
    """
    Rebuild best_models.json from the decision, keeping every horizon present.

    Horizons that were not replaced must still list their historical, otherwise the
    next training run would have nothing to re-score for them.
    """
    try:
        previous = s3.get_object(Bucket=ARTIFACTS_BUCKET, Key=REGISTRY_KEY)["Body"].read()
        s3.put_object(
            Bucket=ARTIFACTS_BUCKET,
            Key=PREV_REGISTRY_KEY,
            Body=previous,
            ContentType="application/json",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise

    horizons = {}
    for horizon, entry in sorted(pending.get("horizons", {}).items()):
        chosen = (
            entry["current"] if entry.get("recommendation") == "replace" else entry["historical"]
        )
        if chosen is None:
            continue
        horizons[horizon] = {
            "model": chosen["model"],
            "source": "current" if entry.get("recommendation") == "replace" else "historical",
            "artifact_s3_uri": f"s3://{ARTIFACTS_BUCKET}/{HISTORICAL_PREFIX}/"
            f"{chosen['model']}_{horizon}.joblib",
            **{k: v for k, v in chosen.items() if k not in ("model", "artifact_s3_uri")},
        }

    registry = {
        "run_id": pending.get("run_id", ""),
        "selected_at": _now().isoformat(timespec="seconds"),
        "selection_metric": pending.get("selection_metric", ""),
        "source_metrics": pending.get("source_metrics", ""),
        "horizons": horizons,
    }
    s3.put_object(
        Bucket=ARTIFACTS_BUCKET,
        Key=REGISTRY_KEY,
        Body=json.dumps(registry, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _split_s3_uri(uri):
    if not uri.startswith("s3://"):
        raise LinkError(f"Not an S3 URI: {uri!r}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket or not key:
        raise LinkError(f"S3 URI is missing a bucket or a key: {uri!r}")
    return bucket, key


def _form_field(event, name):
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8", "replace")
    values = urllib.parse.parse_qs(body).get(name)
    return values[0].strip() if values else ""


def lambda_handler(event, context):
    method = (event.get("requestContext", {}).get("http") or {}).get("method", "GET")
    query = event.get("queryStringParameters") or {}
    run_id = (query.get("run_id") or "").strip()
    token = (query.get("token") or "").strip()

    try:
        pending = load_pending()
        validate(pending, run_id, token)
    except LinkError as exc:
        return _page(403, "Cannot act on this link", f"<p>{exc}</p>")

    if method != "POST":
        return _page(200, "Confirm model decision", _confirm_body(pending, run_id, token))

    decision = _form_field(event, "decision")
    if decision not in (APPROVE, REJECT):
        return _page(400, "Unknown decision", "<p>Expected approve or reject.</p>")

    try:
        promoted = apply_decision(pending, decision)
    except (ClientError, KeyError, LinkError) as exc:
        return _page(500, "Decision failed", f"<p>{exc}</p><p>Nothing was changed.</p>")

    # Past tense: this is a record of what happened, and validate() reads it back
    # into "This decision was already approved at ...".
    pending["status"] = "approved" if decision == APPROVE else "rejected"
    pending["decided_at"] = _now().isoformat(timespec="seconds")
    pending["decided_by_ip"] = (event.get("requestContext", {}).get("http") or {}).get("sourceIp")
    save_pending(pending)

    if decision == REJECT:
        body = "<p>Nothing was promoted. Production keeps the retained models.</p>"
    else:
        rows = "".join(
            f"<li><b>{h}</b> &rarr; {model} <code>{key}</code></li>" for h, model, key in promoted
        )
        kept = [
            h
            for h, e in sorted(pending.get("horizons", {}).items())
            if e.get("recommendation") != "replace"
        ]
        body = f"<p>Promoted {len(promoted)} model(s):</p><ul>{rows}</ul>"
        if kept:
            body += f"<p>Kept the retained model for: {', '.join(kept)}.</p>"
        body += (
            "<p>The previous registry was saved as "
            f"<code>{PREV_REGISTRY_KEY}</code> in case you need to roll back.</p>"
        )

    return _page(200, f"Decision recorded: {pending['status']}", body)


def _confirm_body(pending, run_id, token):
    rows = []
    for horizon, entry in sorted(pending.get("horizons", {}).items()):
        current = entry.get("current") or {}
        historical = entry.get("historical") or {}
        verdict = entry.get("recommendation", "")
        mark = "REPLACE" if verdict == "replace" else "keep"
        # "(none)" rather than a dash: that is what the email prints for the same
        # case, and "- (no metrics)" reads like a rendering failure.
        retained = f"{historical['model']} ({_fmt(historical)})" if historical else "(none)"
        new_best = f"{current['model']} ({_fmt(current)})" if current else "(none)"
        rows.append(
            "<tr>"
            f"<td><b>{horizon}</b></td>"
            f"<td>{retained}</td>"
            f"<td>{new_best}</td>"
            f"<td>{_fmt_delta(entry.get('improvement'))}</td>"
            f"<td><b>{mark}</b></td>"
            "</tr>"
        )

    query = urllib.parse.urlencode({"run_id": run_id, "token": token})

    # The token rides in a hidden field rather than the action URL so that it does
    # not end up in proxy or server logs a second time.
    def form(decision, label):
        return (
            f'<form method="POST" style="display:inline">'
            f'<input type="hidden" name="decision" value="{decision}">'
            f'<input type="hidden" name="run_id" value="{run_id}">'
            f'<input type="hidden" name="token" value="{token}">'
            f'<button type="submit">{label}</button></form>'
        )

    buttons = (
        f"{form(APPROVE, 'Apply the recommendation')} &nbsp; "
        f"{form(REJECT, 'Keep everything as is')}"
    )

    return f"""
<p>Run <code>{run_id}</code>
 &middot; noise band: <code>{(pending.get("threshold") or 0) * 100:.1f}%</code>
 &middot; link expires {_fmt_expiry(pending.get("expires_at"))}</p>
<p>Current and historical models were both scored on this run's validation window, so the
numbers are directly comparable.</p>
<div class="container">
<table>
<tr>
<th>Horizon</th>
<th>Retained model</th>
<th>New best model</th>
<th>Improvement</th>
<th>Action</th>
</tr>
{"".join(rows)}
</table>
<div class="buttons">{buttons}</div>
</div>
<p>Query string for reference: <code>?{query}</code></p>
<p>Nothing changes until you submit.</p>
"""


def _fmt(entry):
    metric = entry.get("valid_wape")
    return f"wape {_fmt_num(metric)}" if metric is not None else "no metrics"


def _fmt_num(value):
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "?"


def _fmt_delta(value):
    try:
        return f"{float(value) * 100:+.1f}%"
    except (TypeError, ValueError):
        # improvement is None when there is no retained model to compare against.
        # Same wording as the email, so the two never disagree.
        return "n/a"


CSS = """
    body {
        font:
          14px/1.5 system-ui,
          sans-serif;
        margin: 2rem auto;
        width: var(--table-width, fit-content);
        max-width: 52rem;
        color: #111;
    }
    table {
    border-collapse: collapse;
    margin: 1rem 0;
    }
    td,
    th {
    border: 1px solid #ccc;
    padding: 0.35rem 0.6rem;
    text-align: left;
    }
    th {
    background: #f5f5f5;
    }
    code {
    background: #f5f5f5;
    padding: 0.1rem 0.3rem;
    }
    button {
    font: inherit;
    padding: 0.4rem 0.9rem;
    margin-right: 0.5rem;
    cursor: pointer;
    }
    h1 {
    font-size: 1.3rem;
    }
    .container {
    width: fit-content;
    max-width: 100%;
    }

    .container table {
    width: max-content;
    margin: 1rem 0;
    }

    .buttons {
    display: flex;
    justify-content: space-between;
    width: var(--table-width);
    gap: 1rem;
    }

    .buttons form {
    margin: 0;
    }

    .buttons button {
    margin-right: 0;
    }
    """

JS = """
    const table = document.querySelector(".container table");
    const body = document.body;

    // Only run the observer if the table actually exists on this page
    if (table) {
        function syncWidth() {
        body.style.setProperty(
            "--table-width",
            `${table.getBoundingClientRect().width}px`,
        );
        }

        new ResizeObserver(syncWidth).observe(table);
        syncWidth();
    }
    """


def _page(status, title, body_html):
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{CSS}</style></head>"
        f"<body><h1>{title}</h1>{body_html}</body><script>{JS}</script></html>"
    )
    return {
        "statusCode": status,
        "headers": {"Content-Type": "text/html; charset=utf-8"},
        "body": html,
    }
