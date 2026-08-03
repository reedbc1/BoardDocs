"""Fetch published meeting minutes from the Diligent Community API.

The generated text files are consumed by ``python_files/rag_system.py``.
Only meetings for which minutes were successfully written are recorded as
fetched, so a later run can pick up minutes that were not yet published.
"""

import argparse
import json
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://slcl.community.diligentoneplatform.com"
DEFAULT_START_DATE = "2026-01-01"
DEFAULT_END_DATE = "9999-12-31"
MINUTES_DOCUMENT_TYPES = {53, 55}
REQUEST_TIMEOUT = 30

ROOT_DIR = Path(__file__).resolve().parent
MINUTES_DIR = ROOT_DIR / "minutes"
FETCHED_IDS_PATH = ROOT_DIR / "json_files" / "fetched_ids.json"

logger = logging.getLogger(__name__)


def html_to_text(html: str) -> str:
    """Convert API HTML to readable text while retaining useful line breaks."""
    soup = BeautifulSoup(html or "", "html.parser")
    text = soup.get_text("\n").replace("\u00a0", " ")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line).strip()


def get_external_document(
    session: requests.Session, document_id: Any, last_modified: Any
) -> str:
    response = session.get(
        f"{BASE_URL}/document/{document_id}/",
        params={"lastModified": last_modified},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def process_document(session: requests.Session, document: dict[str, Any]) -> str | None:
    """Return text for a minutes document, or None for other document types."""
    document_type = document.get("DocumentType")
    if document_type not in MINUTES_DOCUMENT_TYPES:
        return None

    html = document.get("Html") or ""
    # Some migrated documents store an external document ID in Html.
    if document_type == 53:
        html = get_external_document(session, html, document.get("LastModified"))

    text = html_to_text(html)
    return text or None


def load_fetched_ids(path: Path = FETCHED_IDS_PATH) -> set[int | str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not read fetched ID file {path}: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"Fetched ID file {path} must contain a JSON list")
    return set(data)


def save_fetched_ids(fetched_ids: set[int | str], path: Path = FETCHED_IDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_ids = sorted(fetched_ids, key=str)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(ordered_ids, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def meeting_metadata(meeting: dict[str, Any]) -> dict[str, Any]:
    """Build the metadata header expected by the RAG date extractor."""
    return {
        "name": meeting.get("Name") or meeting.get("CleanName") or "",
        "description": meeting.get("MeetingLocation") or "",
        "meeting_id": meeting.get("Id"),
        "date": meeting.get("MeetingDate"),
    }


def exclusive_api_end_date(end_date: str) -> str:
    """Translate the CLI's inclusive end date to the API's exclusive boundary."""
    parsed = date.fromisoformat(end_date)
    if parsed == date.max:
        return end_date
    return (parsed + timedelta(days=1)).isoformat()


def write_minutes(meeting: dict[str, Any], texts: list[str], output_dir: Path) -> Path:
    meeting_date = str(meeting.get("MeetingDate") or "").split("T", 1)[0]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", meeting_date):
        raise ValueError(f"Meeting {meeting.get('Id')} has an invalid date: {meeting_date!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{meeting_date}.txt"
    metadata = json.dumps(meeting_metadata(meeting), indent=2)
    content = f"{metadata}\n\n---\n\n" + "\n\n".join(texts).strip() + "\n"
    output_path.write_text(content, encoding="utf-8")
    return output_path


def fetch_meetings(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: Path = MINUTES_DIR,
    fetched_ids_path: Path = FETCHED_IDS_PATH,
) -> int:
    """Fetch all newly published minutes and return the number of files written."""
    # Validate both values locally and make the user-facing end date inclusive.
    date.fromisoformat(start_date)
    api_end_date = exclusive_api_end_date(end_date)
    fetched_ids = load_fetched_ids(fetched_ids_path)
    files_written = 0

    with requests.Session() as session:
        response = session.get(
            f"{BASE_URL}/Services/MeetingsService.svc/meetings",
            params={"from": start_date, "to": api_end_date, "loadall": "false"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        meetings = response.json()
        if not isinstance(meetings, list):
            raise RuntimeError("Meetings API returned an unexpected response")

        for meeting in meetings:
            meeting_id = meeting.get("Id")
            if meeting_id is None or meeting_id in fetched_ids:
                continue

            logger.info("Checking meeting %s (%s)", meeting_id, meeting.get("MeetingDate"))
            try:
                docs_response = session.get(
                    f"{BASE_URL}/Services/MeetingsService.svc/meetings/{meeting_id}/meetingDocuments",
                    timeout=REQUEST_TIMEOUT,
                )
                docs_response.raise_for_status()
                documents = docs_response.json().get("Documents") or []
                texts = [text for doc in documents if (text := process_document(session, doc))]
                if not texts:
                    logger.info("No published minutes for meeting %s", meeting_id)
                    continue

                output_path = write_minutes(meeting, texts, output_dir)
            except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                logger.error("Could not process meeting %s: %s", meeting_id, exc)
                continue

            fetched_ids.add(meeting_id)
            save_fetched_ids(fetched_ids, fetched_ids_path)
            files_written += 1
            logger.info("Saved %s", output_path)

    return files_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=os.getenv("SCRAPE_START_DATE", DEFAULT_START_DATE))
    parser.add_argument("--end-date", default=os.getenv("SCRAPE_END_DATE", DEFAULT_END_DATE))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    files_written = fetch_meetings(args.start_date, args.end_date)
    logger.info("Fetch complete: %d new minutes file(s)", files_written)


if __name__ == "__main__":
    main()
