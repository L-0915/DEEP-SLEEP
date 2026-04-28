"""
ClinicalTrials.gov crawler for sleep medicine clinical trial data.

Uses the ClinicalTrials.gov API v2 to search for sleep-related clinical
trials, fetching structured data including trial design, eligibility
criteria, interventions, results, and publications. Exports data as
JSONL for integration into the DeepSleep LLM training pipeline.

Provides high-quality structured clinical data for understanding sleep
medicine research trends, treatment approaches, and patient populations.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ClinicalTrials.gov API v2 configuration
CTGOV_API_BASE = "https://clinicaltrials.gov/api/v2"

# Rate limiting for ClinicalTrials.gov API
CTGOV_RATE_LIMIT = 1.0  # seconds between requests (be conservative)
DEFAULT_TIMEOUT = 60
MAX_RETRY_ATTEMPTS = 5
RETRY_BACKOFF_BASE = 2.0
RETRY_BACKOFF_MAX = 60.0
PAGE_SIZE = 100  # Maximum records per API request

# Sleep-related search queries for ClinicalTrials.gov
SLEEP_SEARCH_QUERIES = [
    'area[Sleep]',
    'condition:sleep AND condition:disorder',
    'condition:insomnia',
    'condition:"obstructive sleep apnea"',
    'condition:"central sleep apnea"',
    'condition:narcolepsy',
    'condition:"restless legs syndrome"',
    'condition:"circadian rhythm"',
    'condition:"REM sleep behavior disorder"',
    'condition:"shift work disorder"',
    'condition:parasomnia',
    'condition:bruxism',
    'condition:"sleep deprivation"',
    'condition:hypersomnia',
    'condition:"sleep initiation"',
    'condition:"sleep maintenance"',
    'intervention:melatonin AND condition:sleep',
    'intervention:"continuous positive airway pressure"',
    'intervention:CPAP AND condition:sleep',
]

# Request headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "DeepSleepLLM/1.0 (Educational sleep medicine research; "
        "+https://github.com/deepsleep)"
    ),
    "Accept": "application/json",
}


@dataclass(frozen=True)
class ClinicalTrial:
    """Structured representation of a ClinicalTrials.gov trial record."""

    nct_id: str
    title: str
    status: str
    phase: Optional[str]
    study_type: str
    brief_summary: str
    detailed_description: str
    conditions: list[str]
    interventions: list[dict[str, str]]
    eligibility_criteria: str
    minimum_age: Optional[str]
    maximum_age: Optional[str]
    sex: Optional[str]
    healthy_volunteers: Optional[str]
    enrollment_count: Optional[int]
    start_date: Optional[str]
    completion_date: Optional[str]
    results_first_posted: Optional[str]
    has_results: bool
    locations: list[dict[str, str]]
    sponsors: list[str]
    collaborators: list[str]
    url: str
    last_updated: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClinicalTrialsCrawler:
    """Crawler for sleep-related clinical trials from ClinicalTrials.gov.

    Uses the ClinicalTrials.gov API v2 to search for and retrieve detailed
    information about sleep medicine clinical trials. Supports resumable
    crawling by tracking processed trial IDs in a local state file.

    Attributes:
        output_dir: Directory for JSONL output files.
    """

    def __init__(self, output_dir: str = "data/raw/clinical_trials") -> None:
        self._output_dir = Path(output_dir)
        self._state_file = self._output_dir / "_clinical_trials_state.json"

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Track processed trial IDs for deduplication and resumability
        self._processed_ids: set[str] = set()
        self._load_state()

        self._last_request_time: float = 0.0

        # HTTP session with connection pooling
        self._session = requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._session.max_redirects = 5

    def _load_state(self) -> None:
        """Load previously processed trial IDs from the state file."""
        if not self._state_file.exists():
            return

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._processed_ids = set(state.get("processed_ids", []))
            logger.info(
                "Resuming: %d clinical trials already processed",
                len(self._processed_ids),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load state file, starting fresh: %s", exc)

    def _save_state(self) -> None:
        """Persist processed trial IDs to the state file."""
        state = {
            "processed_ids": sorted(list(self._processed_ids)[:200000]),
            "processed_count": len(self._processed_ids),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _rate_limit_wait(self) -> None:
        """Wait to satisfy ClinicalTrials.gov API rate limit."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < CTGOV_RATE_LIMIT:
            time.sleep(CTGOV_RATE_LIMIT - elapsed)
        self._last_request_time = time.monotonic()

    def _api_request(
        self,
        params: dict[str, Any],
        endpoint: str = "/studies",
    ) -> Optional[dict[str, Any]]:
        """Make a rate-limited API request to ClinicalTrials.gov.

        Args:
            params: Query parameters for the API request.
            endpoint: API endpoint path.

        Returns:
            JSON response as dictionary, or None on failure.
        """
        url = f"{CTGOV_API_BASE}{endpoint}"

        for attempt in range(MAX_RETRY_ATTEMPTS):
            self._rate_limit_wait()
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as exc:
                if exc.response is not None:
                    status = exc.response.status_code
                    if status == 429:
                        backoff = min(
                            RETRY_BACKOFF_BASE ** (attempt + 1),
                            RETRY_BACKOFF_MAX,
                        )
                        logger.warning(
                            "Rate limited by ClinicalTrials.gov (429), "
                            "backing off %.1fs",
                            backoff,
                        )
                        time.sleep(backoff)
                    elif status in (400, 404):
                        logger.warning(
                            "Client error %d for params: %s",
                            status,
                            str(params)[:200],
                        )
                        return None
                    elif status >= 500:
                        backoff = min(
                            RETRY_BACKOFF_BASE ** (attempt + 1),
                            RETRY_BACKOFF_MAX,
                        )
                        logger.warning(
                            "Server error %d, backing off %.1fs", status, backoff
                        )
                        time.sleep(backoff)
                    else:
                        logger.warning(
                            "HTTP %d for ClinicalTrials.gov (attempt %d): %s",
                            status,
                            attempt + 1,
                            exc,
                        )
                else:
                    logger.warning(
                        "HTTP error for ClinicalTrials.gov (attempt %d): %s",
                        attempt + 1,
                        exc,
                    )

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Request failed for ClinicalTrials.gov (attempt %d): %s",
                    attempt + 1,
                    exc,
                )

            if attempt < MAX_RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt + 1))

        logger.error(
            "Failed ClinicalTrials.gov request after %d attempts",
            MAX_RETRY_ATTEMPTS,
        )
        return None

    def _parse_trial(self, protocol_section: dict[str, Any]) -> Optional[ClinicalTrial]:
        """Parse a trial record from the ClinicalTrials.gov API response.

        Extracts structured fields from the protocol section of the API
        response into a ClinicalTrial dataclass.

        Args:
            protocol_section: The 'protocolSection' object from the API response.

        Returns:
            ClinicalTrial dataclass instance, or None if parsing fails.
        """
        try:
            id_module = protocol_section.get("identificationModule", {})
            nct_id = id_module.get("nctId", "")
            if not nct_id:
                return None

            status_module = protocol_section.get("statusModule", {})
            design_module = protocol_section.get("designModule", {})
            description_module = protocol_section.get("descriptionModule", {})
            eligibility_module = protocol_section.get("eligibilityModule", {})
            contacts_locations_module = protocol_section.get(
                "contactsLocationsModule", {}
            )
            sponsor_collab_module = protocol_section.get(
                "sponsorCollaboratorsModule", {}
            )

            # Extract conditions
            conditions = (
                description_module.get("conditionsModule", {})
                .get("conditions", [])
            )
            conditions = [c for c in conditions if isinstance(c, str)]

            # Extract interventions
            interventions_list = (
                design_module.get("interventionsModule", {})
                .get("interventions", [])
            )
            interventions: list[dict[str, str]] = []
            for interv in interventions_list:
                if isinstance(interv, dict):
                    interventions.append({
                        "type": interv.get("type", ""),
                        "name": interv.get("name", ""),
                        "description": interv.get("description", ""),
                    })

            # Extract locations
            locations_list = (
                contacts_locations_module.get("locations", [])
            )
            locations: list[dict[str, str]] = []
            for loc in locations_list:
                if isinstance(loc, dict):
                    facility = loc.get("facility", {})
                    locations.append({
                        "name": facility.get("name", "") if isinstance(facility, dict) else "",
                        "city": loc.get("city", ""),
                        "state": loc.get("state", ""),
                        "country": loc.get("country", ""),
                    })

            # Extract sponsors and collaborators
            lead_sponsor = sponsor_collab_module.get("leadSponsor", {})
            sponsors: list[str] = []
            if isinstance(lead_sponsor, dict) and lead_sponsor.get("name"):
                sponsors.append(lead_sponsor["name"])

            collaborators = (
                sponsor_collab_module.get("collaboratorsModule", {})
                .get("collaborators", [])
            )
            collab_names: list[str] = []
            for collab in collaborators:
                if isinstance(collab, dict) and collab.get("name"):
                    collab_names.append(collab["name"])

            # Extract phase
            phases = design_module.get("phases", [])
            phase = phases[0] if phases else None

            return ClinicalTrial(
                nct_id=nct_id,
                title=id_module.get("briefTitle", ""),
                status=status_module.get("overallStatus", ""),
                phase=phase,
                study_type=design_module.get("studyType", ""),
                brief_summary=(
                    description_module.get("briefSummary", "")
                ),
                detailed_description=(
                    description_module.get("detailedDescription", "")
                ),
                conditions=conditions,
                interventions=interventions,
                eligibility_criteria=eligibility_module.get(
                    "eligibilityCriteria", ""
                ),
                minimum_age=eligibility_module.get("minimumAge"),
                maximum_age=eligibility_module.get("maximumAge"),
                sex=eligibility_module.get("sex"),
                healthy_volunteers=eligibility_module.get(
                    "healthyVolunteers"
                ),
                enrollment_count=(
                    status_module.get("enrollmentInfo", {})
                    .get("count", 0)
                    if isinstance(
                        status_module.get("enrollmentInfo"), dict
                    )
                    else None
                ),
                start_date=status_module.get("startDateStruct", {})
                .get("date")
                if isinstance(
                    status_module.get("startDateStruct"), dict
                )
                else None,
                completion_date=(
                    status_module.get("completionDateStruct", {})
                    .get("date")
                    if isinstance(
                        status_module.get("completionDateStruct"), dict
                    )
                    else None
                ),
                results_first_posted=(
                    status_module.get("resultsFirstPostedStruct", {})
                    .get("date")
                    if isinstance(
                        status_module.get(
                            "resultsFirstPostedStruct"
                        ),
                        dict,
                    )
                    else None
                ),
                has_results=(
                    status_module.get("completionDateStruct", {})
                    .get("date") is not None
                    if isinstance(
                        status_module.get("completionDateStruct"), dict
                    )
                    else False
                ),
                locations=locations,
                sponsors=sponsors,
                collaborators=collab_names,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                last_updated=(
                    status_module.get("lastUpdatePostDateStruct", {})
                    .get("date")
                    if isinstance(
                        status_module.get("lastUpdatePostDateStruct"),
                        dict,
                    )
                    else ""
                ),
            )

        except Exception as exc:
            logger.error("Failed to parse trial record: %s", exc)
            return None

    def search_trials(
        self,
        query: str,
        max_results: int = 5000,
    ) -> list[dict[str, Any]]:
        """Search ClinicalTrials.gov for sleep-related clinical trials.

        Paginates through search results using the API v2 cursor-based
        pagination. Deduplicates against already-processed trial IDs.

        Args:
            query: Search query string (supports ClinicalTrials.gov query syntax).
            max_results: Maximum number of trials to retrieve.

        Returns:
            List of trial dictionaries.
        """
        logger.info(
            "Searching ClinicalTrials.gov for: '%s' (max_results=%d)",
            query,
            max_results,
        )

        trials: list[dict[str, Any]] = []
        seen_ids: set[str] = set(self._processed_ids)

        # Fields to request from the API
        requested_fields = [
            "NCTId",
            "BriefTitle",
            "OverallStatus",
            "Phase",
            "StudyType",
            "BriefSummary",
            "DetailedDescription",
            "Condition",
            "InterventionType",
            "InterventionName",
            "InterventionDescription",
            "EligibilityCriteria",
            "MinimumAge",
            "MaximumAge",
            "Sex",
            "HealthyVolunteers",
            "EnrollmentCount",
            "StartDate",
            "CompletionDate",
            "ResultsFirstPostedDate",
            "LocationFacility",
            "LocationCity",
            "LocationState",
            "LocationCountry",
            "LeadSponsorName",
            "CollaboratorName",
            "LastUpdatePostDate",
        ]

        params: dict[str, Any] = {
            "query.loc": query,
            "pageSize": PAGE_SIZE,
            "format": "json",
            "fields": ",".join(requested_fields),
        }

        total_fetched = 0

        while total_fetched < max_results:
            data = self._api_request(params)

            if data is None:
                logger.warning("API request returned None for query: %s", query)
                break

            studies = data.get("studies", [])
            if not studies:
                logger.info("No more studies for query: %s", query)
                break

            for study in studies:
                protocol = study.get("protocolSection", {})
                trial = self._parse_trial(protocol)

                if trial is None:
                    continue

                nct_id = trial.nct_id
                if nct_id in seen_ids:
                    continue

                trials.append(trial.to_dict())
                seen_ids.add(nct_id)
                total_fetched += 1

            total_available = data.get("totalCount", 0)
            logger.info(
                "Fetched %d/%d trials (total available: %d) for query: '%s'",
                total_fetched,
                max_results,
                total_available,
                query,
            )

            # Check for next page token (cursor-based pagination)
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

            params["pageToken"] = next_page_token

        logger.info(
            "Search '%s' complete: %d trials collected", query, len(trials)
        )
        return trials

    def _save_trials(
        self,
        trials: list[dict[str, Any]],
        filename: str,
    ) -> int:
        """Append trial records to a JSONL output file.

        Args:
            trials: List of trial dictionaries to save.
            filename: Name of the JSONL output file.

        Returns:
            Number of new records written.
        """
        if not trials:
            return 0

        output_path = self._output_dir / filename
        written = 0

        with open(output_path, "a", encoding="utf-8") as f:
            for trial in trials:
                nct_id = trial.get("nct_id", "")
                if nct_id in self._processed_ids:
                    continue

                f.write(json.dumps(trial, ensure_ascii=False) + "\n")
                self._processed_ids.add(nct_id)
                written += 1

        logger.info("Wrote %d trials to %s", written, output_path)
        return written

    def crawl_sleep_trials(self) -> list[dict[str, Any]]:
        """Main entry point for crawling sleep-related clinical trials.

        Executes all configured sleep-related search queries against
        ClinicalTrials.gov, deduplicates results, and saves them as JSONL.
        Supports resumable crawling.

        Returns:
            List of all collected trial dictionaries.
        """
        logger.info(
            "Starting ClinicalTrials.gov sleep trials crawl (%d queries)",
            len(SLEEP_SEARCH_QUERIES),
        )
        start_time = time.time()
        all_trials: list[dict[str, Any]] = []
        seen_ids: set[str] = set(self._processed_ids)

        for query_idx, query in enumerate(SLEEP_SEARCH_QUERIES):
            logger.info(
                "Processing query %d/%d: '%s'",
                query_idx + 1,
                len(SLEEP_SEARCH_QUERIES),
                query,
            )

            try:
                trials = self.search_trials(query, max_results=5000)

                # Deduplicate
                new_trials = [
                    t for t in trials if t["nct_id"] not in seen_ids
                ]
                logger.info(
                    "%d new trials from query '%s' (%d duplicates)",
                    len(new_trials),
                    query,
                    len(trials) - len(new_trials),
                )

                # Save per-query results
                query_file = f"trials_query_{query_idx}.jsonl"
                self._save_trials(new_trials, query_file)

                all_trials.extend(new_trials)
                seen_ids.update(t["nct_id"] for t in new_trials)

            except Exception as exc:
                logger.error(
                    "Failed to process query '%s': %s", query, exc
                )

            # Save state periodically
            self._save_state()

        # Save combined output
        combined_file = self._output_dir / "clinical_trials_sleep_all.jsonl"
        self._save_trials(all_trials, "clinical_trials_sleep_all.jsonl")

        elapsed = time.time() - start_time

        # Compute summary statistics
        status_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        for trial in all_trials:
            status = trial.get("status", "Unknown")
            phase = trial.get("phase") or "N/A"
            status_counts[status] = status_counts.get(status, 0) + 1
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        logger.info(
            "ClinicalTrials.gov crawl complete. %d trials in %.1f seconds. "
            "Status distribution: %s. Phase distribution: %s",
            len(all_trials),
            elapsed,
            dict(status_counts),
            dict(phase_counts),
        )

        return all_trials
