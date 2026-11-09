"""
Ingestion Worker — background worker with bounded concurrency, retry logic,
and startup recovery for document processing jobs.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session
from app.models.ingestion_job import IngestionJob
from app.repositories import document_repository

logger = logging.getLogger("rag_engine.workers.ingestion")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class IngestionWorker:
    """
    Background worker thread that picks up PENDING ingestion jobs
    and processes them using a thread pool.
    """

    def __init__(
        self,
        ingestion_service,
        db_session_factory,
        max_concurrency: int = 2,
        poll_interval: float = 3.0,
        job_retry_limit: int = 3,
    ):
        self._ingestion_service = ingestion_service
        self._session_factory = db_session_factory
        self._max_concurrency = max_concurrency
        self._poll_interval = poll_interval
        self._job_retry_limit = job_retry_limit

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="rag_worker_pool",
        )
        self._active_jobs = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background worker thread."""
        logger.info(
            "Starting IngestionWorker (concurrency=%d, poll=%.1fs)...",
            self._max_concurrency, self._poll_interval
        )
        self._recover_stale_jobs()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="rag_ingestion_worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("IngestionWorker background thread started.")

    def stop(self, timeout: float = 10.0) -> None:
        """Signal graceful shutdown and wait for in-progress tasks."""
        logger.info("Stopping IngestionWorker...")
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("IngestionWorker stopped.")

    # ------------------------------------------------------------------
    # Worker Loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Poll SQLite database periodically for pending ingestion jobs."""
        while not self._stop_event.is_set():
            try:
                available_slots = self._max_concurrency - len(self._active_jobs)
                if available_slots > 0:
                    pending_jobs = self._fetch_pending_jobs(limit=available_slots)
                    for job in pending_jobs:
                        with self._lock:
                            if job.id not in self._active_jobs:
                                self._active_jobs.add(job.id)
                                self._executor.submit(
                                    self._process_job, job.id, job.document_id
                                )
            except Exception as exc:
                logger.error("Error in ingestion worker polling loop: %s", exc, exc_info=True)

            self._stop_event.wait(self._poll_interval)

    def _fetch_pending_jobs(self, limit: int):
        db: Session = self._session_factory()
        try:
            return document_repository.get_pending_jobs(db, limit=limit)
        finally:
            db.close()

    def _process_job(self, job_id: str, document_id: str) -> None:
        """Process a single job in worker thread pool."""
        logger.info("Worker picked up job '%s' for document '%s'", job_id, document_id)
        db: Session = self._session_factory()
        try:
            # Mark RUNNING
            document_repository.update_job_status(
                db,
                job_id=job_id,
                status="RUNNING",
                progress_message="Ingestion processing started",
            )
            db.close()

            # Execute the complete ingestion pipeline
            self._ingestion_service.process_document(document_id)

            # Mark COMPLETED
            db = self._session_factory()
            document_repository.update_job_status(
                db,
                job_id=job_id,
                status="COMPLETED",
                progress_message="Document processing completed successfully",
            )
            logger.info("Job '%s' completed successfully.", job_id)

        except Exception as exc:
            logger.error("Job '%s' failed: %s", job_id, exc)
            db = self._session_factory()
            job = document_repository.get_job(db, job_id)
            if job:
                current_attempts = job.attempt_count or 1
                if current_attempts < self._job_retry_limit:
                    document_repository.update_job_status(
                        db,
                        job_id=job_id,
                        status="PENDING",
                        error_message=f"Attempt {current_attempts} failed: {exc}. Retrying.",
                        progress_message=f"Scheduled for retry ({current_attempts + 1}/{self._job_retry_limit})",
                    )
                    logger.info("Job '%s' scheduled for retry (%d/%d)", job_id, current_attempts + 1, self._job_retry_limit)
                else:
                    document_repository.update_job_status(
                        db,
                        job_id=job_id,
                        status="FAILED",
                        error_message=f"Job failed after {current_attempts} attempts: {exc}",
                        progress_message="Processing permanently failed",
                    )
                    logger.error("Job '%s' permanently marked FAILED.", job_id)
        finally:
            db.close()
            with self._lock:
                self._active_jobs.discard(job_id)

    def _recover_stale_jobs(self) -> None:
        """
        Recover jobs that were left in RUNNING state due to an abrupt shutdown/crash.
        Resets them to PENDING so they can be re-executed cleanly.
        """
        db: Session = self._session_factory()
        try:
            stale_jobs = (
                db.query(IngestionJob)
                .filter(IngestionJob.status == "RUNNING")
                .all()
            )
            if stale_jobs:
                logger.warning("Found %d interrupted RUNNING jobs on startup. Resetting to PENDING.", len(stale_jobs))
                for job in stale_jobs:
                    job.status = "PENDING"
                    job.progress_message = "Reset to PENDING after server restart"
                    job.updated_at = _now_utc()
                db.commit()
        except Exception as exc:
            logger.error("Failed to recover stale jobs on startup: %s", exc)
        finally:
            db.close()
