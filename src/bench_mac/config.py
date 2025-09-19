"""
Centralized Configuration Management for BenchMAC.

This module uses pydantic-settings to manage all application-wide settings.
It provides a single, typed `Settings` object that can be imported and used
throughout the application.

Configuration can be overridden via a `.env` file in the project root or
by setting environment variables (e.g., `BENCHMAC_LOG_LEVEL=DEBUG`).
"""

from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Defines the application's configuration settings.
    """

    # --- General Settings ---
    cli_default_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="The logging level for the application.",
    )

    # --- Directory and Path Settings ---
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="The absolute path to the project's root directory.",
    )

    @property
    def data_dir(self) -> Path:
        """Path to the directory containing benchmark data."""
        return self.project_root / "data"

    _dockerfiles_dir_override: Path | None = None

    @property
    def dockerfiles_dir(self) -> Path:
        """The default directory containing per-instance Dockerfiles."""
        if self._dockerfiles_dir_override is not None:
            return self._dockerfiles_dir_override
        return self.data_dir / "dockerfiles"

    @property
    def instances_file(self) -> Path:
        """Path to the JSONL file containing benchmark instances."""
        return self.data_dir / "instances.jsonl"

    @property
    def silver_patches_dir(self) -> Path:
        """Path to the directory containing silver patch files."""
        return self.data_dir / "silver_patches"

    @property
    def benchmac_temp_dir(self) -> Path:
        """Path to the directory containing the BenchMAC repository."""
        return self.project_root / ".benchmac"

    @property
    def cache_dir(self) -> Path:
        """Path to the root directory for temporary and cached files."""
        return self.benchmac_temp_dir / "cache"

    @property
    def evaluations_dir(self) -> Path:
        """Path to the directory where evaluation results are stored."""
        return self.benchmac_temp_dir / "evaluations"

    @property
    def silver_patches_repos_dir(self) -> Path:
        """Path to the directory where repositories are temporarily cloned for silver patch generation."""  # noqa: E501
        return self.cache_dir / "silver_patches_repos"

    @property
    def experiments_dir(self) -> Path:
        """Path to the directory where experiments are stored."""
        return self.benchmac_temp_dir / "experiments"

    # --- Docker Settings ---
    docker_host: str | None = Field(
        default=None,
        description="The host for the Docker daemon socket (e.g., 'unix:///var/run/docker.sock'). "  # noqa: E501
        "If None, the library will try to auto-detect.",
    )

    # --- Pydantic-Settings Configuration ---
    model_config = SettingsConfigDict(
        # Prefix for environment variables (e.g., BENCHMAC_LOG_LEVEL)
        env_prefix="BENCHMAC_",
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    def initialize_directories(self) -> None:
        """Creates all necessary cache and data directories."""
        logger.info("Initializing BenchMAC directories...")
        self.data_dir.mkdir(exist_ok=True)
        self.silver_patches_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.silver_patches_repos_dir.mkdir(exist_ok=True)
        self.evaluations_dir.mkdir(exist_ok=True)
        self.dockerfiles_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
        logger.info("✅ Directories initialized.")


# Create a single, importable instance of the settings
settings = Settings()
