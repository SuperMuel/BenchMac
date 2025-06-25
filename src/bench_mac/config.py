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

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Defines the application's configuration settings.
    """

    # --- General Settings ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
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

    @property
    def instances_file(self) -> Path:
        """Path to the JSONL file containing benchmark instances."""
        return self.data_dir / "instances.jsonl"

    @property
    def silver_patches_dir(self) -> Path:
        """Path to the directory containing silver patch files."""
        return self.data_dir / "silver_patches"

    @property
    def cache_dir(self) -> Path:
        """Path to the root directory for temporary and cached files."""
        return self.project_root / ".benchmac_cache"

    @property
    def logs_dir(self) -> Path:
        """Path to the directory where evaluation logs are stored."""
        return self.cache_dir / "logs"

    @property
    def silver_patches_repos_dir(self) -> Path:
        """Path to the directory where repositories are temporarily cloned for silver patch generation."""  # noqa: E501
        return self.cache_dir / "silver_patches_repos"

    # --- Docker Settings ---
    docker_base_image_name: str = Field(
        default="benchmac-base",
        description="The name for the foundational Docker base image.",
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
        print("Initializing BenchMAC directories...")
        settings.data_dir.mkdir(exist_ok=True)
        settings.silver_patches_dir.mkdir(exist_ok=True)
        settings.cache_dir.mkdir(exist_ok=True)
        settings.logs_dir.mkdir(exist_ok=True)
        settings.silver_patches_repos_dir.mkdir(exist_ok=True)
        print("✅ Directories initialized.")


# Create a single, importable instance of the settings
settings = Settings()
