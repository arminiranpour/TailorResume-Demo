import os


class Settings:
    """Configuration stub. Replace with robust settings management later."""

    def __init__(self) -> None:
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.provider = os.getenv("AI_PROVIDER", "local")
        # TODO: Add structured config loading and validation.


settings = Settings()
