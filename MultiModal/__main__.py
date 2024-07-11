import uvicorn
import sys
sys.path.append("./")
from MultiModal.settings import settings
from pathlib import Path
import yaml

uvicorn_logger = uvicorn.config.logger
uvicorn_logger.info("Starting up the application...")

# logging_config_path = r"C:\Users\jvish\OneDrive\Documents\VISH_Stuff\VW_Proj_backend\MultiModal\logging_config.yaml"
# print(logging_config_path)
# with open(logging_config_path, 'r') as file:
#     logging_config = yaml.safe_load(file)


def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run(
        "MultiModal.web.application:get_app",
        workers=settings.workers_count,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.value.lower(),
        factory=True,
        # log_config=logging_config,
        # timeout_graceful_shutdown=2,
    )


if __name__ == "__main__":
    main()
