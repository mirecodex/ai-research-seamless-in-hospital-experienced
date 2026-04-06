from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Info
    APP_ENV: str
    APP_NAME: str
    APP_VERSION: str
    
    # App Config
    SCHEDULER_TIMEZONE: Optional[str] = "Asia/Jakarta"
    LIMIT_ALEMBIC_SCOPE: Optional[int] = 0
    ENABLE_CRONJOB: Optional[int] = 0
    ENABLE_APM: Optional[int] = 0
    
    # Docker
    DOCKER_CONTAINER_NAME: str
    DOCKER_PORTS: str
    DOCKER_WORKER_COUNT: int

    # Security
    JWT_HS_SECRET: str
    JWT_RS_PRIVATE_KEY: str
    JWT_RS_PUBLIC_KEY: str
    SIGNATURE_SECRET: str
    SIGNATURE_TIMEOUT: int
    ALLOWED_ORIGINS: Optional[str] = "http://localhost:8080,http://localhost:5173"
    JWT_ROLES_INDEX: Optional[str] = 'sub'
    
    # Redis (General & Ratelimit)
    REDIS_RATELIMIT_HOST: str
    REDIS_RATELIMIT_PORT: int
    REDIS_RATELIMIT_DB: int
    CACHE_HOST: str
    CACHE_PORT: int
    CACHE_DB: int
    CACHE_PASSWORD: str
    CACHE_USERNAME: str
    CACHE_EXPIRES_SEC: int
    
    # Database
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str

    # Typesense
    TYPESENSE_API_KEY: str
    TYPESENSE_HOST: str
    TYPESENSE_PORT: str
    TYPESENSE_PROTOCOL: str
    TYPESENSE_PATH: str

    # Clickhouse
    CLICKHOUSE_HOST: str
    CLICKHOUSE_HTTP_PORT: str
    CLICKHOUSE_USER: str
    CLICKHOUSE_PASSWORD: str
    CLICKHOUSE_DATABASE: str    
    
    # APM
    APM_SERVER_URL: str
    APM_SERVICE_NAME: str
    
    # embed config
    BASE_URL_EMBED: str
    ASYNC_QWEN3_EMBED: str

    MONGODB_TYPE: str
    MONGODB_ATLAS_USERNAME: str
    MONGODB_ATLAS_PASSWORD: str
    MONGODB_ATLAS_HOST: str
    MONGODB_ATLAS_APP_NAME: str
    
    MONGODB_HOST: str
    MONGODB_PORT: int
    MONGODB_USERNAME: str
    MONGODB_PASSWORD: str
    MONGODB_DB_NAME: str
    MONGO_COLLECTION_NAME: str

    # Model Config
    GEMINI_REGULAR_MODEL: Optional[str] = None
    GEMINI_MINI_MODEL: Optional[str] = None
    GEMINI_THINKING_MODEL: Optional[str] = None
    OPENAI_REGULAR_MODEL: Optional[str] = None
    OPENAI_MINI_MODEL: Optional[str] = None
    OPENAI_THINKING_MODEL: Optional[str] = None

    # Vertex AI & Google AAPI
    GOOGLE_PROJECT_NAME: str
    GOOGLE_LOCATION_NAME: str
    SERVICE_ACCOUNT_SCOPE: str
    SERVICE_ACCOUNT_FILE: str

    # MCP
    MCP_CONFIG_AI_SEARCH: Optional[str] = None
    MCP_CONFIG_HOPE_RETRIEVER: Optional[str] = None
    MCP_SESSION: bool
    MCP_HEALTH_CHECK_URL: Optional[str] 
    
    # Azure API
    AZURE_API_KEY: str
    AZURE_API_KEY_002: str
    AZURE_API_KEY_DEV: str
    AZURE_API_VERSION: str
    AZURE_API_VERSION_002: str
    AZURE_API_VERSION_DEV: str
    AZURE_ENDPOINT: str
    AZURE_ENDPOINT_002: str
    AZURE_ENDPOINT_DEV: str

    # Claude Models
    CLAUDE_3_7_SONNET_MODEL: str
    CLAUDE_4_SONNET_MODEL: str

    # AWS API
    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    
    # Langsmith
    # langsmith_tracing: str
    # langsmith_api_key: str
    # langsmith_project: str
    # langsmith_endpoint: str
   
    # Other
    PHOENIX_API_KEY: Optional[str] = None
    PHOENIX_ENDPOINT: Optional[str] = None
    BASE_URL_UPLOADER: str

    # Navigation
    AI_SEARCH_BASE_URL: Optional[str] = None
    AI_SEARCH_TIMEOUT_S: Optional[float] = 5.0
    AI_SEARCH_WEBHOOK_URL: Optional[str] = None
    VIRTUAL_QUEUE_BASE_URL: Optional[str] = None
    PLAYWRIGHT_POOL_SIZE: Optional[int] = 3
    PLAYWRIGHT_TIMEOUT_MS: Optional[int] = 5000
    GRAPH_DATA_DIR: Optional[str] = "data/graphs"
    FLOOR_SVG_DIR: Optional[str] = "data/floors"
    S3_ROUTE_IMAGE_PREFIX: Optional[str] = "navigation/rendered/"
    GRAPH_SYNC_REDIS_CHANNEL: Optional[str] = "graph:update"
    ROUTE_CACHE_TTL_S: Optional[int] = 3600
    CB_FAILURE_THRESHOLD: Optional[int] = 3
    CB_RECOVERY_TIMEOUT_S: Optional[int] = 30
    DEFAULT_BUILDING: Optional[str] = "shlv"

    # LLM Gateway (Siloam LiteLLM proxy)
    LLM_PROVIDER: Optional[str] = "litellm"  # "litellm" or "vertex"
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = "Claude-4.5-Haiku"

    model_config = SettingsConfigDict(env_file=".env")

env = Settings()

def reload():
    env.__init__()
