from elasticapm.contrib.starlette import ElasticAPM
from starlette.middleware.cors import CORSMiddleware
from .apm import apm
from .setting import env

origins = env.ALLOWED_ORIGINS.split(",")

def setup_middleware(app):
    kwargs = dict(
        allow_origins=origins,
        allow_credentials=True,
        allow_headers=["*"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        max_age=600,
    )
    if env.APP_ENV == "local":
        kwargs["allow_origin_regex"] = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    app.add_middleware(CORSMiddleware, **kwargs)
    if env.ENABLE_APM == 1:
        app.add_middleware(ElasticAPM, client=apm.client)
