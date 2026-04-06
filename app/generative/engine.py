from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from config.setting import env
from config.credentials import google_credential
from langchain_aws import ChatBedrock
import boto3

class GenAI:
    def __init__(self):
        self.project = env.GOOGLE_PROJECT_NAME
        try:
            self.credentials = google_credential()
        except Exception:
            self.credentials = None
    
    def chatGgenai(self, model, think: bool=False, streaming: bool=False):
        budget = -1 if think else 0
        return ChatGoogleGenerativeAI(
            model=model, 
            temperature=0, 
            project=self.project, 
            location=env.GOOGLE_LOCATION_NAME,
            credentials=self.credentials,
            streaming=streaming,
            thinking_budget=budget,
        )

    def chatAzureOpenAi(
        self,
        model: str,
        deployment: str = "003",
        disable_temperature: bool = False,
        temperature: float = 0.0,
        **kwargs
    ) -> AzureChatOpenAI:
        version_configs = {
            "002": {
                "api_key": env.AZURE_API_KEY_002,
                "api_version": env.AZURE_API_VERSION_002,
                "azure_endpoint": env.AZURE_ENDPOINT_002,
            },
            "003": {
                "api_key": env.AZURE_API_KEY,
                "api_version": env.AZURE_API_VERSION,
                "azure_endpoint": env.AZURE_ENDPOINT,
            },
            "dev": {
                "api_key": env.AZURE_API_KEY_DEV,
                "api_version": env.AZURE_API_VERSION_DEV,
                "azure_endpoint": env.AZURE_ENDPOINT_DEV,
            }
        }

        args = {
            "model": model,
            "temperature": temperature,
            **version_configs.get(deployment, {}),
            **kwargs,
        }

        if disable_temperature or deployment == "dev":
            args.pop("temperature", None)
        return AzureChatOpenAI(**args)
            
    def chatBedrock(
        self,
        model: str = env.CLAUDE_3_7_SONNET_MODEL,
        temperature: float = 0.0,
        region_name: str = env.AWS_REGION,
        aws_access_key_id: str = env.AWS_ACCESS_KEY_ID,
        aws_secret_access_key: str = env.AWS_SECRET_ACCESS_KEY,
        return_session: bool = False,
        **kwargs
    ) -> ChatBedrock:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        return ChatBedrock(
            model_id = model,
            model_kwargs={"temperature": temperature},
            client=session.client('bedrock-runtime'),
            region_name=region_name,
            **kwargs
        ) if not return_session else session

    def chatLiteLLM(self, model: str = None, temperature: float = 0.0, **kwargs) -> ChatOpenAI:
        return ChatOpenAI(
            model=model or env.LLM_MODEL,
            base_url=env.LLM_BASE_URL,
            api_key=env.LLM_API_KEY,
            temperature=temperature,
            **kwargs,
        )
