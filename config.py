# config.py
from dataclasses import dataclass, field # added from dataclasses import dataclass, field from pathlib import Path import os
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv() #load_dotenv() call outside the class

@dataclass #dataclass. This automatically generates methods like __init__
class Config: #class Config:     def __init__(self):	(Removed)
    # 1) API
    mistral_api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", "")) #The key is now fetched dynamically from the environment variable MISTRAL_API_KEY using os.getenv()

    # 2) Processing parameters #chunk size and overlap were increased (from 1000/100 to 1200/200)
    chunk_size: int = 1200          
    chunk_overlap: int = 200       

    # 3) Paths #using pathlib.Path and __post_init__. This approach calculates all paths relative to a dynamic root_dir (the directory of config.py), 
    # ensuring the paths work regardless of where the script is run. The init=False flag means these paths are calculated later in __post_init__
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    results_summaries_dir: Path = field(init=False)
    results_analyses_dir: Path = field(init=False)

    # 4) Model params
    model_name: str = "mistral-large-latest"
    temperature: float = 0.2 #Increased the temperature from 0.1 to 0.2,
    max_tokens: int = 256

    # 5) Logging #Added a logging configuration setting.
    log_level: str = "INFO"

    def __post_init__(self): #__post_init__. This dataclass method is called after __init__ is automatically generated and called
        self.data_dir = self.root_dir / "data" #Path Calculation. All application paths are defined here relative to self.root_dir
        self.processed_dir = self.root_dir / "data" / "processed"
        self.results_summaries_dir = self.root_dir / "results" / "summaries"
        self.results_analyses_dir = self.root_dir / "results" / "analyses"

        for p in [self.data_dir, self.processed_dir, self.results_summaries_dir, self.results_analyses_dir]: p.mkdir(parents=True, exist_ok=True) 
        #ensures all required directories exist Config            

        self.validate() #call to a validation method

    def validate(self): #separate method for configuration validation.
        if not self.mistral_api_key: #
            raise ValueError( 
                "Missing MISTRAL_API_KEY. Add it to your environment or a .env file."
            )
        if self.chunk_overlap >= self.chunk_size: #Validation Check
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
