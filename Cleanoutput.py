from pydantic import BaseModel, validator
import re

# Clean the summary using Pydantic
class CleanedSummary(BaseModel):
    summary: str

    @validator("summary")
    def clean_summary(cls, v):
        # Remove markdown bold formatting (**) but retain links
        v = re.sub(r'\*\*(.*?)\*\*', r'\1', v)  # Removes markdown bold formatting (**) but retains text
        v = re.sub(r'\n', ' ', v)  # Remove newline characters to make the summary a single paragraph
        return v.strip()