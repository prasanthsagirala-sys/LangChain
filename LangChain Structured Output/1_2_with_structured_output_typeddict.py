from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

#Schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in review"]
    summary: Annotated[str, "A brief summary of review"] 
    sentiment: Annotated[Literal["pos","neg"], "Return sentiment of Review. Positive, Negative or Neutral"] 
    #Selects output from the given Literals
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

structured_model = model.with_structured_output(Review)

prompt = '''The hardware is great, but the software feels bloated. 
There are too many pre-installed apps that i can't remove. Also, the UI looks outdated compared to other brands.
Hoping for a software update to fix this.
'''

result = structured_model.invoke('''The product is really great!
''') # As there are no pros or cons, it will be skipped

print(result)

