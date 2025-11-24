# User gives a text document -> generate notes and a quiz -> combine and give to user
from re import template
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model = 'gpt-5.1')
model2 = ChatOpenAI(model = 'gpt-5.1')
#model2 = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

prompt1 = PromptTemplate(
    template = 'Generate short and simple notes from the following text \n {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Genarate 5 short Q&A from following text \n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} \n quiz -> {quiz}',
    input_variables = ['notes','quiz']
)

parser = StrOutputParser()

#Two parts -> Parallel chain part, Merge chain part

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser 

chain = parallel_chain | merge_chain

with open('sample_text.txt') as f:
    text = "".join(f.readlines())

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()

'''
# Linear Regression: Notes and Quiz

---

## Notes

- Linear regression is a statistical model that shows the relationship between a dependent variable (response) and one or more independent variables (explanatory variables).

- Simple linear regression: one explanatory variable.
  Multiple linear regression: two or more explanatory variables.

- Multivariate linear regression is different: it predicts multiple dependent variables at the same time.

- Linear regression uses a linear function (often the conditional mean) of the predictors to model the response.

- It is a supervised machine learning algorithm that learns from labeled data to make predictions on new data.

- Linear regression is popular because:
  - It is easier to fit than nonlinear models.
  - Its statistical properties are easier to analyze.

- Main uses:
  1. Prediction/forecasting: build a model to predict the response for new values of the predictors.
  2. Explanation: understand how much variation in the response is explained by the predictors and which predictors matter.

- Common fitting method: least squares (minimizes the sum of squared errors).

- Variants/alternatives:
  - Least absolute deviations (uses absolute errors).
  - Ridge regression (L2 penalty).
  - Lasso (L1 penalty).

- MSE (Mean Squared Error) gives large weight to big errors; it can be bad when there are many outliers. In such cases, more robust cost functions are preferred.

- Least squares can also be used for models that are not linear, so “least squares” and “linear model” are not the same thing.

---

## Quiz

1. **Q:** What is linear regression in statistics?
   **A:** It is a model that estimates the relationship between a scalar response (dependent variable) and one or more explanatory (independent) variables using a linear function.

2. **Q:** How do simple and multiple linear regression differ?
   **A:** Simple linear regression has exactly one explanatory variable, while multiple linear regression has two or more explanatory variables.

3. **Q:** How is multiple linear regression different from multivariate linear regression?
   **A:** Multiple linear regression predicts one dependent variable from several independent variables, whereas multivariate linear regression predicts multiple dependent variables simultaneously.

4. **Q:** Why is linear regression considered a supervised machine learning algorithm?
   **A:** Because it learns from labeled data (inputs with known outputs) to find an optimized linear function that can make predictions on new data.

5. **Q:** What is the most common method used to fit linear regression models?
   **A:** The least squares approach, which minimizes the sum of squared differences between observed and predicted values.
            +---------------------------+
            | Parallel<notes,quiz>Input |
            +---------------------------+
                 **               **
              ***                   ***
            **                         **
+----------------+                +----------------+
| PromptTemplate |                | PromptTemplate |
+----------------+                +----------------+
          *                               *
          *                               *
          *                               *
  +------------+                    +------------+
  | ChatOpenAI |                    | ChatOpenAI |
  +------------+                    +------------+
          *                               *
          *                               *
          *                               *
+-----------------+              +-----------------+
| StrOutputParser |              | StrOutputParser |
+-----------------+              +-----------------+
                 **               **
                   ***         ***
                      **     **
           +----------------------------+
           | Parallel<notes,quiz>Output |
           +----------------------------+
                          *
                          *
                          *
                 +----------------+
                 | PromptTemplate |
                 +----------------+
                          *
                          *
                          *
                   +------------+
                   | ChatOpenAI |
                   +------------+
                          *
                          *
                          *
                +-----------------+
                | StrOutputParser |
                +-----------------+
                          *
                          *
                          *
              +-----------------------+
              | StrOutputParserOutput |
              +-----------------------+
'''