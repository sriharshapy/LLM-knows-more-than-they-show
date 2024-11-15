import pandas as pd
import re

# Sample data
data = {
    'Text': [
        """<s>[INST] <<SYS>> System prompt: This is a medical question-answering session. <</SYS>> What is (are) Leigh's Disease ? [/INST] Leigh's disease is a rare inherited neurometabolic disorder that affects the central nervous system. This progressive disorder begins in infants between the ages of three months and two years. Rarely, it occurs in teenagers and adults. Leigh's disease can be caused by mutations in mitochondrial DNA or by deficiencies of an enzyme called pyruvate dehydrogenase. Symptoms of Leigh's disease usually progress rapidly. The earliest signs may be poor sucking ability, and the loss of head control and motor skills. These symptoms may be accompanied by loss of appetite, vomiting, irritability, continuous crying, and seizures. As the disorder progresses, symptoms may also include generalized weakness, lack of muscle tone, and episodes of lactic acidosis, which can lead to impairment of respiratory and kidney function. In Leigh's disease, genetic mutations in mitochondrial DNA interfere with the energy sources that run cells in an area of the brain that plays a role in motor movements. The primary function of mitochondria is to convert the energy in glucose and fatty acids into a substance called adenosine triphosphate (ATP). The energy in ATP drives virtually all of a cell's metabolic functions. Genetic mutations in mitochondrial DNA, therefore, result in a chronic lack of energy in these cells, which in turn affects the central nervous system and causes progressive degeneration of motor functions. There is also a form of Leigh's disease (called X-linked Leigh's disease) which is the result of mutations in a gene that produces another group of substances that are important for cell metabolism. This gene is only found on the X chromosome. </s>"""
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to extract question and answer
def extract_qa(text):
    # Remove system prompt
    text_no_sys = re.sub(r'<<SYS>>.*?<</SYS>>', '', text, flags=re.DOTALL)

    # Extract question
    question_match = re.search(r'\[INST\](.*?)\[/INST\]', text_no_sys, flags=re.DOTALL)
    question = question_match.group(1).strip() if question_match else ''

    # Extract answer
    answer_match = re.search(r'\[/INST\](.*?)</s>', text_no_sys, flags=re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ''

    return pd.Series({'Question': question, 'Answer': answer})

# Apply the function to the DataFrame
df[['Question', 'Answer']] = df['Text'].apply(extract_qa)

# Display the resulting DataFrame
print(df[['Question', 'Answer']])
