from langchain_ollama import OllamaLLM
from app import rag_answer

judge_llm = OllamaLLM(model="llama3.1")  # same as your RAG model

def evaluate_answer(input_text, output_text, reference_text):
    prompt = f"""
You are an evaluator. Check if the following answer is correct.
Question: {input_text}
Answer: {output_text}
Reference: {reference_text}

Respond with:
- CORRECT if it matches
- INCORRECT if it does not
"""
    verdict = judge_llm.invoke(prompt)
    return verdict

input_q = "What are symptoms of diabetes?"
output = rag_answer(input_q)
reference = "Frequent urination, increased thirst, fatigue, weight loss, slow wound healing, blurry vision"

print("Judge verdict:", evaluate_answer(input_q, output, reference))
