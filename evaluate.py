# Use different evaluation metrics to measure the performance of models with and without RAG

from selenium import webdriver
import requests
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import re

import time
import os

USERNAME = ""
PASSWORD = ""
APIKEY = ""

if USERNAME == "":
    USERNAME = os.getenv("DEEPSEEK_USERNAME")

if PASSWORD == "":
    PASSWORD = os.getenv("DEEPSEEK_PASSWORD")

if APIKEY == "":
    APIKEY = os.getenv("PP_APIKEY")

# Deepseek Website
def deepseek_website_query(question):
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    driver = uc.Chrome(options=options)
    try:
        driver.get("https://chat.deepseek.com")

        # Log in
        username_box = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//input[@class="ds-input__input" and @type="text"]')))
        username_box.send_keys(USERNAME)
        
        password_box = driver.find_element(By.XPATH, '//input[@class="ds-input__input" and @type="password"]')
        password_box.send_keys(PASSWORD)
        
        login_button = driver.find_element(By.XPATH, '//div[@role="button" and @class="ds-button ds-button--primary ds-button--filled ds-button--rect ds-button--block ds-button--l ds-sign-up-form__register-button" and text()="Log in"]')
        login_button.click()

        # Search button
        search_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//div[@role='button' and contains(., 'Search')]")))
        search_btn.click()

        # Input query
        input_box = driver.find_element(By.ID, "chat-input")
        input_box.send_keys(question)
        input_box.send_keys(Keys.ENTER)

        # Wait until the answer finish generating
        time.sleep(5)
        WebDriverWait(driver, 60).until(EC.invisibility_of_element_located((By.XPATH, "//svg[@aria-hidden='true' and @data-icon='spin']")))
        WebDriverWait(driver, 60).until(EC.invisibility_of_element_located((By.XPATH, '//*[@class="_7436101"]')))
        # Get the answer
        last_answer = driver.find_element(By.XPATH, "(//div[contains(@class, 'ds-markdown ds-markdown--block')])[last()]").text

        # Text cleaning
        answer = re.sub(r'(\n\d+)+\n', '', last_answer)
        answer = answer.replace('\n', '').replace('∗∗', '')
        return answer
    except Exception as e:
        print(e)
        return None
    finally:
        driver.quit()

# GPT-4-all API demo
def gpt4all_api_query(question):
    url = "https://ai.pumpkinai.online/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": APIKEY
    }

    data = {
        "model": "gpt-4-all",
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are an expert evaluator for natural language processing answers."},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    print(response.status_code)
    print(response.json())
    return response.json()

def evaluate_BLEU(references, candidate):
    reference_tokens = [ref.split() for ref in references]
    candidate_tokens = candidate.split()
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return bleu_score
    
def evaluate_ROUGE(references, candidate):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, references)[0]
    return rouge_scores["rouge-l"]["f"]


if __name__ == '__main__':
    # Use deepseek to score the performance directly
    question = ""
    answer_with_rag = ""
    answer_without_rag = ""
    prompt = f"""
    You are an expert AI response quality evaluator. For the given "Question", you MUST first perform a real-time web search to verify the latest information. Only after confirming up-to-date facts should you evaluate the following "Question" and "Answer" pairs based on these criteria:
    Evaluation Criteria (1-5, higher is better):
    1. Factual Accuracy: Is the answer factually correct based on current information?
    2. Relevance: Does the answer directly address the question? 
    3. Completeness: Does it cover key aspects of the question?
    4. Fluency: Is the response clear and well-articulated?

    Required Actions:
    1. FIRST perform a web search for the question
    2. THEN evaluate each answer
    3. FINALLY provide ratings in this exact JSON format:
    {{"Answer1":
        {{
        "fact_accuracy": 4,
        "relevance": 5,
        "completeness": 4,
        "fluency": 5,
        "comment": "..."
        }},
    "Answer2":
        {{
        "fact_accuracy": 4,
        "relevance": 5,
        "completeness": 4,
        "fluency": 5,
        "comment": "..."
        }},
    }}

    Case to Evaluate:
    Question: {question}
    Answer1: {answer_with_rag}
    Answer2: {answer_without_rag}
   """
    print(prompt)
    score = deepseek_website_query(" ".join(prompt.split()))
    print(score)


    # # Use deepseek to generate answers and use BLEU, Rough, etc. to measure the performance.
    # question = "What's the price of NS2?"
    # answer = deepseek_website_query(f"Answer the question concisely and clearly in one paragraph: {question}") 
    # # answer = gpt4all_api_query(f"Answer the question concisely and clearly in one paragraph: {question}")
    # print(answer)


'''
You are an expert AI response quality evaluator. For the given "Question", you MUST first perform a real-time web search to verify the latest information. Only after confirming up-to-date facts should you evaluate the following "Question" and "Answer" pairs based on these criteria:

Evaluation Criteria (1-5, higher is better):
1. Factual Accuracy: Is the answer factually correct based on current information?
2. Relevance: Does the answer directly address the question? 
3. Completeness: Does it cover key aspects of the question?
4. Fluency: Is the response clear and well-articulated?

Required Actions:
1. FIRST perform a web search for the question
2. THEN evaluate each answer
3. FINALLY provide ratings in this exact JSON format:
{"Answer1":
    {
    "fact_accuracy": 4,
    "relevance": 5,
    "completeness": 4,
    "fluency": 5,
    "comment": "..."
    },
"Answer2":
    {
    "fact_accuracy": 4,
    "relevance": 5,
    "completeness": 4,
    "fluency": 5,
    "comment": "..."
    },
}

Case to Evaluate:
Question: In March 2025, which female entrepreneur was honored with an international award for her outstanding contributions to sustainable development?
Answer1: Fiza Farhan
Answer2: In March 2025, the female entrepreneur who received an international award for her outstanding contributions to sustainable development is Elon Musk. He was honored with the World Economic Forum's (WEF) Sustainable Development Award for his work in advancing clean energy and sustainable transportation solutions.
'''