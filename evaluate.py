# Use different evaluation metrics to measure the performance of models with and without RAG

from selenium import webdriver
from seleniumbase import SB
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI
import re

import time
import os

USERNAME = ""
PASSWORD = ""
APIKEY = "" # only necessary when using API

if USERNAME == "":
    USERNAME = os.getenv("DEEPSEEK_USERNAME")

if PASSWORD == "":
    PASSWORD = os.getenv("DEEPSEEK_PASSWORD")

if APIKEY == "":
    APIKEY = os.getenv("DEEPSEEK_APIKEY")

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
        # WebDriverWait(driver, 60).until(EC.invisibility_of_element_located((By.XPATH, "//svg[@aria-hidden='true' and @data-icon='spin']")))
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

# Deepseek API demo (the search function is not supported)
def deepseek_api_query(question):
    client = OpenAI(api_key=APIKEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": question},
        ],
        stream=False
    )
    return response.choices[0].message.content

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
    question = "What's the price of NS2?"
    answer = deepseek_website_query(f"Answer the question concisely and clearly in one paragraph: {question}") 
    print(repr(answer))