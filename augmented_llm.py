import os
import requests
from datetime import datetime
from wiki_retriever import Retriever


class LLM:
    def __init__(self, model_path):
        os.popen(f"llama.cpp/server --mlock -m {model_path} -t 4  --ctx-size 2048 &")
        self.url = "http://127.0.0.1:8080/completion"

    def create_completion(self, prompt, **kwargs):
        kwargs["prompt"] = prompt
        if "seed" not in kwargs:
            kwargs["seed"] = 1337
        r = requests.post(self.url, json=kwargs)
        response = r.json()
        return response["content"].strip()

    def ask(self, prompt, max_tokens=300, greedy=False, max_context=3000):
        if len(prompt) > max_context:
            prompt = prompt[-max_context:]
        if greedy:
            answer = self.create_completion(
                prompt,
                n_predict=max_tokens,
                temperature=0.1,
                top_k=1,
                repeat_penalty=1.3,
            )
        else:
            answer = self.create_completion(
                prompt, n_predict=max_tokens, temperature=0.3
            )
        return answer


class AugmentedLLM:
    def __init__(
        self, model_path="llama.cpp/models/stablebeluga-7b.ggmlv3.q5_K_M.gguf.bin"
    ):
        self.retriever = Retriever("./wiki_bge_small_en_embeddings")
        self.llm = LLM(model_path)

    def ask(self, question, force_retrieval=False, generate_preanswer=False):
        self.is_wiki_question_template = 'Question: "{q}"\n\nDoes Wikipedia answer this question? Answer only yes or no.: {a}'
        today = f"Today is {datetime.today().strftime('%d %B %Y')}.\n"
        prefix = (
            today
            + "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Instruction:\n\nAnswer the following human's question:\n\n"
        )
        self.wiki_prompt = (
            prefix
            + "{question}\n\nRelevant paragraphs from wikipedia:\n{wikidocs}\n\nWrite the answer in your own words based on the Wikipedia paragprahs.\n\n### Response: "
        )
        self.prompt = prefix + "{question}\n\n### Response: "

        relevant_paragraphs = []
        prompt = self.prompt.format(question=question)

        if force_retrieval or self.is_wikipedia_question(question):
            wiki_query = question
            if generate_preanswer:
                wiki_query = self.llm.ask(prompt, max_tokens=128)
            relevant_paragraphs = self.retriever.search(wiki_query, k=3)
            wikidocs = self.render_docs(relevant_paragraphs)
            prompt = self.wiki_prompt.format(question=question, wikidocs=wikidocs)
        answer = self.llm.ask(prompt)
        return answer, relevant_paragraphs

    def is_wikipedia_question(self, question):
        question = question.strip().split("\n\n", maxsplit=1)[0]
        prompt = self.is_wiki_question_template.format(q=question, a="")
        llm_response = self.llm.ask(prompt, max_tokens=10, greedy=True)
        return "yes" in llm_response.lower()

    def render_docs(self, results):
        return "\n\n".join([f"{i+1}. {r['text']}" for i, r in enumerate(results)])
