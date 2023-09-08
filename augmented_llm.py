import json
import os
import requests
from datetime import datetime
from wiki_retriever import Retriever


class LLM:
    def __init__(self, model_path, port=8080):
        os.popen(f"llama.cpp/server --mlock -m {model_path} --port {port} -ngl 1 -t 4 --ctx-size 2048 &")
        self.url = f"http://127.0.0.1:{port}/completion"

    def create_completion(self, prompt, **kwargs):
        kwargs["prompt"] = prompt
        if "seed" not in kwargs:
            kwargs["seed"] = 1337
        kwargs["stream"] = True
        r = requests.post(self.url, json=kwargs, stream=True, timeout=60)
        for line in r.iter_lines(decode_unicode=True):
            if line:
                response = json.loads(line[5:])["content"]
                response = response.replace("ð", "")
                if len(response) > 0:
                    yield response

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
        for a in answer:
            yield a


class AugmentedLLM:
    def __init__(self, model_path, port=8080):
        self.retriever = Retriever("./wiki_bge_small_en_embeddings")
        self.llm = LLM(model_path, port)

    def ask(
        self, question, previous="", force_retrieval=False, generate_preanswer=False
    ):
        self.is_wiki_question_template = 'Question: "{q}"\n\nDoes Wikipedia answer this question? Answer only yes or no.: {a}'
        prefix = (
            f"### System:\nToday is {datetime.today().strftime('%d %B %Y')}.\n"
            + "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Instruction:\n\nAnswer the following human's question:\n\n"
            + previous
        )
        self.wiki_prompt = (
            prefix
            + "{question}\n\nRelevant paragraphs from wikipedia:\n{wikidocs}\n\nWrite the answer in your own words based on the Wikipedia paragprahs.\n\n### Response: "
        )
        self.prompt = prefix + "{question}\n\n### Response: "

        relevant_paragraphs = []
        prompt = self.prompt.format(question=question)

        is_wiki_question = self.is_wikipedia_question(question)
        if force_retrieval or is_wiki_question:
            if is_wiki_question:
                yield "*Looking into Wikipedia...* ", None
            wiki_query = question
            if generate_preanswer:
                gen = self.llm.ask(prompt, max_tokens=128)
                wiki_query = "".join(gen)
            relevant_paragraphs = self.retriever.search(wiki_query, k=3)
            yield "*Reading relevant paragraphs...*  \n", None
            wikidocs = self.render_docs(relevant_paragraphs)
            prompt = self.wiki_prompt.format(question=question, wikidocs=wikidocs)
        answer = self.llm.ask(prompt)
        for a in answer:
            yield a, relevant_paragraphs

    def is_wikipedia_question(self, question):
        question = question.strip().split("\n\n", maxsplit=1)[0]
        prompt = self.is_wiki_question_template.format(q=question, a="")
        gen = self.llm.ask(prompt, max_tokens=10, greedy=True)
        llm_response = ""
        for chunk in gen:
            llm_response += chunk.lower()
            if "yes" in llm_response:
                return True
        return False

    def render_docs(self, results):
        return "\n\n".join([f"{i+1}. {r['text']}" for i, r in enumerate(results)])
