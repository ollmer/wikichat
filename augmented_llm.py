from datetime import datetime
from wiki_retriever import Retriever
from llama_cpp import Llama

class LLM:
    def __init__(self, model_path):
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=2000000,
            n_threads=10,
            n_ctx=2048,
            verbose=False,
        )

    def ask(self, prompt, max_tokens=300):
        if len(prompt) > 3000:
            prompt = prompt[-3000:]
        output = self.model.create_completion(prompt, max_tokens=max_tokens, temperature=0.3)
        answer = output["choices"][0]["text"].strip()
        return answer

    def greedy(self, prompt, max_tokens=128):
        if len(prompt) > 3000:
            prompt = prompt[-3000:]
        output = self.model.create_completion(prompt, max_tokens=max_tokens, temperature=0.1, top_k=1, repeat_penalty=1.3)
        answer = output["choices"][0]["text"].strip()
        return answer


class AugmentedLLM:
    def __init__(self, model_path="./models/stable-platypus2-13b.ggmlv3.q4_K_M.bin"):
        self.retriever = Retriever("./wiki_bge_small_en_embeddings")
        self.llm = LLM(model_path)

    def ask(self, question, force_retrieval=False):
        self.is_wiki_question_template = 'Question: "{q}"\n\nDoes Wikipedia answer this question? Answer only yes or no.: {a}'        
        today = f"Today is {datetime.today().strftime('%d %B %Y')}.\n"
        prefix = today + "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Instruction:\n\nAnswer the following human's question:\n\n"
        self.wiki_prompt = prefix + "{question}\n\nRelevant paragraphs from wikipedia:\n{wikidocs}\n\nWrite the answer in your own words based on the Wikipedia paragprahs.\n\n### Response: "
        self.prompt = prefix + "{question}\n\n### Response: "

        relevant_paragraphs = []
        prompt = self.prompt.format(question=question)
        
        if force_retrieval or self.is_wikipedia_question(question):
            preanswer = self.llm.ask(prompt, max_tokens=256)
            relevant_paragraphs = self.retriever.search(preanswer, k=3)
            wikidocs = self.render_docs(relevant_paragraphs)
            prompt = self.wiki_prompt.format(question=question, wikidocs=wikidocs)
        
        answer = self.llm.ask(prompt)
        return answer, relevant_paragraphs

    def is_wikipedia_question(self, question):
        question = question.strip().split("\n\n", maxsplit=1)[0]
        prompt = self.is_wiki_question_template.format(q=question, a="")
        llm_response = self.llm.greedy(prompt, max_tokens=10)
        return "yes" in llm_response.lower()

    def render_docs(self, results):
        return "\n\n".join([f"{i+1}. {r['text']}" for i, r in enumerate(results)])
