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
        output = self.model(prompt, max_tokens=max_tokens)
        answer = output["choices"][0]["text"].strip()
        return answer


class AugmentedLLM:
    def __init__(self, model_path="./models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q4_K_M.bin"):
        self.retriever = Retriever("./wiki_mpnet_index")
        self.llm = LLM(model_path)

        self.is_question_template = 'Question: "{q}"\nIs wikipedia question:\n{a}'
        self.wiki_prompt = "Question: {question}\n\nWikipedia suggestions:\n{wikidocs}\n\nWrite the answer in your own words based on the given suggestions from wikipedia.\n\nAnswer: "
        self.prompt = "Question: {question}\n\nAnswer: "

        wiki_question_examples = [
            ("What is your name?", "no"),
            ("What is your age?", "no"),
            ("What is the capital of France?", "yes"),
            ("What is the meaning of life?", "yes"),
            ("Are you a human?", "no"),
            ("Where do you live?", "no"),
            ("How pathetic", "no"),
            ("What is the best movie of all time?", "yes"),
            ("I am tired", "no"),
            ("Tell me about quantum mechanics", "yes"),
            ("Name 5 most popular programming languages", "yes"),
        ]
        self.is_wiki_question_fewshots = "\n\n".join(
            [self.is_question_template.format(q=q, a=a) for q, a in wiki_question_examples]
        ) + "\n\n"


    def ask(self, question, force_retrieval=False):
        if force_retrieval or self.is_wikipedia_question(question):
            relevant_paragraphs = self.retriever.search(question, k=3)
            wikidocs = self.render_docs(relevant_paragraphs)
            prompt = self.wiki_prompt.format(question=question, wikidocs=wikidocs)
        else:
            relevant_paragraphs = []
            prompt = self.prompt.format(question=question)

        answer = self.llm.ask(prompt)
        return answer, relevant_paragraphs

    def is_wikipedia_question(self, question):
        question = question.strip().split("\n")[0]
        prompt = self.is_wiki_question_fewshots + self.is_question_template.format(q=question, a="")
        llm_response = self.llm.ask(prompt, max_tokens=10).lower()
        return "yes" in llm_response

    def render_docs(self, results):
        return "\n\n".join([f"{i+1}. {r['text']}" for i, r in enumerate(results)])
