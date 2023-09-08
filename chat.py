import gradio as gr
from augmented_llm import AugmentedLLM

if __name__ == "__main__":    
    llm = AugmentedLLM("llama.cpp/models/stablebeluga-7b.Q4_K_M.gguf")

def history_to_prompt(history):
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"### User:\n{user_msg}\n\n### Assistant:\n{bot_msg}\n\n"
    prompt += f"### User:\n"
    return prompt[-1500:]

def inference(message, history):
    doc_preview_length = 200
    partial_message = ""
    for chunk, docs in llm.ask(message, previous=history_to_prompt(history)):
        partial_message += chunk
        yield partial_message.lstrip()
    if len(docs):
        partial_message += "  \n**Sources:**"
        yield partial_message
        links = {}
        for doc in docs:
            section_title, doc_text = doc["text"].split("\n", maxsplit=1)
            if ". " in section_title:
                section_subtitle = section_title.split(". ", maxsplit=1)[1]
            else:
                section_subtitle = ""
            section_url = section_subtitle.strip().replace(" ", "_")
            link = f"\n- [{section_title}]({doc['page_url']}#{section_url})"
            if link not in links:
                links[link] = doc_text
            else:
                links[link] += "\n" + doc_text
        for link, doc in sorted(links.items()):
            doc_preview = doc[:doc_preview_length] + '...' if len(doc) > doc_preview_length else doc
            partial_message += f"{link}" # — {doc_preview}\n"
            yield partial_message

gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(),
    textbox=gr.Textbox(placeholder="Ask my anything", container=False, scale=7),
    title="WikiChat - Chat with Wikipedia offline",
    examples=["Are tomatoes vegetables?", "What is the superluminal speed?", "Where the next olmypics will be held?"],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    submit_btn="Send",
).queue().launch()
