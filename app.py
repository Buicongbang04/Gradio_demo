import gradio as gr
import warnings
warnings.filterwarnings("ignore")
import os
from huggingface_hub import InferenceClient
from functools import partial

system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, but do not make up answers. If you don't know the answer, just say that you don't know. Do not try to make up an answer."

def inference(prompt, hf_token, model, model_name):
    messages = [{"role": "system",
                 "content" : system_prompt},
                 {"role": "user",
                  "content" : prompt}]
    if hf_token is None or not hf_token.strip():
        hf_token = os.getenv("HF_TOKEN")
    
    client = InferenceClient(model=model, token=hf_token)
    tokens = f"**`{model_name}`**\n\n"

    for completion in client.chat_completion(messages, max_tokens=200, stream=True):
        tokens = completion.choices[0].delta.content
        tokens += token
        yield tokens

def hide_textbox():
    return gr.Textbox(visible=False)

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Gradio Demo</h1></center>")
    gr.Markdown("<center><h2>Gradio is a Python library that allows you to quickly create UIs for machine learning models.</h2></center>")


    prompt = gr.Textbox(label="Prompt", placeholder="Type something here...", lines=3, max_lines=5)
    token = gr.Textbox(label="Hugging Face Token", placeholder="Your Hugging Face token here...")

    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button(variant="primary", size="sm")
            code_btn = gr.Button("View Code", variant="secondary", size="sm")

    with gr.Row() as output_row:
        llama_output = gr.Markdown("LLaMA 3-70B Instruct")
        noust_output = gr.Markdown("Nous-Hermes 2 Mixtral 8x7B DPO")
        zephyr_output = gr.Markdown("Zephyr ORPO 141B A35B")

    # prompt.submit(
    #     fn=inference,
    #     inputs=[prompt, token],
    #     outputs=[llama_output, noust_output, zephyr_output],
    #     show_progress="hidden",
    # )
    # generate_btn.click(
    #     fn=inference,
    #     inputs=[prompt, token],
    #     outputs=[llama_output, noust_output, zephyr_output],
    #     show_progress="hidden",
    # )
    gr.on(
        triggers=[prompt.submit, generate_btn.click],
        fn=hide_textbox,
        inputs=None,
        outputs=[token],
    )

    gr.on(
        triggers=[prompt.submit, generate_btn.click],
        fn=partial(inference,
                   model="meta-llama/Meta-Llama-3-70b-Instruct",
                   model_name="LLaMA 3-70B Instruct"),
        inputs=[prompt, token],
        outputs=[llama_output],
        show_progress="hidden",
    )

demo.launch()