import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

# Configure environment paths before importing heavy dependencies so child modules see them
hf_cache_dir = os.path.join(cmd_args.model_dir, "hf_cache")
torch_cache_dir = os.path.join(cmd_args.model_dir, "torch_cache")
os.environ.setdefault("INDEXTTS_USE_DEEPSPEED", "0")
os.environ.setdefault("HF_HOME", hf_cache_dir)
os.environ.setdefault("HF_HUB_CACHE", hf_cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache_dir)
os.environ.setdefault("TORCH_HOME", torch_cache_dir)
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(torch_cache_dir, exist_ok=True)

import gradio as gr
from indextts import infer
from indextts.infer_v2_modded import IndexTTS2
from tools.i18n.i18n import I18nAuto
from modelscope.hub import api

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    is_fp16=cmd_args.is_fp16,
    use_cuda_kernel=False,
)

logger = logging.getLogger(__name__)

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES = [
    "Match prompt audio",
    "Use emotion reference audio",
    "Use emotion vector",
    "Use emotion text description",
]
os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0)]
                             )


def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_sentence=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
    if emo_control_method == 1:
        emo_weight = emo_weight
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        if vec_sum > 1.5:
            gr.Warning("Emotion vector sum cannot exceed 1.5. Adjust the sliders and retry.")
            return
    else:
        vec = None

    print(f"Emo control mode:{emo_control_method},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                       **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    batch_rows_state = gr.State([])
    next_batch_id_state = gr.State(1)
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Accordion("Emotion Settings", open=True):
        with gr.Row():
            emo_control_method = gr.Radio(
                choices=EMO_CHOICES,
                type="index",
                value=EMO_CHOICES[0],
                label="Emotion Control Mode",
            )

    with gr.Group(visible=False) as emotion_reference_group:
        with gr.Row():
            emo_upload = gr.Audio(label="Emotion Reference Audio", type="filepath")
        with gr.Row():
            emo_weight = gr.Slider(label="Emotion Weight", minimum=0.0, maximum=1.6, value=0.8, step=0.01)

    with gr.Row():
        emo_random = gr.Checkbox(label="Random Emotion Sampling", value=False, visible=False)

    with gr.Group(visible=False) as emotion_vector_group:
        with gr.Row():
            with gr.Column():
                vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
            with gr.Column():
                vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec6 = gr.Slider(label="Low Mood", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.4, value=0.0, step=0.05)

    with gr.Group(visible=False) as emo_text_group:
        with gr.Row():
            emo_text = gr.Textbox(label="Emotion Description", placeholder="Describe the target emotion", value="", info="e.g., happy, angry, sad")

    with gr.Accordion("Advanced Generation Settings", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**GPT2 Sampling Settings** _Parameters affect diversity and speed._")
                with gr.Row():
                    do_sample = gr.Checkbox(label="do_sample", value=True, info="Enable sampling")
                    temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                with gr.Row():
                    top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                    num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                with gr.Row():
                    repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                    length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info="Maximum generated mel tokens")
            with gr.Column(scale=2):
                gr.Markdown("**Sentence Settings** _Controls sentence splitting._")
                with gr.Row():
                    max_text_tokens_per_sentence = gr.Slider(
                        label="Max tokens per sentence",
                        value=120,
                        minimum=20,
                        maximum=tts.cfg.gpt.max_text_tokens,
                        step=2,
                        key="max_text_tokens_per_sentence",
                        info="Higher values mean longer sentences; adjust between 80-200",
                    )
                with gr.Accordion("Preview sentences", open=True) as sentences_settings:
                    sentences_preview = gr.Dataframe(
                        headers=["Index", "Sentence", "Token Count"],
                        key="sentences_preview",
                        wrap=True,
                    )
        advanced_params = [
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
        ]

    with gr.Tabs():
        with gr.Tab("Single Generation"):
            with gr.Row():
                prompt_audio = gr.Audio(label="Prompt Audio", key="prompt_audio",
                                        sources=["upload", "microphone"], type="filepath")
                with gr.Column():
                    input_text_single = gr.TextArea(label="Text", key="input_text_single", placeholder="Enter text to synthesize", info=f"Model version {tts.model_version or '1.0'}")
                    gen_button = gr.Button("Generate", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="Generated Result", visible=True, key="output_audio")

            if len(example_cases) > 0:
                gr.Examples(
                    examples=example_cases,
                    examples_per_page=20,
                    inputs=[prompt_audio,
                            emo_control_method,
                            input_text_single,
                            emo_upload,
                            emo_weight,
                            emo_text,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
                )

        with gr.Tab("Batch Generation"):
            gr.Markdown("Manage multiple prompt audios, give each its own text, generate in bulk, and retry specific entries as needed.")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        dataset_path_input = gr.Textbox(
                            label="Dataset train.txt path",
                            value="vivy_va_dataset/train.txt",
                            scale=3,
                            placeholder="Path to train.txt"
                        )
                        load_dataset_button = gr.Button("Load Dataset", scale=1)
                    batch_file_input = gr.Files(
                        label="Add prompt audio files",
                        file_types=["audio"],
                        file_count="multiple",
                        type="filepath"
                    )
                    batch_table = gr.Dataframe(
                        headers=["ID", "Prompt", "Text", "Output", "Status", "Last Generated"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        row_count=(0, "dynamic"),
                        col_count=6,
                        interactive=False,
                        value=[]
                    )
                with gr.Column():
                    selected_entry = gr.Dropdown(label="Select entry", choices=[], value=None, interactive=True)
                    batch_prompt_player = gr.Audio(label="Prompt Audio", type="filepath", interactive=False)
                    batch_output_player = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
                    batch_text_input = gr.TextArea(label="Text", placeholder="Enter text for this entry", interactive=True)
                    batch_status = gr.Markdown(value="No entry selected.")
                    with gr.Row():
                        generate_all_button = gr.Button("Generate All")
                        regenerate_button = gr.Button("Regenerate Selected")
                    with gr.Row():
                        delete_entry_button = gr.Button("Delete Selected")
                        clear_entries_button = gr.Button("Clear All")

    def on_input_text_change(text, max_tokens_per_sentence):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            sentences = tts.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=int(max_tokens_per_sentence))
            data = []
            for i, s in enumerate(sentences):
                sentence_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, sentence_str, tokens_count])
            return {
                sentences_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            return {
                sentences_preview: gr.update(value=[], visible=True, type="array"),
            }
    def on_method_select(emo_control_method):
        if emo_control_method == 1:
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 2:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 3:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        else:
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    def build_batch_table_data(rows):
        table_data = []
        for row in rows or []:
            text_preview = row.get("text", "") or ""
            if text_preview and len(text_preview) > 60:
                text_preview = text_preview[:57] + "..."
            table_data.append([
                row.get("id"),
                os.path.basename(row.get("prompt_path", "")) if row.get("prompt_path") else "",
                text_preview,
                os.path.basename(row.get("output_path", "")) if row.get("output_path") else "",
                row.get("status", "Pending"),
                row.get("last_generated", "")
            ])
        return table_data

    def find_batch_row(rows, row_id):
        if row_id is None:
            return None
        for row in rows or []:
            if row.get("id") == row_id:
                return row
        return None

    def resolve_batch_selection(rows, selected_value):
        choices = [str(row.get("id")) for row in rows or []]
        if not choices:
            return gr.update(choices=[], value=None), None
        if selected_value is not None:
            selected_str = str(selected_value)
            if selected_str in choices:
                return gr.update(choices=choices, value=selected_str), int(selected_str)
        selected_str = choices[-1]
        return gr.update(choices=choices, value=selected_str), int(selected_str)

    def prepare_batch_selection(rows, selected_value):
        dropdown_update, resolved_id = resolve_batch_selection(rows, selected_value)
        row = find_batch_row(rows, resolved_id)
        prompt_update = gr.update(value=row.get("prompt_path") if row else None)
        output_update = gr.update(value=row.get("output_path") if row and row.get("output_path") else None)
        text_update = gr.update(value=row.get("text", "") if row else "")
        return dropdown_update, resolved_id, prompt_update, output_update, text_update, row

    def format_batch_status(row, message=None):
        if not row:
            base = "No entry selected."
        else:
            lines = [f"Row {row.get('id')}: {row.get('status', 'Pending')}"]
            if row.get("text"):
                preview = row.get("text")
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                lines.append(f"Text: {preview}")
            if row.get("output_path"):
                lines.append(f"Output: {row.get('output_path')}")
            if row.get("last_generated"):
                lines.append(f"Last generated: {row.get('last_generated')}")
            base = "\n".join(lines)
        if message:
            if base:
                base = f"{base}\n{message}"
            else:
                base = message
        return gr.update(value=base)

    def add_batch_prompts(files, rows, next_id, selected_value):
        rows = rows or []
        next_id = next_id or 1
        files = files or []
        updated_rows = [dict(row) for row in rows]
        prompts_dir = os.path.abspath(os.path.join(current_dir, "prompts"))
        os.makedirs(prompts_dir, exist_ok=True)
        added = 0
        last_added_id = None
        for file_path in files:
            if not file_path:
                continue
            safe_name = os.path.basename(file_path)
            timestamp = int(time.time() * 1000)
            target_name = f"batch_prompt_{next_id}_{timestamp}_{safe_name}"
            target_path = os.path.join(prompts_dir, target_name)
            try:
                shutil.copy(file_path, target_path)
            except Exception as exc:
                logger.exception("Failed to store prompt %s", file_path)
                gr.Warning(f"Failed to add {safe_name}: {exc}")
                continue
            entry = {
                "id": next_id,
                "prompt_path": target_path,
                "output_path": None,
                "status": "Pending",
                "last_generated": "",
                "text": "",
            }
            updated_rows.append(entry)
            added += 1
            last_added_id = entry["id"]
            next_id += 1
        selected_seed = last_added_id if added else selected_value
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(updated_rows, selected_seed)
        table_update = gr.update(value=build_batch_table_data(updated_rows))
        status_message = None
        if added:
            status_message = f"Added {added} prompt{'s' if added != 1 else ''}."
        elif files:
            status_message = "No new prompts were added."
        status_update = format_batch_status(selected_row, status_message)
        return updated_rows, next_id, gr.update(value=None), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def load_dataset_entries(dataset_path, rows, next_id, selected_value, progress=gr.Progress()):
        rows = rows or []
        next_id = next_id or 1
        dataset_path = (dataset_path or "").strip()
        if not dataset_path:
            gr.Warning("Provide a dataset train.txt path before loading.")
            dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, next_id, gr.update(value=""), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        dataset_path_abs = dataset_path if os.path.isabs(dataset_path) else os.path.abspath(os.path.join(current_dir, dataset_path))
        if not os.path.exists(dataset_path_abs):
            gr.Warning(f"Dataset file not found: {dataset_path_abs}")
            dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        dataset_dir = os.path.dirname(dataset_path_abs)
        audio_dirs = [dataset_dir, os.path.join(dataset_dir, "wavs"), os.path.join(dataset_dir, "audio")]

        try:
            lines = Path(dataset_path_abs).read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            gr.Warning(f"Failed to read dataset file: {exc}")
            dropdown_update, _, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        updated_rows = [dict(row) for row in rows]
        prompts_dir = os.path.abspath(os.path.join(current_dir, "prompts"))
        os.makedirs(prompts_dir, exist_ok=True)

        existing_prompts = {os.path.basename(r.get("prompt_path", "")) for r in updated_rows if r.get("prompt_path")}

        total_lines = len(lines)
        added = 0
        skipped_missing_audio = 0
        skipped_invalid = 0

        progress(0.0, desc="Parsing dataset")
        for idx, line in enumerate(lines):
            progress(min((idx + 1) / max(total_lines, 1), 0.95), desc=f"Processing line {idx + 1}/{total_lines}")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split("|", 1)
            if len(parts) != 2:
                skipped_invalid += 1
                continue
            audio_name = parts[0].strip()
            text_value = parts[1].strip()
            if not audio_name or not text_value:
                skipped_invalid += 1
                continue

            audio_source = None
            for base_dir in audio_dirs:
                candidate = os.path.join(base_dir, audio_name)
                if os.path.exists(candidate):
                    audio_source = candidate
                    break
            if not audio_source:
                skipped_missing_audio += 1
                continue

            unique_suffix = f"dataset_{next_id}_{int(time.time() * 1000)}"
            target_name = f"{unique_suffix}_{os.path.basename(audio_name)}"
            if target_name in existing_prompts:
                target_name = f"{unique_suffix}_{next_id}_{os.path.basename(audio_name)}"
            target_path = os.path.join(prompts_dir, target_name)
            try:
                shutil.copy(audio_source, target_path)
            except Exception as exc:
                logger.exception("Failed to copy dataset prompt %s", audio_source)
                gr.Warning(f"Failed to copy {audio_name}: {exc}")
                skipped_missing_audio += 1
                continue

            entry = {
                "id": next_id,
                "prompt_path": target_path,
                "output_path": None,
                "status": "Pending",
                "last_generated": "",
                "text": text_value,
            }
            updated_rows.append(entry)
            existing_prompts.add(target_name)
            added += 1
            next_id += 1

        selected_seed = updated_rows[-1]["id"] if added else selected_value
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(updated_rows, selected_seed)
        table_update = gr.update(value=build_batch_table_data(updated_rows))

        status_parts = []
        if added:
            status_parts.append(f"Loaded {added} entries")
        if skipped_missing_audio:
            status_parts.append(f"{skipped_missing_audio} missing audio")
        if skipped_invalid:
            status_parts.append(f"{skipped_invalid} invalid lines")
        status_message = ", ".join(status_parts) if status_parts else "No new entries loaded."
        status_update = format_batch_status(selected_row, status_message)
        progress(1.0, desc="Dataset load complete")
        return updated_rows, next_id, gr.update(value=dataset_path), table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def validate_emotion_settings(emo_control_method, vec_values):
        mode = emo_control_method if isinstance(emo_control_method, int) else getattr(emo_control_method, "value", 0)
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            mode = 0
        vec = None
        if mode == 2:
            vec = vec_values
            if sum(vec_values) > 1.5:
                gr.Warning("Emotion vector sum cannot exceed 1.5. Adjust the sliders and retry.")
                return mode, None
        return mode, vec

    def build_generation_kwargs(do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens):
        try:
            top_k_value = int(top_k)
        except (TypeError, ValueError):
            top_k_value = 0
        try:
            num_beams_value = int(num_beams)
        except (TypeError, ValueError):
            num_beams_value = 1
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": top_k_value if top_k_value > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams_value,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        return kwargs

    def generate_all_batch(rows, selected_value, emo_control_method, emo_ref_path, emo_weight,
                           vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                           emo_text, emo_random, max_text_tokens_per_sentence,
                           do_sample, top_p, top_k, temperature,
                           length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                           progress=gr.Progress()):
        rows = rows or []
        if not rows:
            gr.Warning("Add prompt audio files before generating.")
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        vec_values = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        mode, vec = validate_emotion_settings(emo_control_method, vec_values)
        if mode == 2 and vec is None:
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(row, "Emotion vector sum exceeded limit.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        try:
            max_tokens = int(max_text_tokens_per_sentence)
        except (TypeError, ValueError):
            max_tokens = 120

        generation_kwargs = build_generation_kwargs(do_sample, top_p, top_k, temperature,
                                                    length_penalty, num_beams, repetition_penalty, max_mel_tokens)

        outputs_dir = os.path.abspath(os.path.join(current_dir, "outputs", "tasks"))
        os.makedirs(outputs_dir, exist_ok=True)

        updated_rows = []
        total = len(rows)
        success_count = 0
        progress(0.0, desc="Starting batch generation")
        for idx, row in enumerate(rows):
            new_row = dict(row)
            prompt_path = new_row.get("prompt_path")
            if not prompt_path or not os.path.exists(prompt_path):
                new_row["status"] = "Error: Prompt missing"
                updated_rows.append(new_row)
                continue
            text_value = (new_row.get("text") or "").strip()
            if not text_value:
                new_row["status"] = "Error: Text missing"
                updated_rows.append(new_row)
                continue
            output_filename = f"batch_row_{new_row['id']}_{int(time.time() * 1000)}.wav"
            output_path = os.path.join(outputs_dir, output_filename)
            try:
                tts.gr_progress = progress
                tts.infer(
                    spk_audio_prompt=prompt_path,
                    text=text_value,
                    output_path=output_path,
                    emo_audio_prompt=emo_ref_path if mode == 1 else None,
                    emo_alpha=float(emo_weight) if mode == 1 else 1.0,
                    emo_vector=vec if mode == 2 else None,
                    use_emo_text=(mode == 3), emo_text=emo_text,
                    use_random=emo_random,
                    verbose=cmd_args.verbose,
                    max_text_tokens_per_sentence=max_tokens,
                    **generation_kwargs,
                )
                new_row["output_path"] = output_path
                new_row["status"] = "Completed"
                new_row["last_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                success_count += 1
            except Exception as exc:
                logger.exception("Batch generation failed for row %s", new_row.get("id"))
                new_row["status"] = f"Error: {exc}"
            updated_rows.append(new_row)
            progress(min((idx + 1) / total, 1.0), desc=f"Generated {idx + 1}/{total}")

        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(updated_rows, selected_value)
        table_update = gr.update(value=build_batch_table_data(updated_rows))
        status_message = f"Generated {success_count}/{total} entr{'ies' if total != 1 else 'y'}."
        status_update = format_batch_status(row, status_message)
        return updated_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def regenerate_batch_entry(rows, selected_value, emo_control_method, emo_ref_path, emo_weight,
                               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                               emo_text, emo_random, max_text_tokens_per_sentence,
                               do_sample, top_p, top_k, temperature,
                               length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                               progress=gr.Progress()):
        rows = rows or []
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(rows, selected_value)
        if not selected_row:
            gr.Warning("Select an entry to regenerate.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(None)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        vec_values = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        mode, vec = validate_emotion_settings(emo_control_method, vec_values)
        if mode == 2 and vec is None:
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(selected_row, "Emotion vector sum exceeded limit.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        try:
            max_tokens = int(max_text_tokens_per_sentence)
        except (TypeError, ValueError):
            max_tokens = 120

        generation_kwargs = build_generation_kwargs(do_sample, top_p, top_k, temperature,
                                                    length_penalty, num_beams, repetition_penalty, max_mel_tokens)

        prompt_path = selected_row.get("prompt_path")
        if not prompt_path or not os.path.exists(prompt_path):
            gr.Warning("Prompt audio file is missing.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(selected_row, "Prompt audio file missing.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        text_value = (selected_row.get("text") or "").strip()
        if not text_value:
            gr.Warning("Enter text for this entry before regenerating.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(selected_row, "Text is missing.")
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        outputs_dir = os.path.abspath(os.path.join(current_dir, "outputs", "tasks"))
        os.makedirs(outputs_dir, exist_ok=True)
        output_filename = f"batch_row_{selected_row['id']}_{int(time.time() * 1000)}.wav"
        output_path = os.path.join(outputs_dir, output_filename)
        result_rows = []
        for row in rows:
            if row.get("id") != selected_row.get("id"):
                result_rows.append(dict(row))
                continue
            updated_row = dict(row)
            try:
                tts.gr_progress = progress
                tts.infer(
                    spk_audio_prompt=prompt_path,
                    text=text_value,
                    output_path=output_path,
                    emo_audio_prompt=emo_ref_path if mode == 1 else None,
                    emo_alpha=float(emo_weight) if mode == 1 else 1.0,
                    emo_vector=vec if mode == 2 else None,
                    use_emo_text=(mode == 3), emo_text=emo_text,
                    use_random=emo_random,
                    verbose=cmd_args.verbose,
                    max_text_tokens_per_sentence=max_tokens,
                    **generation_kwargs,
                )
                updated_row["output_path"] = output_path
                updated_row["status"] = "Completed"
                updated_row["last_generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as exc:
                logger.exception("Regeneration failed for row %s", updated_row.get("id"))
                updated_row["status"] = f"Error: {exc}"
            result_rows.append(updated_row)

        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(result_rows, selected_value)
        table_update = gr.update(value=build_batch_table_data(result_rows))
        status_update = format_batch_status(row, "Regeneration finished.")
        return result_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def delete_batch_entry(rows, selected_value):
        rows = rows or []
        dropdown_update, resolved_id, prompt_update, output_update, text_update, selected_row = prepare_batch_selection(rows, selected_value)
        if not selected_row:
            gr.Warning("Select an entry to delete.")
            table_update = gr.update(value=build_batch_table_data(rows))
            status_update = format_batch_status(None)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update
        remaining_rows = [dict(row) for row in rows if row.get("id") != selected_row.get("id")]
        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(remaining_rows, None)
        table_update = gr.update(value=build_batch_table_data(remaining_rows))
        status_update = format_batch_status(row, "Entry deleted.")
        return remaining_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    def clear_batch_rows(rows, next_id):
        dropdown_update = gr.update(choices=[], value=None)
        prompt_update = gr.update(value=None)
        output_update = gr.update(value=None)
        text_update = gr.update(value="")
        status_update = format_batch_status(None, "Batch list cleared.")
        return [], 1, gr.update(value=[]), dropdown_update, prompt_update, output_update, text_update, status_update

    def on_select_batch_entry(selected_value, rows):
        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
        status_update = format_batch_status(row)
        return dropdown_update, prompt_update, output_update, text_update, status_update

    def update_batch_text(new_text, rows, selected_value):
        rows = rows or []
        try:
            selected_id = int(selected_value) if selected_value is not None else None
        except (TypeError, ValueError):
            selected_id = None

        if selected_id is None:
            gr.Warning("Select an entry before editing text.")
            table_update = gr.update(value=build_batch_table_data(rows))
            dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(rows, selected_value)
            status_update = format_batch_status(row)
            return rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

        updated_rows = []
        target_row = None
        for row in rows:
            row_id = row.get("id")
            new_row = dict(row)
            if row_id == selected_id:
                new_row["text"] = new_text
                if new_row.get("output_path"):
                    new_row["status"] = "Pending"
                target_row = new_row
            updated_rows.append(new_row)

        dropdown_update, resolved_id, prompt_update, output_update, text_update, row = prepare_batch_selection(updated_rows, selected_id)
        table_update = gr.update(value=build_batch_table_data(updated_rows))
        status_message = "Text updated. Regenerate to apply." if target_row else ""
        status_update = format_batch_status(row, status_message if status_message else None)
        return updated_rows, table_update, dropdown_update, prompt_update, output_update, text_update, status_update

    emo_control_method.select(on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emo_random,
                 emotion_vector_group,
                 emo_text_group]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    max_text_tokens_per_sentence.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_sentence,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    batch_file_input.upload(
        add_batch_prompts,
        inputs=[batch_file_input, batch_rows_state, next_batch_id_state, selected_entry],
        outputs=[batch_rows_state, next_batch_id_state, batch_file_input, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    load_dataset_button.click(
        load_dataset_entries,
        inputs=[dataset_path_input, batch_rows_state, next_batch_id_state, selected_entry],
        outputs=[batch_rows_state, next_batch_id_state, dataset_path_input, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    selected_entry.change(
        on_select_batch_entry,
        inputs=[selected_entry, batch_rows_state],
        outputs=[selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    batch_text_input.change(
        update_batch_text,
        inputs=[batch_text_input, batch_rows_state, selected_entry],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    generate_all_button.click(
        generate_all_batch,
        inputs=[batch_rows_state, selected_entry, emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                *advanced_params],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    regenerate_button.click(
        regenerate_batch_entry,
        inputs=[batch_rows_state, selected_entry, emo_control_method, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random, max_text_tokens_per_sentence,
                *advanced_params],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    delete_entry_button.click(
        delete_batch_entry,
        inputs=[batch_rows_state, selected_entry],
        outputs=[batch_rows_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )

    clear_entries_button.click(
        clear_batch_rows,
        inputs=[batch_rows_state, next_batch_id_state],
        outputs=[batch_rows_state, next_batch_id_state, batch_table, selected_entry, batch_prompt_player, batch_output_player, batch_text_input, batch_status]
    )



if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
