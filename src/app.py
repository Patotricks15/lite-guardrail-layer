from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import streamlit as st

from predict import (
    DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
    DEFAULT_PROMPT_INJECTION_ARTIFACT,
    DEFAULT_TOXICITY_ARTIFACT,
    load_artifact,
    load_model,
    predict,
)
from fine_tune_system_safes import run_fine_tuning


st.set_page_config(page_title="Prompt Safety Detector", page_icon=":shield:")


@st.cache_resource(show_spinner=False)
def load_detection_resources():
    artifact_specs = (
        (DEFAULT_PROMPT_INJECTION_ARTIFACT, "prompt_injection"),
        (DEFAULT_OUT_OF_CONTEXT_ARTIFACT, "out_of_context"),
        (DEFAULT_TOXICITY_ARTIFACT, "toxicity"),
    )
    artifacts = tuple(
        load_artifact(str(Path(path).resolve()), task)
        for path, task in artifact_specs
    )
    model_names = {artifact["embedding_model_name"] for artifact in artifacts}
    if len(model_names) != 1:
        raise ValueError("all artifacts must use the same embedding model")
    embedding_model = load_model(model_names.pop())
    return artifacts, embedding_model


try:
    with st.spinner("Loading detection models..."):
        load_detection_resources()
except Exception as error:
    st.error(f"Could not load detection models: {error}")
    st.stop()

st.title("Prompt Safety Detector")
analysis_tab, fine_tuning_tab, models_tab = st.tabs(
    ["Análise", "Fine-tuning", "Meus modelos"]
)
models_root = Path("artifact/fine_tuned")
model_dirs = (
    sorted(path for path in models_root.iterdir() if path.is_dir())
    if models_root.exists()
    else []
)


def load_model_system_prompt(model_name: str) -> str:
    if model_name == "Modelo base":
        return ""
    prompt_path = models_root / model_name / "system_prompt.txt"
    if not prompt_path.is_file():
        return ""
    return prompt_path.read_text(encoding="utf-8")

with models_tab:
    st.subheader("Meus modelos")
    if not model_dirs:
        st.info("Nenhum modelo fine-tuned foi criado ainda.")
    else:
        for model_dir in model_dirs:
            artifacts = sorted(model_dir.glob("*_model.pkl"))
            st.write(f"**{model_dir.name}**: {len(artifacts)}/3 modelos prontos")
            st.caption(str(model_dir))

with fine_tuning_tab:
    st.subheader("Criar modelo fine-tuned")
    with st.form("fine_tuning_form"):
        model_name = st.text_input("Nome do modelo", placeholder="meu_modelo")
        safe_file = st.file_uploader("Exemplos safe (.xlsx)", type=["xlsx"])
        out_of_context_file = st.file_uploader(
            "Exemplos fora de contexto (.xlsx, opcional)",
            type=["xlsx"],
        )
        system_file = st.file_uploader("System prompt (.txt)", type=["txt"])
        safe_column = st.text_input("Coluna de exemplos", value="UserMessage")
        execute_fine_tuning = st.form_submit_button("Executar", type="primary")

    if execute_fine_tuning:
        if (
            not model_name.strip()
            or not safe_file
            or not system_file
        ):
            st.error("Informe o nome e envie o system prompt e o Excel safe.")
        elif Path(model_name.strip()).name != model_name.strip():
            st.error("O nome deve ser simples, sem pastas ou separadores.")
        else:
            try:
                with TemporaryDirectory() as temporary_dir:
                    temporary_path = Path(temporary_dir)
                    safe_path = temporary_path / "safe_examples.xlsx"
                    system_path = temporary_path / "system_prompt.txt"
                    safe_path.write_bytes(safe_file.getvalue())
                    system_path.write_bytes(system_file.getvalue())
                    system_prompt = system_path.read_text(encoding="utf-8")
                    progress = st.progress(0, text="Preparando fine-tuning...")

                    def update_progress(done, total, task_name):
                        progress.progress(
                            done / total,
                            text=f"Processando {task_name}...",
                        )

                    fine_tuning_kwargs = {
                        "safe_column": safe_column,
                        "progress_callback": update_progress,
                        "system_prompt_source_path": system_path,
                    }
                    if out_of_context_file:
                        out_of_context_path = (
                            temporary_path / "out_of_context_examples.xlsx"
                        )
                        out_of_context_path.write_bytes(out_of_context_file.getvalue())
                        fine_tuning_kwargs.update(
                            {
                                "out_of_context_excel_paths": [out_of_context_path],
                                "out_of_context_column": safe_column,
                            }
                        )
                    result = run_fine_tuning(
                        model_name.strip(),
                        system_prompt,
                        [safe_path],
                        **fine_tuning_kwargs,
                    )
                    progress.progress(1.0, text="Fine-tuning concluído")
                st.success(f"Modelo '{result['system_name']}' criado com sucesso.")
                st.json(result)
            except Exception as error:
                st.error(f"Fine-tuning falhou: {error}")

with analysis_tab:
    st.subheader("Analisar prompt")
    available_models = ["Modelo base"] + [path.name for path in model_dirs]
    selected_model = st.selectbox("Modelo", available_models)
    default_system_prompt = load_model_system_prompt(selected_model)
    prompt_key = f"system_prompt_{selected_model}"
    if st.session_state.get("selected_analysis_model") != selected_model:
        st.session_state[prompt_key] = default_system_prompt
        st.session_state.selected_analysis_model = selected_model
    with st.form("prediction_form"):
        system_prompt = st.text_area(
            "System prompt",
            height=160,
            key=prompt_key,
        )
        user_prompt = st.text_area("User prompt", height=160)
        submitted = st.form_submit_button("Analisar", type="primary")

    if submitted:
        if not system_prompt.strip() or not user_prompt.strip():
            st.error("System prompt e user prompt são obrigatórios.")
        else:
            try:
                with st.spinner("Analisando..."):
                    started_at = perf_counter()
                    artifacts_dir = (
                        models_root / selected_model
                        if selected_model != "Modelo base"
                        else None
                    )
                    artifact_paths = [
                        path if artifacts_dir is None else artifacts_dir / path.name
                        for path in (
                            DEFAULT_PROMPT_INJECTION_ARTIFACT,
                            DEFAULT_OUT_OF_CONTEXT_ARTIFACT,
                            DEFAULT_TOXICITY_ARTIFACT,
                        )
                    ]
                    result = predict(system_prompt, user_prompt, *artifact_paths)
                    result["execution_time_ms"] = round(
                        (perf_counter() - started_at) * 1000, 3
                    )
                st.json(result)
            except Exception as error:
                st.error(f"Análise falhou: {error}")