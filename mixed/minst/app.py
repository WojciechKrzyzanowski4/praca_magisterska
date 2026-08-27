from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mixed.minst.minst_dataset import (
    DEFAULT_EVO_OUTPUT_FILE,
    DEFAULT_GRAD_OUTPUT_FILE,
    DEFAULT_OUTPUT_DIR,
    MNISTBundle,
    dataset_summary,
    load_mnist,
    sample_visualization_indices,
)
from mixed.minst.minst_evo import EvolutionConfig, evolve_model, load_evo_model
from mixed.minst.minst_grad import load_grad_model, train_model
from mixed.minst.minst_model import load_model, predict_images, save_model


st.set_page_config(
    page_title="MNIST: GD vs neuroewolucja",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GD_COLOR = "#ffb000"
EVO_COLOR = "#ff4d8d"
CNN_CODE = """class SimpleCNN(Sequential):
    def __init__(self):
        super().__init__(
            Conv2DLayer(1, 16, kernel_size=3, padding=1), # [C1]
            ReLULayer(),                                  # [C2]
            MaxPool2DLayer(kernel_size=2, stride=2),      # [C3]
            Conv2DLayer(16, 32, kernel_size=3, padding=1),# [C4]
            ReLULayer(),                                  # [C5]
            MaxPool2DLayer(kernel_size=2, stride=2),      # [C6]
            FlattenLayer(),                               # [F1]
            DenseLayer(32 * 7 * 7, 128),                 # [F2]
            ReLULayer(),                                  # [F3]
            DenseLayer(128, 10),                          # [F4]
        )
"""

TRAINING_CODE = """for batch_images, batch_labels in iterate_minibatches(...):
    logits = model.forward(batch_images)                  # [T1]
    loss = loss_function.forward(logits, batch_labels)    # [T2]
    loss_gradient = loss_function.backward()              # [T3]
    model.backward(loss_gradient)                         # [T4]
    optimizer.step()                                      # [T5]
    optimizer.zero_grad()                                 # [T6]
"""


@dataclass
class TrainingTrace:
    epochs: list[int]
    train_losses: list[float]
    test_accuracies: list[float]
    training_time: float
    model_name: str = "GD"
    checkpoint_path: str | None = None


@st.cache_data(show_spinner=False)
def load_bundle(train_limit: int, test_limit: int) -> MNISTBundle:
    return load_mnist(
        normalize=True,
        flatten=False,
        train_limit=train_limit,
        test_limit=test_limit,
    )


def make_loss_figure(
    trace: TrainingTrace | None,
    *,
    title: str,
    color: str,
) -> go.Figure:
    fig = go.Figure()
    if trace is not None:
        fig.add_trace(
            go.Scatter(
                x=trace.epochs,
                y=trace.train_losses,
                mode="lines+markers",
                name=f"{trace.model_name} train loss",
                line=dict(color=color, width=4),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="krok",
        yaxis_title="CrossEntropy loss",
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def make_training_figure(
    gd_trace: TrainingTrace | None,
    evo_trace: TrainingTrace | None = None,
) -> go.Figure:
    fig = go.Figure()
    if gd_trace is not None:
        fig.add_trace(
            go.Scatter(
                x=gd_trace.epochs,
                y=gd_trace.train_losses,
                mode="lines+markers",
                name="GD loss train",
                line=dict(color=GD_COLOR, width=4),
            )
        )
    if evo_trace is not None:
        fig.add_trace(
            go.Scatter(
                x=evo_trace.epochs,
                y=evo_trace.train_losses,
                mode="lines+markers",
                name="EVO loss train",
                line=dict(color=EVO_COLOR, width=4),
            )
        )
    fig.update_layout(
        title="Porownanie funkcji straty",
        xaxis_title="krok treningu",
        yaxis_title="CrossEntropy loss",
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def make_confusion_figure(y_true: np.ndarray, y_pred: np.ndarray | None) -> go.Figure:
    matrix = np.zeros((10, 10), dtype=np.int64)
    if y_pred is not None:
        for expected, predicted in zip(y_true, y_pred):
            matrix[int(expected), int(predicted)] += 1
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(range(10)),
            y=list(range(10)),
            colorscale="Teal",
            hovertemplate="true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Macierz pomylek dla probki testowej",
        xaxis_title="predykcja",
        yaxis_title="etykieta",
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


def render_sample_grid(
    bundle: MNISTBundle,
    *,
    sample_count: int,
    seed: int,
    gd_checkpoint_path: str | None,
    evo_checkpoint_path: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    images = bundle.x_test
    labels = bundle.y_test
    indices = sample_visualization_indices(len(images), count=sample_count, seed=seed)
    gd_predictions = None
    gd_confidences = None
    evo_predictions = None
    evo_confidences = None

    if gd_checkpoint_path:
        try:
            model, _ = load_grad_model(gd_checkpoint_path)
            gd_predictions, gd_confidences = predict_images(model, images[indices])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.warning(f"Nie udalo sie wczytac checkpointu GD: {exc}")

    if evo_checkpoint_path:
        try:
            model, _ = load_evo_model(evo_checkpoint_path)
            evo_predictions, evo_confidences = predict_images(model, images[indices])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.warning(f"Nie udalo sie wczytac checkpointu EVO: {exc}")

    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            position = row * cols + col
            if position >= len(indices):
                continue
            idx = int(indices[position])
            label = int(labels[idx])
            caption = f"true={label}"
            if gd_predictions is not None and gd_confidences is not None:
                caption += f" | GD={int(gd_predictions[position])} ({float(gd_confidences[position]):.2f})"
            if evo_predictions is not None and evo_confidences is not None:
                caption += f" | EVO={int(evo_predictions[position])} ({float(evo_confidences[position]):.2f})"
            columns[col].image(images[idx], caption=caption, clamp=True, width="stretch")

    return gd_predictions, evo_predictions


def run_training(
    bundle: MNISTBundle,
    *,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    output_path: str,
    live_delay_ms: int,
    progress,
    status,
    gd_chart,
    evo_chart,
    comparison_chart,
    current_evo_trace: TrainingTrace | None = None,
) -> TrainingTrace:
    trace = TrainingTrace(epochs=[], train_losses=[], test_accuracies=[], training_time=0.0, model_name="GD")
    start_time = time.perf_counter()

    def update_interface(epoch, loss, accuracy, _model):
        trace.epochs.append(epoch)
        trace.train_losses.append(float(loss))
        trace.test_accuracies.append(float(accuracy))
        trace.training_time = time.perf_counter() - start_time

        progress.progress(epoch / max(1, epochs))
        status.write(f"GD epoch {epoch}/{epochs}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        gd_chart.plotly_chart(
            make_loss_figure(trace, title="Strata modelu gradientowego", color=GD_COLOR),
            width="stretch",
            key=f"live_gd_loss_{epoch}",
        )
        evo_chart.plotly_chart(
            make_loss_figure(current_evo_trace, title="Strata modelu ewolucyjnego", color=EVO_COLOR),
            width="stretch",
            key=f"live_evo_loss_during_gd_{epoch}",
        )
        comparison_chart.plotly_chart(make_training_figure(trace, current_evo_trace), width="stretch", key=f"live_training_gd_{epoch}")
        if live_delay_ms > 0:
            time.sleep(live_delay_ms / 1000.0)

    model, _ = train_model(
        bundle,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=42,
        callback=update_interface,
    )
    trace.checkpoint_path = save_model(model, output_path)
    return trace


def run_evolution_training(
    bundle: MNISTBundle,
    *,
    batch_size: int,
    generations: int,
    population_size: int,
    elite_count: int,
    mutation_std: float,
    mutation_scope: str,
    initialization: str,
    initial_checkpoint_path: str | None,
    eval_samples_per_class: int,
    diversity_weight: float,
    seed: int,
    output_path: str,
    live_delay_ms: int,
    progress,
    status,
    evo_chart,
    comparison_chart,
    current_gd_trace: TrainingTrace | None = None,
) -> TrainingTrace:
    base_model = None
    if initialization == "warm_start_gd" and initial_checkpoint_path and os.path.isfile(initial_checkpoint_path):
        base_model = load_model(initial_checkpoint_path)

    config = EvolutionConfig(
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        mutation_std=mutation_std,
        mutation_scope=mutation_scope,
        samples_per_class=eval_samples_per_class,
        diversity_weight=diversity_weight,
        batch_size=batch_size,
        seed=seed,
    )
    trace = TrainingTrace(epochs=[], train_losses=[], test_accuracies=[], training_time=0.0, model_name="EVO")
    start_time = time.perf_counter()

    def update_interface(
        generation,
        generation_loss,
        generation_train_accuracy,
        generation_test_accuracy,
        _model,
    ):
        trace.epochs.append(generation)
        trace.train_losses.append(float(generation_loss))
        trace.test_accuracies.append(float(generation_test_accuracy))
        trace.training_time = time.perf_counter() - start_time

        progress.progress(generation / max(1, generations))
        status.write(
            f"EVO generation {generation}/{generations}: "
            f"loss={generation_loss:.4f}, eval_acc={generation_train_accuracy:.4f}, "
            f"test_acc={generation_test_accuracy:.4f}"
        )
        evo_chart.plotly_chart(
            make_loss_figure(trace, title="Strata modelu ewolucyjnego", color=EVO_COLOR),
            width="stretch",
            key=f"live_evo_loss_{generation}",
        )
        comparison_chart.plotly_chart(make_training_figure(current_gd_trace, trace), width="stretch", key=f"live_training_evo_{generation}")
        if live_delay_ms > 0:
            time.sleep(live_delay_ms / 1000.0)

    best_model, _ = evolve_model(
        bundle,
        base_model=base_model,
        config=config,
        callback=update_interface,
    )
    trace.checkpoint_path = save_model(best_model, output_path)
    return trace


def render_dataset_metrics(bundle: MNISTBundle) -> None:
    summary = dataset_summary(bundle)
    train = summary["train"]
    test = summary["test"]
    metrics = st.columns(4)
    metrics[0].metric("Train samples", f"{bundle.x_train.shape[0]:,}".replace(",", " "))
    metrics[1].metric("Test samples", f"{bundle.x_test.shape[0]:,}".replace(",", " "))
    metrics[2].metric("Image shape", "28 x 28")
    metrics[3].metric("Range", f"{train['min']:.1f}-{train['max']:.1f}")
    st.caption(f"train shape={train['shape']} | test shape={test['shape']}")


def render_network_screen() -> None:
    st.subheader("Kod zrodlowy sieci CNN")
    st.code(CNN_CODE, language="python", line_numbers=True)

    st.subheader("Opis matematyczny")
    st.markdown("**[C1], [C4] Splot.** Filtry konwolucyjne przeksztalcaja obraz w mapy cech:")
    st.latex(r"h_{k,i,j}^{(\ell)}=\sigma\left(b_k+\sum_c\sum_{u,v}W_{k,c,u,v}^{(\ell)}x_{c,i+u,j+v}^{(\ell)}\right)")
    st.markdown("**[C3], [C6] Max pooling.** Warstwa zmniejsza rozdzielczosc map cech, zachowujac najsilniejsze odpowiedzi lokalne.")
    st.latex(r"h_{k,i,j}^{pool}=\max_{(u,v)\in P_{i,j}}h_{k,u,v}")
    st.markdown("**[F2]-[F4] Klasyfikator.** Po splaszczeniu wektor cech trafia do warstw liniowych, a wyjscie ma 10 logitow.")
    st.latex(r"z=f_\theta(x)\in\mathbb{R}^{10},\qquad \hat{y}=\arg\max_c z_c")


def render_training_screen() -> None:
    st.subheader("Kod zrodlowy funkcji treningowych")
    st.code(TRAINING_CODE, language="python", line_numbers=True)

    st.subheader("Funkcja straty")
    st.markdown("**[T4] Cross-entropy.** Dla klasyfikacji wieloklasowej minimalizowana jest ujemna log-wiarygodnosc poprawnej klasy:")
    st.latex(r"L(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp z_{i,y_i}}{\sum_{c=0}^{9}\exp z_{i,c}}")
    st.markdown("**[T5]-[T6] Backpropagation i Adam.** Gradient funkcji straty aktualizuje wszystkie filtry konwolucyjne oraz warstwy liniowe.")
    st.latex(r"\theta_{k+1}=\operatorname{Adam}(\theta_k,\nabla_\theta L(\theta_k),\alpha)")
    st.markdown("**[EVO] Mutacja i selekcja.** Populacja sieci jest oceniana funkcja cross-entropy, a kolejne pokolenie powstaje z elit oraz ich zaszumionych kopii.")
    st.latex(r"\theta'=\theta+\varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,\sigma^2 I)")
    st.latex(r"\theta^\star_g=\arg\min_{\theta\in P_g}L(\theta),\qquad P_{g+1}=\operatorname{elite}(P_g)\cup\operatorname{mutate}(\operatorname{elite}(P_g))")


def render_interactive_lab() -> None:

    default_grad_checkpoint = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_GRAD_OUTPUT_FILE)
    default_evo_checkpoint = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_EVO_OUTPUT_FILE)

    with st.expander("Parametry eksperymentu własnego", expanded=True):
        with st.expander("Dane", expanded=True):
            train_limit = st.slider("train_limit", 500, 60000, 5000, step=500)
            test_limit = st.slider("test_limit", 100, 10000, 1000, step=100)
            sample_count = st.slider("sample_count", 4, 24, 12, step=4)
            seed = st.number_input("sample_seed", value=42, step=1)
            st.caption("Własna implementacja NumPy działa na CPU.")
        with st.expander("Gradient", expanded=True):
            epochs = st.slider("gd_epochs", 1, 30, 3)
            batch_size = st.select_slider("gd_batch_size", options=[32, 64, 128, 256], value=64)
            learning_rate = st.number_input(
                "gd_learning_rate",
                min_value=1e-5,
                max_value=1e-1,
                value=1e-3,
                step=1e-4,
                format="%.5f",
            )
            grad_checkpoint_path = st.text_input("gd_checkpoint_path", value=default_grad_checkpoint)
        with st.expander("Ewolucja", expanded=True):
            evo_generations = st.slider("evo_generations", 1, 200, 12)
            evo_population_size = st.slider("evo_population_size", 2, 40, 12)
            evo_elite_count = st.slider("evo_elite_count", 1, 10, 2)
            evo_mutation_std = st.number_input(
                "evo_mutation_std",
                min_value=1e-4,
                max_value=0.5,
                value=0.01,
                step=0.005,
                format="%.4f",
            )
            evo_initialization = st.selectbox(
                "evo_initialization",
                options=["warm_start_gd", "random"],
            )
            evo_mutation_scope = st.selectbox(
                "evo_mutation_scope",
                options=["classifier", "all"],
            )
            evo_eval_samples_per_class = st.slider("evo_eval_samples_per_class", 5, 200, 40, step=5)
            evo_diversity_weight = st.number_input(
                "evo_diversity_weight",
                min_value=0.0,
                max_value=5.0,
                value=0.25,
                step=0.05,
                format="%.2f",
            )
            evo_checkpoint_path = st.text_input("evo_checkpoint_path", value=default_evo_checkpoint)
        live_delay_ms = st.slider("live_delay_ms", 0, 500, 60, step=20)
        start = st.button("Start", type="primary")
        reset = st.button("Reset")

    if reset:
        for key in ["mnist_gd_trace", "mnist_evo_trace", "mnist_grad_checkpoint", "mnist_evo_checkpoint"]:
            st.session_state.pop(key, None)
        st.rerun()

    bundle = load_bundle(int(train_limit), int(test_limit))
    gd_trace: TrainingTrace | None = st.session_state.get("mnist_gd_trace")
    evo_trace: TrainingTrace | None = st.session_state.get("mnist_evo_trace")
    active_grad_checkpoint = st.session_state.get("mnist_grad_checkpoint")
    active_evo_checkpoint = st.session_state.get("mnist_evo_checkpoint")

    experiment_tab, network_tab, training_tab = st.tabs(
        ["Eksperyment", "Siec CNN", "Funkcje treningowe"]
    )

    with network_tab:
        render_network_screen()

    with training_tab:
        render_training_screen()

    with experiment_tab:
        render_dataset_metrics(bundle)

        top_left, top_right = st.columns([1.0, 1.0])
        gd_loss_chart = top_left.empty()
        evo_loss_chart = top_right.empty()
        comparison_chart = st.empty()
        gd_loss_chart.plotly_chart(
            make_loss_figure(gd_trace, title="Strata modelu gradientowego", color=GD_COLOR),
            width="stretch",
            key="gd_loss_static",
        )
        evo_loss_chart.plotly_chart(
            make_loss_figure(evo_trace, title="Strata modelu ewolucyjnego", color=EVO_COLOR),
            width="stretch",
            key="evo_loss_static",
        )
        comparison_chart.plotly_chart(
            make_training_figure(gd_trace, evo_trace),
            width="stretch",
            key="training_static",
        )

        progress = st.progress(0.0)
        status = st.empty()

        if start:
            gd_trace = run_training(
                bundle,
                batch_size=int(batch_size),
                epochs=int(epochs),
                learning_rate=float(learning_rate),
                output_path=grad_checkpoint_path,
                live_delay_ms=int(live_delay_ms),
                progress=progress,
                status=status,
                gd_chart=gd_loss_chart,
                evo_chart=evo_loss_chart,
                comparison_chart=comparison_chart,
                current_evo_trace=None,
            )
            st.session_state["mnist_gd_trace"] = gd_trace
            st.session_state["mnist_grad_checkpoint"] = gd_trace.checkpoint_path
            active_grad_checkpoint = gd_trace.checkpoint_path
            evo_trace = run_evolution_training(
                bundle,
                batch_size=int(batch_size),
                generations=int(evo_generations),
                population_size=int(evo_population_size),
                elite_count=int(evo_elite_count),
                mutation_std=float(evo_mutation_std),
                mutation_scope=str(evo_mutation_scope),
                initialization=str(evo_initialization),
                initial_checkpoint_path=active_grad_checkpoint,
                eval_samples_per_class=int(evo_eval_samples_per_class),
                diversity_weight=float(evo_diversity_weight),
                seed=int(seed),
                output_path=evo_checkpoint_path,
                live_delay_ms=int(live_delay_ms),
                progress=progress,
                status=status,
                evo_chart=evo_loss_chart,
                comparison_chart=comparison_chart,
                current_gd_trace=gd_trace,
            )
            st.session_state["mnist_evo_trace"] = evo_trace
            st.session_state["mnist_evo_checkpoint"] = evo_trace.checkpoint_path
            active_evo_checkpoint = evo_trace.checkpoint_path
            status.write(f"Gotowe. GD: {gd_trace.checkpoint_path} | EVO: {evo_trace.checkpoint_path}")
            progress.progress(1.0)

        left, right = st.columns([1.15, 1.0])
        with left:
            st.subheader("Predykcje na probce testowej")
            gd_predictions, evo_predictions = render_sample_grid(
                bundle,
                sample_count=int(sample_count),
                seed=int(seed),
                gd_checkpoint_path=(
                    active_grad_checkpoint
                    if active_grad_checkpoint and os.path.isfile(active_grad_checkpoint)
                    else None
                ),
                evo_checkpoint_path=(
                    active_evo_checkpoint
                    if active_evo_checkpoint and os.path.isfile(active_evo_checkpoint)
                    else None
                ),
            )
        with right:
            indices = sample_visualization_indices(len(bundle.x_test), count=int(sample_count), seed=int(seed))
            y_true = bundle.y_test[indices]
            confusion_source = gd_predictions if gd_predictions is not None else evo_predictions
            right.plotly_chart(make_confusion_figure(y_true, confusion_source), width="stretch", key="confusion")

        tables = []
        for trace in (gd_trace, evo_trace):
            if trace is None:
                continue
            tables.append(
                pd.DataFrame(
                    {
                        "model": trace.model_name,
                        "step": trace.epochs,
                        "train_loss": trace.train_losses,
                        "test_accuracy": trace.test_accuracies,
                        "training_time": trace.training_time,
                    }
                )
            )
        if tables:
            st.dataframe(pd.concat(tables).tail(20), width="stretch")


def main() -> None:
    st.title("MNIST: GD vs neuroewolucja")
    st.caption(
        "Interaktywne laboratorium do ręcznej konfiguracji i obserwacji uczenia. "
        "Oficjalny protokół badawczy znajduje się w mixed.minst.experiment."
    )
    render_interactive_lab()


if __name__ == "__main__":
    main()
