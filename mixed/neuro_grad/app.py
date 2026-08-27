from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mixed.neuro_grad.data import (
    CURVE_FUNCTIONS,
    SURFACE_FUNCTIONS,
    TARGET_FUNCTIONS,
    generate_data,
)
from mixed.neuro_grad.training import (
    ExperimentConfig,
    TrainingTrace,
    make_seeded_model_factory,
    train_clipped_gd,
    train_zero_temp_mc,
)


st.set_page_config(
    page_title="GD vs neuroewolucja",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_COLOR = "#d7dee8"
TARGET_COLOR = "#45d483"
GD_COLOR = "#ffb000"
MC_COLOR = "rgba(0, 209, 255, 0.34)"
MC_MEAN_COLOR = "#ff4d8d"
NETWORK_CODE = """class FullyConnectedRegressor(Sequential):
    def __init__(self, depth, width, input_size=1, output_size=1):
        layers = [
            DenseLayer(input_size, width),              # [M1]
            TanhLayer(),                                # [M4]
        ]
        for _ in range(depth - 1):
            layers.extend([
                DenseLayer(width, width),               # [M2]
                TanhLayer(),                            # [M4]
            ])
        layers.append(DenseLayer(width, output_size))   # [M3]
        super().__init__(*layers)


class Sequential:
    def forward(self, x):
        state = x                                      # [M5]
        for layer in self.layers:
            state = layer(state)                       # [M6]
        return state                                   # [M7]

"""

SURFACE_NETWORK_CODE = """class FullyConnectedSurfaceRegressor(FullyConnectedRegressor):
    def __init__(self, depth, width):
        super().__init__(
            depth=depth,
            width=width,
            input_size=2,                              # [S1]
            output_size=1                              # [S2]
        )

# Training data has rows [x1, x2], and target values z.
# The same forward pass is used as in the 1D model.
"""

TRAINING_CODE = """def train_clipped_gd(model, data, config):
    loss_function = MSELoss()
    optimizer = SGD(model.parameters(), config.learning_rate)

    prediction = model.forward(data.x)                 # [GD1]
    loss = loss_function(prediction, data.y)           # [GD2]
    model.backward(loss_function.backward())           # [GD3]

    grad_norm = sqrt(sum(p.grad ** 2 for p in params)) # [GD4]
    for parameter in model.parameters():
        parameter.grad /= grad_norm
    optimizer.step()                                   # [GD5]
    optimizer.zero_grad()


def train_zero_temp_mc(model_factory, data, config):
    sigma = lr * sqrt(2*pi)                            # [MC1]
    old_loss = mse(model.forward(data.x), data.y)       # [MC2]

    for epoch in range(1, epochs + 1):
        old_state = copy(model.parameters())           # [MC3]
        for parameter in model.parameters():
            parameter.data += Normal(0, sigma)         # [MC4]

        new_loss = mse(model.forward(data.x), data.y)   # [MC5]
        if new_loss <= old_loss:                       # [MC6]
            old_loss = new_loss
        else:
            restore(old_state)                         # [MC7]
"""


def _mean_array(arrays: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(arrays, axis=0), axis=0)


def _final_mc_trace(mc_traces: list[TrainingTrace]) -> TrainingTrace | None:
    if not mc_traces:
        return None
    min_len = min(len(trace.epochs) for trace in mc_traces)
    aligned = [trace for trace in mc_traces if len(trace.epochs) >= min_len]
    return TrainingTrace(
        epochs=aligned[0].epochs[:min_len],
        losses=_mean_array([trace.losses[:min_len] for trace in aligned]),
        params=_mean_array([trace.params[:min_len] for trace in aligned]),
        predictions=_mean_array([trace.predictions[:min_len] for trace in aligned]),
        training_time=sum(trace.training_time for trace in aligned),
    )


def make_prediction_figure(
    x: np.ndarray,
    y: np.ndarray,
    clean_y: np.ndarray,
    target_name: str,
    gd_trace: TrainingTrace | None,
    mc_traces: list[TrainingTrace],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x.reshape(-1),
            y=y.reshape(-1),
            mode="markers",
            name="dane treningowe",
            marker=dict(size=5, color=DATA_COLOR, opacity=0.72),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x.reshape(-1),
            y=clean_y.reshape(-1),
            mode="lines",
            name=target_name,
            line=dict(color=TARGET_COLOR, width=2, dash="dot"),
        )
    )
    if gd_trace is not None and gd_trace.predictions.size:
        fig.add_trace(
            go.Scatter(
                x=x.reshape(-1),
                y=gd_trace.predictions[-1].reshape(-1),
                mode="lines",
                name="GD",
                line=dict(color=GD_COLOR, width=4),
            )
        )
    if mc_traces:
        mean_prediction = _mean_array([trace.predictions[-1] for trace in mc_traces])
        fig.add_trace(
            go.Scatter(
                x=x.reshape(-1),
                y=mean_prediction.reshape(-1),
                mode="lines",
                name="srednia MC",
                line=dict(color=MC_MEAN_COLOR, width=4),
            )
        )
        for idx, trace in enumerate(mc_traces[:3]):
            fig.add_trace(
                go.Scatter(
                    x=x.reshape(-1),
                    y=trace.predictions[-1].reshape(-1),
                    mode="lines",
                    name=f"MC {idx + 1}",
                    line=dict(color=MC_COLOR, width=1.4),
                    showlegend=idx == 0,
                )
            )
    fig.update_layout(
        title="Dopasowanie modelu",
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(l=10, r=10, t=45, b=10),
        height=430,
    )
    return fig


def _surface_axes(data) -> tuple[np.ndarray, np.ndarray]:
    if data.grid_shape is None:
        raise ValueError("Surface data must define grid_shape")
    rows, cols = data.grid_shape
    x1 = data.x[:, 0].reshape(rows, cols)
    x2 = data.x[:, 1].reshape(rows, cols)
    return x1, x2


def _surface_values(values: np.ndarray, data) -> np.ndarray:
    if data.grid_shape is None:
        raise ValueError("Surface data must define grid_shape")
    return values.reshape(data.grid_shape)


def make_surface_prediction_figure(
    data,
    gd_trace: TrainingTrace | None,
    mc_traces: list[TrainingTrace],
) -> go.Figure:
    x1, x2 = _surface_axes(data)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x1,
            y=x2,
            z=_surface_values(data.clean_y, data),
            name=data.target_name,
            colorscale="Greens",
            opacity=0.42,
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=data.x[:, 0],
            y=data.x[:, 1],
            z=data.y.reshape(-1),
            mode="markers",
            name="dane treningowe",
            marker=dict(size=2.5, color=DATA_COLOR, opacity=0.55),
        )
    )
    if gd_trace is not None and gd_trace.predictions.size:
        fig.add_trace(
            go.Surface(
                x=x1,
                y=x2,
                z=_surface_values(gd_trace.predictions[-1], data),
                name="GD",
                colorscale=[[0, GD_COLOR], [1, GD_COLOR]],
                opacity=0.78,
                showscale=False,
            )
        )
    if mc_traces:
        mean_prediction = _mean_array([trace.predictions[-1] for trace in mc_traces])
        fig.add_trace(
            go.Surface(
                x=x1,
                y=x2,
                z=_surface_values(mean_prediction, data),
                name="srednia MC",
                colorscale=[[0, MC_MEAN_COLOR], [1, MC_MEAN_COLOR]],
                opacity=0.66,
                showscale=False,
            )
        )
    fig.update_layout(
        title="Dopasowanie powierzchni",
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="z",
        ),
        margin=dict(l=10, r=10, t=45, b=10),
        height=520,
    )
    return fig


def make_loss_figure(
    gd_trace: TrainingTrace | None,
    mc_traces: list[TrainingTrace],
) -> go.Figure:
    fig = go.Figure()
    if gd_trace is not None:
        fig.add_trace(
            go.Scatter(
                x=gd_trace.epochs,
                y=gd_trace.losses,
                mode="lines",
                name="GD",
                line=dict(color=GD_COLOR, width=4, dash="dot"),
            )
        )
    for idx, trace in enumerate(mc_traces):
        fig.add_trace(
            go.Scatter(
                x=trace.epochs,
                y=trace.losses,
                mode="lines",
                name=f"MC {idx + 1}",
                line=dict(color=MC_COLOR, width=1.2),
                showlegend=idx == 0,
            )
        )
    mean_mc = _final_mc_trace(mc_traces)
    if mean_mc is not None:
        fig.add_trace(
            go.Scatter(
                x=mean_mc.epochs,
                y=mean_mc.losses,
                mode="lines",
                name="srednia MC",
                line=dict(color=MC_MEAN_COLOR, width=4),
            )
        )
    fig.update_layout(
        title="Przebieg funkcji straty",
        xaxis_title="epoka",
        yaxis_title="MSE",
        yaxis_type="log",
        margin=dict(l=10, r=10, t=45, b=10),
        height=360,
    )
    return fig


def make_parameter_figure(
    gd_trace: TrainingTrace | None,
    mc_traces: list[TrainingTrace],
    parameter_index: int,
) -> go.Figure:
    fig = go.Figure()
    if gd_trace is not None and gd_trace.params.shape[1] > parameter_index:
        fig.add_trace(
            go.Scatter(
                x=gd_trace.epochs,
                y=gd_trace.params[:, parameter_index],
                mode="lines",
                name="GD",
                line=dict(color=GD_COLOR, width=4, dash="dot"),
            )
        )
    for idx, trace in enumerate(mc_traces):
        if trace.params.shape[1] <= parameter_index:
            continue
        fig.add_trace(
            go.Scatter(
                x=trace.epochs,
                y=trace.params[:, parameter_index],
                mode="lines",
                name=f"MC {idx + 1}",
                line=dict(color=MC_COLOR, width=1.2),
                showlegend=idx == 0,
            )
        )
    mean_mc = _final_mc_trace(mc_traces)
    if mean_mc is not None and mean_mc.params.shape[1] > parameter_index:
        fig.add_trace(
            go.Scatter(
                x=mean_mc.epochs,
                y=mean_mc.params[:, parameter_index],
                mode="lines",
                name="srednia MC",
                line=dict(color=MC_MEAN_COLOR, width=4),
            )
        )
    fig.update_layout(
        title=f"Trajektoria parametru x_{parameter_index}",
        xaxis_title="epoka",
        yaxis_title="wartosc parametru",
        margin=dict(l=10, r=10, t=45, b=10),
        height=360,
    )
    return fig


def render_dashboard(
    data,
    gd_trace: TrainingTrace | None,
    mc_traces: list[TrainingTrace],
    parameter_index: int,
    key_prefix: str,
) -> None:
    metrics = st.columns(3)
    gd_loss = gd_trace.losses[-1] if gd_trace is not None else np.nan
    mean_mc = _final_mc_trace(mc_traces)
    mc_loss = mean_mc.losses[-1] if mean_mc is not None else np.nan
    total_time = (gd_trace.training_time if gd_trace is not None else 0.0) + (
        mean_mc.training_time if mean_mc is not None else 0.0
    )
    metrics[0].metric("Final loss GD", f"{gd_loss:.6f}" if np.isfinite(gd_loss) else "-")
    metrics[1].metric("Mean final loss MC", f"{mc_loss:.6f}" if np.isfinite(mc_loss) else "-")
    metrics[2].metric("Training time", f"{total_time:.2f}s")

    left, right = st.columns([1.15, 1.0])
    if data.input_dim == 2:
        prediction_figure = make_surface_prediction_figure(data, gd_trace, mc_traces)
    else:
        prediction_figure = make_prediction_figure(
            data.x,
            data.y,
            data.clean_y,
            data.target_name,
            gd_trace,
            mc_traces,
        )
    left.plotly_chart(prediction_figure, width="stretch", key=f"{key_prefix}_prediction")
    right.plotly_chart(make_loss_figure(gd_trace, mc_traces), width="stretch", key=f"{key_prefix}_loss")
    st.plotly_chart(
        make_parameter_figure(gd_trace, mc_traces, parameter_index),
        width="stretch",
        key=f"{key_prefix}_parameter",
    )


def render_training_progress(
    gd_trace: TrainingTrace | None,
    mc_traces: list[TrainingTrace],
    parameter_index: int,
    key_prefix: str,
) -> None:
    metrics = st.columns(3)
    gd_loss = gd_trace.losses[-1] if gd_trace is not None else np.nan
    mean_mc = _final_mc_trace(mc_traces)
    mc_loss = mean_mc.losses[-1] if mean_mc is not None else np.nan
    total_time = (gd_trace.training_time if gd_trace is not None else 0.0) + (
        mean_mc.training_time if mean_mc is not None else 0.0
    )
    metrics[0].metric("Final loss GD", f"{gd_loss:.6f}" if np.isfinite(gd_loss) else "-")
    metrics[1].metric("Mean final loss MC", f"{mc_loss:.6f}" if np.isfinite(mc_loss) else "-")
    metrics[2].metric("Training time", f"{total_time:.2f}s")

    left, right = st.columns([1.0, 1.0])
    left.plotly_chart(make_loss_figure(gd_trace, mc_traces), width="stretch", key=f"{key_prefix}_loss")
    right.plotly_chart(
        make_parameter_figure(gd_trace, mc_traces, parameter_index),
        width="stretch",
        key=f"{key_prefix}_parameter",
    )


def render_network_screen() -> None:
    st.subheader("Kod zrodlowy sieci neuronowej")
    st.code(NETWORK_CODE, language="python", line_numbers=True)
    st.subheader("Wariant dla powierzchni 3D")
    st.code(SURFACE_NETWORK_CODE, language="python", line_numbers=True)

    st.subheader("Opis matematyczny")
    st.markdown("**[M1]-[M3] Warstwy afiniczne.** Kazda warstwa liniowa realizuje przeksztalcenie:")
    st.latex(r"z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}")
    st.markdown("Dla pierwszej warstwy mamy `h^(0)=x`, a dla warstwy wyjsciowej zamiast ukrytej aktywacji zwracana jest predykcja modelu.")

    st.markdown("**[M4]-[M6] Aktywacja nieliniowa.** Kazda warstwa ukryta stosuje funkcje `tanh`:")
    st.latex(r"h^{(\ell)} = \tanh\left(z^{(\ell)}\right)")
    st.markdown("To wlasnie ten krok odroznia model od zwyklej regresji liniowej i pozwala dopasowywac nieliniowe funkcje celu.")

    st.markdown("**[M7] Predykcja modelu.** Ostatnia warstwa zwraca wartosc funkcji aproksymowanej przez siec:")
    st.latex(r"\hat{y}=f_\theta(x)=W^{(L+1)}h^{(L)}+b^{(L+1)}")

    st.markdown("**[S1]-[S2] Przejscie do problemu 3D.** Dla powierzchni wejscie ma dwa wymiary, ale wyjscie nadal jest skalarne:")
    st.latex(r"x_i=(x_{i,1},x_{i,2})\in\mathbb{R}^{2},\qquad \hat{z}_i=f_\theta(x_{i,1},x_{i,2})\in\mathbb{R}")
    st.markdown("Zmienia sie tylko wymiar pierwszej macierzy wag. Dla pierwszej warstwy:")
    st.latex(r"W^{(1)}\in\mathbb{R}^{m\times d},\qquad d\in\{1,2\}")

    st.markdown("**Wektor parametrow.** Wszystkie macierze wag i wektory biasow sa splaszczane do jednego wektora:")
    st.latex(r"\theta = \operatorname{vec}(W^{(1)}, b^{(1)}, \ldots, W^{(L+1)}, b^{(L+1)})")
    st.markdown("Po splaszczeniu optymalizator nie rozroznia, czy parametry pochodza z modelu 1D czy 2D. Widzi tylko punkt `theta` w przestrzeni parametrow.")


def render_training_screen() -> None:
    st.subheader("Kod zrodlowy funkcji treningowych")
    st.code(TRAINING_CODE, language="python", line_numbers=True)

    st.subheader("Wspolna funkcja straty")
    st.markdown("**[GD1]-[GD2] oraz [MC2], [MC5].** Obie metody minimalizuja ten sam blad MSE:")
    st.latex(r"U(\theta)=\frac{1}{N}\sum_{i=1}^{N}\left(f_\theta(x_i)-y_i\right)^2")
    st.markdown("Ten zapis jest niezalezny od wymiaru wejscia. Dla krzywej `x_i` jest skalarem, a dla powierzchni jest wektorem `(x_{i,1}, x_{i,2})`:")
    st.latex(
        r"""
        U(\theta)=\frac{1}{N}\sum_{i=1}^{N}
        \left(f_\theta(x_{i,1},x_{i,2})-z_i\right)^2
        """
    )

    st.subheader("Przyciety spadek gradientowy")
    st.markdown("**[GD3] Gradient.** Backpropagation oblicza pochodne funkcji straty wzgledem parametrow:")
    st.latex(r"g_k=\nabla_\theta U(\theta_k)")
    st.markdown("**[GD4] Norma gradientu.** Gradient traktowany jest jako jeden splaszczony wektor:")
    st.latex(r"\lVert g_k\rVert_2=\sqrt{\sum_j\left(\frac{\partial U}{\partial \theta_j}\right)^2}")
    st.markdown("**[GD5] Aktualizacja.** Krok ma stala dlugosc kontrolowana przez `learning_rate`:")
    st.latex(r"\theta_{k+1}=\theta_k-\alpha\frac{g_k}{\lVert g_k\rVert_2}")

    st.subheader("Zerotemperaturowe Monte Carlo")
    st.markdown("**[MC1] Skala mutacji.** Skala losowej mutacji jest dobrana do kroku GD:")
    st.latex(r"\sigma=\alpha\sqrt{2\pi}")
    st.markdown("**[MC4] Propozycja mutacji.** Kazdy parametr dostaje niezalezny szum Gaussowski:")
    st.latex(r"\theta'_k=\theta_k+\varepsilon_k,\qquad \varepsilon_k\sim\mathcal{N}(0,\sigma^2 I)")
    st.markdown("**[MC6]-[MC7] Akceptacja zerotemperaturowa.** Mutacja zostaje przyjeta tylko wtedy, gdy nie pogarsza funkcji celu:")
    st.latex(
        r"""
        \theta_{k+1}=
        \begin{cases}
        \theta'_k, & U(\theta'_k)\le U(\theta_k),\\
        \theta_k, & U(\theta'_k)>U(\theta_k).
        \end{cases}
        """
    )
    st.markdown("Dla malych mutacji sredni zaakceptowany krok ma ten sam kierunek co przyciety GD:")
    st.latex(r"\mathbb{E}[\Delta\theta]=-\frac{\sigma}{\sqrt{2\pi}}\frac{\nabla U(\theta)}{\lVert\nabla U(\theta)\rVert_2}")
    st.markdown("Wniosek pozostaje taki sam dla danych 1D i 2D, poniewaz mutacja oraz gradient dzialaja w przestrzeni parametrow `theta`, a nie bezposrednio w przestrzeni wejsc.")


def render_interactive_lab() -> None:

    with st.expander("Parametry eksperymentu własnego", expanded=True):
        experiment_type = st.selectbox(
            "experiment_type",
            options=["curve", "surface"],
            format_func=lambda key: "1D curve" if key == "curve" else "3D surface",
        )
        target_options = CURVE_FUNCTIONS if experiment_type == "curve" else SURFACE_FUNCTIONS
        target_function = st.selectbox(
            "target_function",
            options=list(target_options.keys()),
            format_func=lambda key: TARGET_FUNCTIONS[key],
        )
        depth = st.slider("depth", 1, 4, 1)
        width = st.slider("width", 4, 128, 40, step=4)
        surface_grid_size = 35
        if experiment_type == "surface":
            surface_grid_size = st.slider("surface_grid_size", 12, 60, 35, step=1)
        epochs = st.slider("epochs", 100, 100000, 2000, step=100)
        learning_rate = st.number_input(
            "learning_rate",
            min_value=1e-6,
            max_value=1e-1,
            value=5e-3,
            step=1e-3,
            format="%.6f",
        )
        mc_trajectories = st.slider("mc_trajectories", 1, 30, 10)
        noise_std = st.slider("noise_std", 0.0, 0.5, 0.0, step=0.01)
        save_every = st.slider("save_every", 1, 200, 20)
        live_update_every = st.slider("live_update_every", 20, 1000, 200, step=20)
        live_delay_ms = st.slider("live_delay_ms", 0, 500, 80, step=20)
        surface_live_preview = False
        if experiment_type == "surface":
            surface_live_preview = st.checkbox("surface_live_preview", value=False)
        seed = st.number_input("seed", value=42, step=1)
        parameter_index = st.number_input("parameter index", min_value=0, value=0, step=1)
        start = st.button("Start", type="primary")
        reset = st.button("Reset")

    if reset:
        for key in ["gd_trace", "mc_traces"]:
            st.session_state.pop(key, None)
        st.rerun()

    config = ExperimentConfig(
        noise_std=float(noise_std),
        target_function=str(target_function),
        seed=int(seed),
        depth=int(depth),
        width=int(width),
        input_size=2 if experiment_type == "surface" else 1,
        output_size=1,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        mc_trajectories=int(mc_trajectories),
        save_every=int(save_every),
        device="cpu",
    )
    data = generate_data(
        n_points=config.n_points,
        x_min=config.x_min,
        x_max=config.x_max,
        noise_std=config.noise_std,
        seed=config.seed,
        target_function=config.target_function,
        experiment_type=str(experiment_type),
        grid_size=int(surface_grid_size),
    )

    experiment_tab, network_tab, training_tab = st.tabs(
        ["Eksperyment", "Siec neuronowa", "Funkcje treningowe"]
    )

    with network_tab:
        render_network_screen()

    with training_tab:
        render_training_screen()

    with experiment_tab:
        dashboard = st.empty()
        progress = st.progress(0.0)
        status = st.empty()

        if start:
            gd_trace: TrainingTrace | None = None
            mc_traces: list[TrainingTrace] = []
            gd_factory = make_seeded_model_factory(config)
            mc_factory = make_seeded_model_factory(config)
            total_units = 1 + config.mc_trajectories
            render_count = 0

            def update_dashboard(stage: str, epoch: int, trace: TrainingTrace) -> None:
                nonlocal gd_trace, mc_traces, render_count
                if stage == "gd":
                    gd_trace = trace
                    unit_offset = 0
                else:
                    idx = int(stage.split("_", 1)[1])
                    while len(mc_traces) <= idx:
                        mc_traces.append(trace)
                    mc_traces[idx] = trace
                    unit_offset = 1 + idx
                current_unit_fraction = min(1.0, epoch / max(1, config.epochs))
                progress.progress(min(1.0, (unit_offset + current_unit_fraction) / total_units))
                status.write(f"{stage}: epoch {epoch}/{config.epochs}")
                should_render = (
                    epoch == 0
                    or epoch == config.epochs
                    or epoch % int(live_update_every) == 0
                )
                if not should_render:
                    return
                render_count += 1
                with dashboard.container():
                    if data.input_dim == 2 and not surface_live_preview:
                        render_training_progress(
                            gd_trace,
                            mc_traces,
                            int(parameter_index),
                            f"live_{render_count}",
                        )
                    else:
                        render_dashboard(
                            data,
                            gd_trace,
                            mc_traces,
                            int(parameter_index),
                            f"live_{render_count}",
                        )
                if live_delay_ms > 0:
                    time.sleep(float(live_delay_ms) / 1000.0)

            gd_trace = train_clipped_gd(gd_factory(), data, config, on_update=update_dashboard)
            mc_traces = []
            mc_traces = train_zero_temp_mc(mc_factory, data, config, on_update=update_dashboard)

            st.session_state["gd_trace"] = gd_trace
            st.session_state["mc_traces"] = mc_traces
            status.write("Gotowe")
            progress.progress(1.0)
            with dashboard.container():
                render_dashboard(data, gd_trace, mc_traces, int(parameter_index), "final")
            time.sleep(0.1)

        gd_trace = st.session_state.get("gd_trace")
        mc_traces = st.session_state.get("mc_traces", [])
        if not start:
            with dashboard.container():
                render_dashboard(data, gd_trace, mc_traces, int(parameter_index), "static")

        if gd_trace is not None:
            table = pd.DataFrame(
                {
                    "epoch": gd_trace.epochs,
                    "gd_loss": gd_trace.losses,
                }
            )
            mean_mc = _final_mc_trace(mc_traces)
            if mean_mc is not None and mean_mc.losses.shape == gd_trace.losses.shape:
                table["mean_mc_loss"] = mean_mc.losses
            st.dataframe(table.tail(10), width="stretch")




def main() -> None:
    st.title("GD vs neuroewolucja")
    st.caption(
        "Interaktywne laboratorium do obserwacji działania obu metod. "
        "Oficjalny protokół badawczy uruchamia moduł mixed.neuro_grad.experiment."
    )
    render_interactive_lab()


if __name__ == "__main__":
    main()
