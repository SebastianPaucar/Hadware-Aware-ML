"""
complete_axo_superpose_dynamic.py

Genera plots superpuestos para múltiples HDF5 AXO outputs detectando
automáticamente las métricas/signals/score-types en común entre los archivos.
"""

import h5py
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from io import BytesIO
import base64
import os
import argparse
from sklearn.metrics import roc_curve, auc

plt.style.use(hep.style.CMS)
DEFAULT_HTML_PATH = "axo_superposed_plots.html"

def open_h5_file(h5_path):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")
    return h5py.File(h5_path, "r")

def basename(path):
    return os.path.splitext(os.path.basename(path))[0]

# ----------------------------
# Helpers to detect commons
# ----------------------------
def intersect_sets(list_of_sets):
    if not list_of_sets:
        return set()
    return set.intersection(*list_of_sets)

# =====================================================
# HISTORY (common keys intersection)
# =====================================================
def create_history_plots(h5_files, html_path):
    print("\n--- History: detect common metrics ---")
    metrics_sets = []
    histories = {}  # file -> dict(metric -> array)
    for p in h5_files:
        with open_h5_file(p) as f:
            if "history" not in f:
                histories[p] = {}
                metrics_sets.append(set())
                continue
            metrics = set(f["history"].keys())
            metrics_sets.append(metrics)
            histories[p] = {k: f["history"][k][:] for k in metrics}

    common_metrics = sorted(intersect_sets(metrics_sets))
    print("Common history metrics:", common_metrics)
    if not common_metrics:
        print("No common history metrics found; skipping history plots.")
        return

    html_parts = ["<html><head><title>History Superposed</title></head><body>",
                  "<h1>Training History Superposed</h1>"]

    for metric in common_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        any_data = False
        for p in h5_files:
            data = histories.get(p, {}).get(metric)
            if data is None:
                continue
            any_data = True
            tag = os.path.basename(p).replace("complete_", "").replace(".h5", "")
            ax.plot(np.arange(len(data)), data, marker='o', linewidth=2, label=f"{tag}")
        if not any_data:
            continue
        hep.cms.label(rlabel="")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=12)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        html_parts.append(f"<h2>{metric}</h2><img src='data:image/png;base64,{img_b64}' width='900'><br>")

    html_parts.append("</body></html>")
    out_path = html_path.replace(".html", "_history.html")
    with open(out_path, "w") as fo:
        fo.write("\n".join(html_parts))
    print(f"Saved history HTML -> {out_path}")

# =====================================================
# ROC from results/latent_scores
# =====================================================

def create_latent_roc_plots(h5_files, html_path):
    """
    Genera curvas ROC usando results/latent_scores:
    - Background vs cada señal
    - Curva calculada con sklearn.metrics.roc_curve
    - AUC mostrado en la leyenda
    """
    print("\n--- Latent ROC detection ---")

    latent_sets = []
    latent_data = {}

    for p in h5_files:
        with open_h5_file(p) as f:
            group = f.get("results/latent_scores")
            if group is None or "Background" not in group:
                latent_sets.append(set())
                latent_data[p] = None
                continue

            signals = [k for k in group.keys() if k != "Background"]
            latent_sets.append(set(signals))

            # Guardar los datos en memoria (no el objeto del archivo)
            data = {
                "background": group["Background"][:],
                "signals": signals,
                "signal_arrays": {sig: group[sig][:] for sig in signals}
            }
            latent_data[p] = data  # se guarda el contenido, no el handle

    common_signals = sorted(set.intersection(*latent_sets)) if latent_sets else []
    print("Common latent signals:", common_signals)
    if not common_signals:
        print("No common latent signals; skipping ROC.")
        return

    html_parts = [
        "<html><head><title>ROC Latent Scores</title></head><body>",
        "<h1>ROC Curves (from results/latent_scores)</h1>"
    ]

    for signal in common_signals:
        fig, ax = plt.subplots(figsize=(8, 7))
        plotted = False

        for p in h5_files:
            ld = latent_data.get(p)
            if not ld or signal not in ld["signals"]:
                continue

            bg = ld["background"]
            sig_scores = ld["signal_arrays"][signal]  # CORREGIDO

            y_true = np.concatenate([np.ones_like(sig_scores), np.zeros_like(bg)])
            y_score = np.concatenate([sig_scores, bg])

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)

            tag = os.path.basename(p).replace("complete_", "").replace(".h5", "")
            ax.plot(fpr, tpr, linewidth=2, marker='o',
                    label=f"{tag} (AUC={roc_auc:.3f})")
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        hep.cms.label(rlabel="")
        ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.6)
        leg = ax.legend(loc="best", title=signal, fontsize=12)
        leg.get_title().set_fontsize(16)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        html_parts.append(f"<h2>{signal}</h2><img src='data:image/png;base64,{img_b64}' width='800'><br>")

    html_parts.append("</body></html>")
    out_path = html_path.replace(".html", "_roc_latent.html")
    with open(out_path, "w") as fo:
        fo.write("\n".join(html_parts))
    print(f"Saved latent ROC HTML -> {out_path}")

# =====================================================
# Score vs Trigger-rate plots (detect common score types & common trigger rates & signals)
# =====================================================
def create_score_rate_plots(h5_files, html_path):
    print("\n--- Score vs Trigger-Rate plots ---")
    # gather per-file score types, trigger strings and signals
    per_file = {}  # p -> {trigger_strs_list, score_types_set, signals_list, data_map}
    sets_triggers, sets_score_types, sets_signals = [], [], []

    for p in h5_files:
        with open_h5_file(p) as f:
            axo_root = f.get("results/axo_scores")
            if axo_root is None:
                per_file[p] = None
                sets_triggers.append(set())
                sets_score_types.append(set())
                sets_signals.append(set())
                continue
            triggers = list(axo_root.keys())
            # detect numeric order if possible
            try:
                triggers_sorted = [k for _, k in sorted([(float(k), k) for k in triggers], key=lambda x: x[0])]
            except Exception:
                triggers_sorted = sorted(triggers)
            # detect score types in first trigger group (exclude "Signal Name")
            first_grp = axo_root[triggers_sorted[0]]
            score_types = [k for k in first_grp.keys() if "Signal Name" not in k]
            sets_triggers.append(set(triggers_sorted))
            sets_score_types.append(set(score_types))
            if "Signal Name" in first_grp:
                sigs = [s.decode() if isinstance(s, bytes) else str(s) for s in first_grp["Signal Name"][:]]
            else:
                # infer length from any score dataset
                sample = first_grp[score_types[0]][:]
                sigs = [f"Signal {i+1}" for i in range(len(sample))]
            sets_signals.append(set(sigs))

            # store arrays per trigger and per score_type
            data_map = {}  # score_type -> array shape (n_triggers, n_signals)
            for st in score_types:
                arrs = []
                for trig in triggers_sorted:
                    grp = axo_root[trig]
                    if st in grp:
                        arrs.append(grp[st][:])
                    else:
                        arrs.append(np.full(len(sigs), np.nan))
                data_map[st] = np.vstack(arrs)  # (n_triggers, n_signals)
            per_file[p] = {"triggers": triggers_sorted, "score_types": score_types, "signals": sigs, "data": data_map}

    common_triggers = sorted(intersect_sets(sets_triggers), key=lambda x: float(x) if x.replace('.','',1).lstrip('-').isdigit() else x)
    common_score_types = sorted(intersect_sets(sets_score_types))
    common_signals = sorted(intersect_sets(sets_signals))
    print("Common triggers:", common_triggers)
    print("Common score types:", common_score_types)
    print("Common signals:", common_signals)

    if not (common_triggers and common_score_types and common_signals):
        print("Missing common triggers/score_types/signals; skipping score-rate plots.")
        return

    html_parts = ["<html><head><title>Score vs Trigger Rate</title></head><body>",
                  "<h1>Score vs Trigger Rate (superposed)</h1>"]

    # convert triggers to numeric array
    try:
        numeric_rates = np.array([float(t) for t in common_triggers])
    except Exception:
        numeric_rates = np.arange(len(common_triggers))

    # For each score_type and each common signal, plot lines from each file
    for st in common_score_types:
        html_parts.append(f"<h2>{st}</h2>")
        for sig in common_signals:
            fig, ax = plt.subplots(figsize=(10, 6))
            plotted = False
            for p in h5_files:
                pf = per_file.get(p)
                if not pf:
                    continue
                if st not in pf["score_types"]:
                    continue
                if sig not in pf["signals"]:
                    continue
                # for this file, pick indices of common_triggers in its trigger list
                idxs = [pf["triggers"].index(t) for t in common_triggers]
                sig_idx = pf["signals"].index(sig)
                vals = pf["data"][st][idxs, sig_idx]
                tag = os.path.basename(p).replace("complete_", "").replace(".h5", "")
                ax.plot(numeric_rates, vals, marker='o', linewidth=2, label=f"{tag}")
                plotted = True
            if not plotted:
                plt.close(fig)
                continue
            hep.cms.label(rlabel="")
            ax.set_xlabel("Trigger rate (kHz)")
            ax.set_ylabel(st)
            ax.grid(True, linestyle="--", alpha=0.6)
            leg = ax.legend(loc="best", title=sig, fontsize=12)
            leg.get_title().set_fontsize(16)
            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            html_parts.append(f"<h3>{sig}</h3><img src='data:image/png;base64,{img_b64}' width='900'><br>")

    html_parts.append("</body></html>")
    out_path = html_path.replace(".html", "_score_rate.html")
    with open(out_path, "w") as fo:
        fo.write("\n".join(html_parts))
    print(f"Saved score vs trigger-rate HTML -> {out_path}")

# =====================================================
# Overlay histograms (common signals)
# =====================================================

from os.path import basename
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import mplhep as hep

def create_overlay_histogram_plots(h5_files, html_path):
    print("\n--- Overlay histograms detection ---")
    hist_sets = []
    hist_data = {}

    for p in h5_files:
        with open_h5_file(p) as f:
            sh = f.get("results/score_hist")
            if sh is None or "signal" not in sh:
                hist_sets.append(set())
                hist_data[p] = None
                continue

            sig_grp = sh["signal"]
            sigs = list(sig_grp.keys())
            hist_sets.append(set(sigs))

            #  Copiar datos de señal a memoria
            signal_arrays = {}
            for sig in sigs:
                grp = sig_grp[sig]
                if "bin" in grp and "count" in grp:
                    signal_arrays[sig] = {
                        "bin": grp["bin"][:],
                        "count": grp["count"][:]
                    }

            #  Copiar background si existe
            bg_bins, bg_counts = None, None
            if "background" in sh:
                bg_grp = sh["background"]
                if "bin" in bg_grp and "count" in bg_grp:
                    bg_bins = bg_grp["bin"][:]
                    bg_counts = bg_grp["count"][:]

            hist_data[p] = {
                "signals": sigs,
                "signal_arrays": signal_arrays,
                "bg_bins": bg_bins,
                "bg_counts": bg_counts
            }

    # Detectar señales comunes
    common_signals = sorted(set.intersection(*hist_sets)) if hist_sets else []
    print("Common histogram signals:", common_signals)
    if not common_signals:
        print("No common histogram signals; skipping histograms.")
        return

    html_parts = [
        "<html><head><title>Overlay Histograms</title></head><body>",
        "<h1>Overlay Histograms (superposed)</h1>"
    ]

    for sig in common_signals:
        fig, ax = plt.subplots(figsize=(10, 6))
        any_plot = False

        for p in h5_files:
            hd = hist_data.get(p)
            if not hd or sig not in hd["signal_arrays"]:
                continue

            arr = hd["signal_arrays"][sig]
            bins = arr["bin"]
            counts = arr["count"]

            if len(bins) != len(counts) + 1:
                bins = bins[:len(counts) + 1]

            ax.step(bins[:-1], counts, where="post", label=basename(p))
            any_plot = True

        # Agregar un background (si alguno existe)
        for hd in hist_data.values():
            if hd and hd["bg_bins"] is not None:
                ax.step(
                    hd["bg_bins"][:-1],
                    hd["bg_counts"],
                    where="post",
                    linestyle="--",
                    color="black",
                    label="Background"
                )
                break

        if not any_plot:
            plt.close(fig)
            continue

        hep.cms.label(rlabel="")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9)
        plt.tight_layout()

        # Guardar en HTML (inline base64)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        html_parts.append(f"<h2>{sig}</h2><img src='data:image/png;base64,{img_b64}' width='900'><br>")

    html_parts.append("</body></html>")
    out_path = html_path.replace(".html", "_overlay_histograms.html")
    with open(out_path, "w") as fo:
        fo.write("\n".join(html_parts))
    print(f"Saved overlay histograms HTML -> {out_path}")


# =====================================================
# Overlay histograms for latent_scores (common signals)
# =====================================================

plt.style.use(hep.style.CMS)

# =====================================================
# Overlay histograms for latent_scores (common signals)
# =====================================================

def create_overlay_histogram_plots_latent_scoresBAK(h5_files, html_path):
    print("\n--- Overlay latent_scores histograms detection ---")

    latent_sets = []
    latent_data = {}

    # Abrimos y cargamos completamente los datos
    for p in h5_files:
        with h5py.File(p, "r") as f:
            grp = f.get("results/latent_scores")
            if grp is None:
                latent_sets.append(set())
                latent_data[p] = None
                continue

            all_keys = list(grp.keys())
            signals = [k for k in all_keys if k.lower() != "background"]
            latent_sets.append(set(signals))

            # Leemos los arrays completos mientras el archivo está abierto
            data_dict = {sig: grp[sig][:] for sig in signals if sig in grp}
            bg_data = grp["Background"][:] if "Background" in grp else None
            latent_data[p] = {"signals": data_dict, "background": bg_data}

    common_signals = sorted(intersect_sets(latent_sets))
    print("Common latent signals:", common_signals)
    if not common_signals:
        print("No common signals found; skipping latent score overlays.")
        return

    # =====================================================
    # Construir bins comunes
    # =====================================================
    print("Computing global bins for all files/signals...")
    all_scores = []
    for data_dict in latent_data.values():
        if not data_dict:
            continue
        for sig in common_signals:
            if sig in data_dict["signals"]:
                all_scores.append(data_dict["signals"][sig])
        if data_dict["background"] is not None:
            all_scores.append(data_dict["background"])

    if not all_scores:
        print("No latent scores found; aborting plot generation.")
        return

    combined = np.concatenate(all_scores)
    vmin, vmax = np.percentile(combined, [0.1, 99.9])
    bins = np.linspace(vmin, vmax, 50)

    html_parts = [
        "<html><head><title>Overlay Latent Histograms</title></head><body>",
        "<h1>Overlay Histograms of Latent Scores (superposed, common bins)</h1>",
    ]

    for sig in common_signals:
        fig, ax = plt.subplots()  # tamaño automático
        any_plot = False

        # Señales
        for p, data_dict in latent_data.items():
            if not data_dict or sig not in data_dict["signals"]:
                continue
            data = data_dict["signals"][sig]
            counts, _ = np.histogram(data, bins=bins)
            label_name = os.path.basename(p).split("_", 1)[-1]
            ax.step(bins[:-1], counts, where="post", label=label_name)
            any_plot = True

        # Backgrounds
        for p, data_dict in latent_data.items():
            if not data_dict or data_dict["background"] is None:
                continue
            bg = data_dict["background"]
            counts, _ = np.histogram(bg, bins=bins)
            label_bg = os.path.basename(p).split("_", 1)[-1] + " (Background)"
            ax.step(bins[:-1], counts, where="post", linestyle="--", color="black", label=label_bg)
            # Etiqueta encima del histograma
            max_count = counts.max()
            mid_bin = (bins[:-1] + bins[1:]) / 2
            ax.text(mid_bin[np.argmax(counts)], max_count*1.05, label_bg,
                    ha="center", va="bottom", fontsize=8, rotation=45)
            any_plot = True

        if not any_plot:
            plt.close(fig)
            continue

        hep.cms.label(rlabel="")

        ax.set_xlabel("Latent Score")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=9, loc="best")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        html_parts.append(f"<h2>{sig}</h2><img src='data:image/png;base64,{img_b64}' width='900'><br>")

    html_parts.append("</body></html>")
    out_path = html_path.replace(".html", "_overlay_latent_histograms.html")
    with open(out_path, "w") as fo:
        fo.write("\n".join(html_parts))
    print(f"Saved overlay latent histogram HTML -> {out_path}")


def create_overlay_histogram_plots_latent_scores(h5_files, html_path):
    print("\n--- Overlay latent_scores histograms detection ---")

    latent_sets = []
    latent_data = {}
    background_data = {}

    # =====================================================
    # Cargar señales y backgrounds
    # =====================================================
    for p in h5_files:
        with h5py.File(p, "r") as f:
            grp = f.get("results/latent_scores")
            if grp is None:
                latent_sets.append(set())
                latent_data[p] = None
                background_data[p] = None
                continue

            all_keys = list(grp.keys())
            signals = [k for k in all_keys if k.lower() != "background"]
            latent_sets.append(set(signals))

            data_dict = {sig: grp[sig][:] for sig in signals if sig in grp}
            latent_data[p] = data_dict

            background_data[p] = grp["Background"][:] if "Background" in grp else None

    # =====================================================
    # Detectar señales comunes
    # =====================================================
    common_signals = sorted(set.intersection(*latent_sets)) if latent_sets else []
    print("Common latent signals:", common_signals)
    if not common_signals:
        print("No common signals found; skipping latent score overlays.")
        return

    # =====================================================
    # Construir bins comunes
    # =====================================================
    print("Computing global bins for all files/signals...")
    all_scores = []
    for data_dict in latent_data.values():
        if not data_dict:
            continue
        for sig in common_signals:
            if sig in data_dict:
                all_scores.append(data_dict[sig])

    if not all_scores:
        print("No latent scores found; aborting plot generation.")
        return

    combined = np.concatenate(all_scores)
    vmin, vmax = np.percentile(combined, [0.1, 99.9])
    bins = np.linspace(vmin, vmax, 50)

    # =====================================================
    # Generar HTML con los histogramas superpuestos
    # =====================================================
    html_parts = [
        "<html><head><title>Overlay Latent Histograms</title></head><body>",
        "<h1>Overlay Histograms of Latent Scores (superposed, common bins)</h1>",
    ]

    for sig in common_signals:
        fig, ax = plt.subplots(figsize=(10, 6))
        any_plot = False
        hep.cms.label(ax=ax, rlabel="")

        for p in h5_files:
            data_dict = latent_data.get(p)
            if not data_dict or sig not in data_dict:
                continue

            tag = os.path.basename(p).replace("complete_", "").replace(".h5", "")

            # --- Señal ---
            data = data_dict[sig]
            counts, _ = np.histogram(data, bins=bins)
            ax.step(bins[:-1], counts, where="post", label=f"{tag}", linewidth=1.6)
            any_plot = True

            # --- Background (relleno transparente) ---
            bkg = background_data.get(p)
            if bkg is not None:
                ax.hist(
                    bkg,
                    bins=bins,
                    histtype="stepfilled",
                    alpha=0.3,
                    label=f"{tag} (Background)",
                )

        if not any_plot:
            plt.close(fig)
            continue

        # Estética
        ax.set_xlabel("Latent Score")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        leg = ax.legend(loc="best", title=sig, fontsize=12)
        leg.get_title().set_fontsize(16)
        plt.tight_layout()

        # Guardar imagen en HTML embebido
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        html_parts.append(f"<img src='data:image/png;base64,{img_b64}' width='900'><br><br>")

    html_parts.append("</body></html>")

    out_path = html_path.replace(".html", "_overlay_latent_histograms.html")
    with open(out_path, "w") as fo:
        fo.write("\n".join(html_parts))
    print(f"Saved overlay latent histogram HTML -> {out_path}")


# =====================================================
# Main CLI
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="AXO Multi-HDF5 Analyzer (dynamic common-metrics, superposed)")
    parser.add_argument("--h5_files", nargs='+', required=True, help="Lista de HDF5 a comparar")
    parser.add_argument("--history_plots", action="store_true", help="Generar history superpuesto")
    parser.add_argument("--roc_plots", action="store_true", help="Generar ROC (FP/FN) superpuesto")
    parser.add_argument("--score_rate_plots", action="store_true", help="Generar plots score vs trigger rate superpuestos")
    parser.add_argument("--overlay_histogram_plots", action="store_true", help="Generar overlay histograms superpuestos")
    parser.add_argument("--latent_roc_plots", action="store_true", help="Generar ROC desde latent_scores superpuestos")
    parser.add_argument("--html_out", default=DEFAULT_HTML_PATH, help="Archivo HTML base de salida")
    parser.add_argument("--overlay_histogram_plots_latent", action="store_true", help="Create overlay histograms from latent_scores (common bins)")

    args = parser.parse_args()

    h5_files = args.h5_files

    if args.history_plots:
        create_history_plots(h5_files, args.html_out)
    if args.roc_plots:
        create_roc_plots(h5_files, args.html_out)
    if args.score_rate_plots:
        create_score_rate_plots(h5_files, args.html_out)
    if args.overlay_histogram_plots:
        create_overlay_histogram_plots(h5_files, args.html_out)
    if args.latent_roc_plots:
        create_latent_roc_plots(h5_files, args.html_out)
    if args.overlay_histogram_plots_latent:
        create_overlay_histogram_plots_latent_scores(h5_files, args.html_out)



if __name__ == "__main__":
    main()

