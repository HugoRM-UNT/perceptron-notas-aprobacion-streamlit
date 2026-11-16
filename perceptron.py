"""Perceptrón para predecir si un estudiante aprueba una experiencia curricular."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

NOTA_MAX = 20
BINARIO = "Binario (unidad aprobada = 1, desaprobada = 0)"
NORMALIZADO = "Notas normalizadas (nota / 20)"
INIT_FIJA = "Fija (valores de clase)"
INIT_ALEATORIA = "Aleatoria U(-1, 1)"
W_CLASE = [0.5, 0.4, 0.3]
WB_CLASE = 1.5


def unidades_aprobadas(notas, nota_min: float = 14) -> int:
    return int(sum(1 for n in notas if n >= nota_min))


def regla_objetivo(notas, nota_min: float = 14) -> int:
    """Aprueba si tiene al menos dos unidades aprobadas."""
    return int(unidades_aprobadas(notas, nota_min) >= 2)


def escalon(net: float) -> int:
    return 1 if net >= 0 else 0


def preparar_entrada(notas, modo: str, nota_min: float) -> np.ndarray:
    notas = np.asarray(notas, dtype=float)
    if modo == BINARIO:
        return (notas >= nota_min).astype(float)
    return notas / NOTA_MAX


def tabla_verdad(nota_min: float = 14) -> pd.DataFrame:
    filas = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                filas.append({"x1 (U1)": a, "x2 (U2)": b, "x3 (U3)": c, "yd": int(a + b + c >= 2)})
    df = pd.DataFrame(filas)
    df.index = [f"Patrón {i:02d}" for i in range(1, 9)]
    return df


def construir_datos(n: int, seed: int, nota_min: float, p_val: float, p_test: float) -> dict:
    """Genera estudiantes sintéticos y separa los datos en train, val y test."""
    rng = np.random.default_rng(seed)
    notas = rng.integers(0, NOTA_MAX + 1, size=(n, 3))
    yd = np.array([regla_objetivo(f, nota_min) for f in notas])
    orden = rng.permutation(n)
    n_test = int(round(n * p_test))
    n_val = int(round(n * p_val))
    return {
        "notas": notas, "yd": yd, "nota_min": nota_min,
        "idx_test": orden[:n_test],
        "idx_val": orden[n_test:n_test + n_val],
        "idx_train": orden[n_test + n_val:],
    }


def subconjunto(datos: dict, parte: str, modo: str):
    idx = datos[f"idx_{parte}"]
    notas = datos["notas"][idx]
    return preparar_entrada(notas, modo, datos["nota_min"]), datos["yd"][idx], notas


def pesos_iniciales(init: str, seed: int, w_fijos=None, wb_fijo=None):
    if init == INIT_ALEATORIA:
        v = np.random.default_rng(10_000 + seed).uniform(-1, 1, 4)
        return v[:3].tolist(), float(v[3])
    return list(w_fijos if w_fijos is not None else W_CLASE), float(WB_CLASE if wb_fijo is None else wb_fijo)


def errores_con_pesos(X, yd, w, wb) -> int:
    pred = (X @ np.asarray(w) + wb >= 0).astype(int)
    return int(np.sum(pred != yd))


def entrenar(X, yd, w_init, wb_init, fa, max_epocas=100, mezclar=False, seed=0, traza_epocas=0):
    """Entrena el perceptrón y conserva los pesos de la mejor época."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    w = [float(v) for v in w_init]
    wb = float(wb_init)
    Xl, yl = X.tolist(), [int(v) for v in yd]
    n = len(Xl)

    traza, historial = [], []
    mejor = {"errores": n + 1, "w": list(w), "wb": wb, "epoca": 0}
    convergio = False

    for epoca in range(1, max_epocas + 1):
        orden = rng.permutation(n).tolist() if mezclar else range(n)
        errores_epoca = 0
        for i in orden:
            x, d = Xl[i], yl[i]
            net = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + wb
            y = 1 if net >= 0 else 0
            error = d - y
            if error:
                for j in range(3):
                    w[j] += fa * error * x[j]
                wb += fa * error
                errores_epoca += 1
            if epoca <= traza_epocas:
                traza.append({
                    "Época": epoca, "Patrón": i + 1,
                    "x1": round(x[0], 3), "x2": round(x[1], 3), "x3": round(x[2], 3),
                    "yd": d, "NET": round(net, 4), "y": y, "error": error,
                    "w1": round(w[0], 4), "w2": round(w[1], 4), "w3": round(w[2], 4),
                    "wbias": round(wb, 4),
                })
        err_eval = errores_con_pesos(X, yd, w, wb)
        historial.append({
            "Época": epoca, "Errores en la época": errores_epoca,
            "Errores con pesos finales": err_eval,
            "w1": w[0], "w2": w[1], "w3": w[2], "wbias": wb,
        })
        if err_eval < mejor["errores"]:
            mejor = {"errores": err_eval, "w": list(w), "wb": wb, "epoca": epoca}
        if errores_epoca == 0:
            convergio = True
            break

    return {
        "w": np.array(mejor["w"]), "wb": mejor["wb"], "epoca_mejor": mejor["epoca"],
        "errores_finales": mejor["errores"], "convergio": convergio,
        "epocas": len(historial), "n_patrones": n,
        "tiempo_s": time.perf_counter() - t0,
        "traza": pd.DataFrame(traza), "historial": pd.DataFrame(historial),
    }


def predecir(modelo: dict, notas):
    x = preparar_entrada(notas, modelo["modo"], modelo["nota_min"])
    net = float(np.dot(modelo["w"], x) + modelo["wb"])
    return x, net, escalon(net)


def metricas(real, pred) -> dict:
    real, pred = np.asarray(real), np.asarray(pred)
    tp = int(np.sum((real == 1) & (pred == 1)))
    tn = int(np.sum((real == 0) & (pred == 0)))
    fp = int(np.sum((real == 0) & (pred == 1)))
    fn = int(np.sum((real == 1) & (pred == 0)))
    acc = (tp + tn) / max(len(real), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"exactitud": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluar(modelo: dict, datos: dict, parte: str):
    X, yd, notas = subconjunto(datos, parte, modelo["modo"])
    pred = (X @ modelo["w"] + modelo["wb"] >= 0).astype(int)
    net = X @ modelo["w"] + modelo["wb"]
    m = metricas(yd, pred)
    df = pd.DataFrame({
        "Unidad 1": notas[:, 0], "Unidad 2": notas[:, 1], "Unidad 3": notas[:, 2],
        "Aprobadas": [unidades_aprobadas(f, datos["nota_min"]) for f in notas],
        "NET": np.round(net, 3), "Real": yd, "Predicho": pred, "Correcto": yd == pred,
    })
    matriz = pd.DataFrame(
        [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
        index=["Real: Desaprueba", "Real: Aprueba"],
        columns=["Pred: Desaprueba", "Pred: Aprueba"],
    )
    return df, m, matriz


def baseline_mayoritaria(datos: dict, parte: str) -> float:
    """Exactitud de predecir siempre la clase más frecuente del conjunto de entrenamiento."""
    clase = int(np.mean(datos["yd"][datos["idx_train"]]) >= 0.5)
    return float(np.mean(datos["yd"][datos[f"idx_{parte}"]] == clase))


def ejecutar_experimentos(datos, modos, fas, inits, n_semillas=10, max_epocas=50):
    filas = []
    for modo in modos:
        Xtr, ytr, _ = subconjunto(datos, "train", modo)
        Xva, yva, _ = subconjunto(datos, "val", modo)
        for fa in fas:
            for init in inits:
                for s in range(n_semillas):
                    w0, b0 = pesos_iniciales(init, s)
                    r = entrenar(Xtr, ytr, w0, b0, fa, max_epocas, mezclar=True, seed=s)
                    filas.append({
                        "Modo": "Binario" if modo == BINARIO else "Normalizado",
                        "fa": fa, "Init": "Fija" if init == INIT_FIJA else "Aleatoria",
                        "Semilla": s, "Convergió": r["convergio"], "Épocas": r["epocas"],
                        "Acc. train": 1 - r["errores_finales"] / r["n_patrones"],
                        "Acc. val": float(np.mean(((Xva @ r["w"] + r["wb"]) >= 0).astype(int) == yva)),
                        "Tiempo (ms)": r["tiempo_s"] * 1000,
                    })
    runs = pd.DataFrame(filas)
    resumen = (runs.groupby(["Modo", "fa", "Init"], sort=False)
               .agg(**{"Convergió (%)": ("Convergió", lambda s: 100 * s.mean()),
                       "Épocas (media)": ("Épocas", "mean"),
                       "Acc. val (media)": ("Acc. val", "mean"),
                       "Acc. val (desv.)": ("Acc. val", "std"),
                       "Acc. train (media)": ("Acc. train", "mean"),
                       "Tiempo ms (media)": ("Tiempo (ms)", "mean")})
               .reset_index())
    resumen["Acc. val (desv.)"] = resumen["Acc. val (desv.)"].fillna(0.0)
    resumen = resumen.sort_values(["Acc. val (media)", "Épocas (media)"], ascending=[False, True]).reset_index(drop=True)
    return runs, resumen
