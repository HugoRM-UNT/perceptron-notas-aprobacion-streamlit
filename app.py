import platform
import sys
from importlib import metadata

import numpy as np
import pandas as pd
import streamlit as st

import perceptron as pc

# ----------------------------------------------------------------------
# Datos del trabajo
# ----------------------------------------------------------------------
CURSO = "Sistemas Inteligentes y Machine Learning"
AUTOR = "Hugo Ernesto Rebaza Mendoza"
MODALIDAD = "Presentación individual"
DOCENTE = "Ms. Torres Villanueva, Marcelino"  

st.set_page_config(page_title="Perceptrón – Experiencia curricular", page_icon="🧠", layout="wide")
st.title("🧠 Perceptrón simple: ¿aprueba la experiencia curricular?")
st.caption(f"{CURSO} · {MODALIDAD}: {AUTOR} · Docente: {DOCENTE}")

# ----------------------------------------------------------------------
# Barra lateral: configuración de los datos (aplica a todo el trabajo)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Datos del experimento")
    nota_min = st.number_input("Nota aprobatoria por unidad", 1.0, 20.0, 14.0, 0.5,
                               help="Una unidad se considera aprobada si su promedio es ≥ este valor.")
    n_est = st.number_input("Estudiantes sintéticos", 50, 5000, 400, 50)
    seed_datos = st.number_input("Semilla de los datos", 0, 9999, 42, 1)
    p_val = st.slider("% validación", 10, 30, 20, 5) / 100
    p_test = st.slider("% prueba (test)", 10, 30, 20, 5) / 100
    st.caption("El conjunto de **test** solo se usa en la pestaña *5 · Evaluación final*, "
               "nunca para ajustar hiperparámetros ni elegir la configuración.")
    st.divider()
    st.markdown(
        "**Resumen**\n\n- Entradas: 3 promedios (0–20)\n- Salida: 1 aprueba / 0 desaprueba\n"
        "- Regla: ≥ 2 unidades aprobadas\n- Modelo: 1 neurona (perceptrón)\n- Aprendizaje: supervisado, regla delta")

cfg_datos = {"nota_min": float(nota_min), "n": int(n_est), "seed": int(seed_datos),
             "p_val": float(p_val), "p_test": float(p_test)}
datos = pc.construir_datos(cfg_datos["n"], cfg_datos["seed"], cfg_datos["nota_min"],
                           cfg_datos["p_val"], cfg_datos["p_test"])

tabs = st.tabs(["1 · Problema y teoría", "2 · Datos", "3 · Entrenamiento",
                "4 · Experimentos", "5 · Evaluación final y demo", "6 · Reproducibilidad y conclusiones"])


def pct(v):
    return f"{v:.1%}"


# ======================================================================
# 1 · PROBLEMA Y TEORÍA  (etapas 01, 02, 04, 09)
# ======================================================================
with tabs[0]:
    st.header("1 · Problema y fundamento teórico")

    st.subheader("Etapa 01 · Definición del problema")
    st.markdown(
        "Se desea entrenar una red neuronal artificial (RNA) que determine si un estudiante "
        "**aprobará o no una experiencia curricular**, bajo estas reglas del ejercicio:\n\n"
        "- **a.** Se consideran **3 promedios** (3 unidades).\n"
        "- **b.** El estudiante debe tener **por lo menos 2 unidades aprobadas** para aprobar.\n"
        "- **c.** Para el resultado final **no se calcula ningún promedio**: solo se aplica la regla b.\n"
        "- **d.** Los promedios de cada unidad están entre **0 y 20**.\n\n"
        f"**Nota aprobatoria usada:** {nota_min:g}. "
        "**Restricciones:** modelo de una sola neurona, salida binaria, entradas numéricas acotadas."
    )

    st.subheader("Etapa 02 · Tipo de aprendizaje")
    st.markdown(
        "**Aprendizaje supervisado de clasificación binaria**: cada patrón de entrada (3 promedios) "
        "tiene una salida deseada *yd* (1 = aprueba, 0 = desaprueba). La salida obtenida *y* se compara "
        "con *yd*, se calcula el error y se corrigen los pesos para minimizarlo. "
        "(El aprendizaje no supervisado, p. ej. redes SOM, no requiere salida deseada y no aplica aquí.)"
    )

    st.subheader("Fundamento: redes neuronales y neurona artificial")
    st.markdown(
        "Una RNA es una red de elementos simples (usualmente adaptativos) interconectados masivamente "
        "en paralelo, con organización jerárquica, que intenta interactuar con los objetos del mundo real "
        "de modo similar al sistema nervioso biológico."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.table(pd.DataFrame({
            "Neurona biológica": ["Dendritas", "Sinapsis", "Núcleo (cuerpo celular)", "Axón"],
            "Neurona artificial": ["Entradas (xj)", "Pesos sinápticos (wij)",
                                   "Sumatoria + función de activación f(a)", "Salida y = f(a)"],
        }).set_index("Neurona biológica"))
    with c2:
        st.markdown("**Funciones de activación típicas:** hardlim, hardlims, purelin, satlins, logsig, "
                    "tansig, radbas. Se usa **hardlim** porque la salida es binaria.")

    st.subheader("Etapa 04 · Entradas, salidas y objetivo")
    st.table(pd.DataFrame({
        "Componente": ["Entradas x1, x2, x3", "Pesos w1, w2, w3", "Sesgo (bias) wb", "Función de activación",
                       "Salida y", "Salida deseada yd", "Factor de aprendizaje fa"],
        "Descripción": [
            "Promedio de cada unidad (binarizado o normalizado según la representación elegida)",
            "Parámetros entrenables, uno por entrada", "Parámetro entrenable; su entrada es siempre 1",
            "Escalón (hardlim): 1 si NET ≥ 0, 0 si NET < 0", "1 = aprueba, 0 = desaprueba",
            "Resultado según la regla b del ejercicio", "Entre 0 y 1; controla el tamaño de cada corrección"],
    }).set_index("Componente"))

    st.subheader("Etapa 09 · Arquitectura y matemática")
    a1, a2 = st.columns([3, 2])
    with a1:
        st.graphviz_chart("""
        digraph G {
            rankdir=LR; bgcolor="transparent";
            node [shape=circle, style=filled, fillcolor="#EAF5EC", color="#2E8B57", fontname="Helvetica"];
            edge [color="#2E8B57", fontname="Helvetica", fontsize=10];
            x1 [label="x1\\n(U1)"]; x2 [label="x2\\n(U2)"]; x3 [label="x3\\n(U3)"];
            b  [label="Bias\\n(1)", fillcolor="#FFF6D6"];
            N  [label="Σ\\nNET", shape=circle, fillcolor="#C8E6C9"];
            F  [label="f(NET)\\nhardlim", shape=box, style="rounded,filled", fillcolor="#C8E6C9"];
            y  [label="y"];
            x1 -> N [label="w1"]; x2 -> N [label="w2"]; x3 -> N [label="w3"];
            b -> N [label="wb"]; N -> F; F -> y;
        }
        """)
    with a2:
        st.table(pd.DataFrame({
            "Elemento": ["Entrada x", "Pesos w", "Bias wb", "NET", "Salida y", "Parámetros entrenables"],
            "Dimensión": ["(3,)", "(3,)", "escalar", "escalar", "escalar {0,1}", "4 (w1, w2, w3, wb)"],
        }).set_index("Elemento"))
    st.latex(r"NET = \sum_{j=1}^{3} w_j\,x_j + w_b\cdot 1 \qquad y = f(NET)=\begin{cases}1 & NET \geq 0\\ 0 & NET < 0\end{cases}")
    st.latex(r"error = y_d - y \qquad W_i(t+1) = W_i(t) + f_a \cdot error \cdot x_i \qquad w_b(t+1) = w_b(t) + f_a\cdot error\cdot 1")
    st.markdown(
        "**Flujo del aprendizaje:** (1) se presenta un patrón; (2) se calcula NET; (3) se aplica f(NET); "
        "(4) se calcula el error; (5) si el error ≠ 0 se corrigen los pesos; (6) se repite con todos los "
        "patrones (una **época**) hasta que una época completa no produzca errores o se agote el máximo de épocas."
    )
    st.info(
        "**Límite teórico (Minsky & Papert, 1969):** un perceptrón simple solo separa clases "
        "**linealmente separables**. Con entradas binarias (aprobada/desaprobada) la regla «al menos 2 de 3» lo es "
        "(p. ej. w = [1, 1, 1], wb = −1.5). Con notas completas la regla depende de un umbral *por unidad*, "
        "que no es lineal en la nota, por lo que una neurona solo la aproxima."
    )

# ======================================================================
# 2 · DATOS  (etapas 03, 05, 06, 07)
# ======================================================================
with tabs[1]:
    st.header("2 · Datos y preprocesamiento")

    st.subheader("Etapa 03 · Obtención y trazabilidad")
    st.markdown(
        "No se usa un conjunto de datos externo: el ejercicio define las reglas, por lo que los datos son "
        "**sintéticos y generados con semilla fija**. Cada estudiante tiene 3 promedios enteros 0–20 "
        "(distribución uniforme) y su salida deseada se calcula con la regla b."
    )
    st.table(pd.DataFrame({
        "Parámetro": ["Generador", "Semilla", "Estudiantes", "Nota aprobatoria", "Distribución de notas", "Etiqueta yd"],
        "Valor": ["numpy.random.default_rng", str(cfg_datos["seed"]), str(cfg_datos["n"]), f"{nota_min:g}",
                  "Uniforme discreta {0, …, 20}", "1 si ≥ 2 unidades con nota ≥ nota aprobatoria, si no 0"],
    }).set_index("Parámetro"))

    st.subheader("Etapa 06 · Separación experimental")
    tr, va, te = datos["idx_train"], datos["idx_val"], datos["idx_test"]
    sep = pd.DataFrame({
        "Conjunto": ["Entrenamiento", "Validación", "Prueba (test)"],
        "Uso": ["Ajustar los pesos", "Comparar configuraciones y elegir la mejor", "Evaluación final, una sola vez"],
        "Estudiantes": [len(tr), len(va), len(te)],
        "% que aprueba": [pct(datos["yd"][i].mean()) for i in (tr, va, te)],
    }).set_index("Conjunto")
    st.table(sep)
    st.caption("La separación se hace por permutación aleatoria con la semilla indicada, antes de cualquier ajuste.")

    st.subheader("Etapa 05 · Análisis exploratorio (solo conjunto de entrenamiento)")
    notas_tr = datos["notas"][tr]
    yd_tr = datos["yd"][tr]
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Frecuencia de cada promedio por unidad**")
        conteo = pd.DataFrame({f"Unidad {j+1}": np.bincount(notas_tr[:, j], minlength=21) for j in range(3)})
        conteo.index.name = "Promedio"
        st.bar_chart(conteo)
    with e2:
        n_apr = np.array([pc.unidades_aprobadas(f, nota_min) for f in notas_tr])
        st.markdown("**Unidades aprobadas por estudiante**")
        dist = pd.DataFrame({"Estudiantes": np.bincount(n_apr, minlength=4)},
                            index=["0", "1", "2", "3"])
        dist.index.name = "Unidades aprobadas"
        st.bar_chart(dist)
    st.markdown(
        f"En entrenamiento aprueba el **{pct(yd_tr.mean())}** y desaprueba el **{pct(1 - yd_tr.mean())}**: "
        f"las clases están **desbalanceadas**, por eso la exactitud se compara siempre con una línea base "
        f"(predecir siempre la clase mayoritaria = {pct(pc.baseline_mayoritaria(datos, 'train'))})."
    )
    st.caption("Esta distribución es consecuencia de generar notas uniformes; no se extrapola a estudiantes reales.")

    st.subheader("Etapa 07 · Preprocesamiento: dos representaciones de entrada")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"**Binaria** — x = 1 si nota ≥ {nota_min:g}, si no 0. Tabla de verdad completa:")
        st.dataframe(pc.tabla_verdad(nota_min))
    with p2:
        st.markdown("**Normalizada** — x = nota / 20 (rango [0, 1]). Ejemplo de los primeros estudiantes:")
        ej = pd.DataFrame(notas_tr[:6], columns=["Nota U1", "Nota U2", "Nota U3"])
        norm = pd.DataFrame(pc.preparar_entrada(notas_tr[:6], pc.NORMALIZADO, nota_min).round(3),
                            columns=["x1", "x2", "x3"])
        st.dataframe(pd.concat([ej, norm], axis=1))

# ======================================================================
# 3 · ENTRENAMIENTO  (etapas 08, 10, 12)
# ======================================================================
with tabs[2]:
    st.header("3 · Modelo de referencia y entrenamiento")
    st.markdown("Se entrena **solo con el conjunto de entrenamiento**; la exactitud en validación se muestra "
                "como seguimiento. El test no se toca aquí.")

    st.subheader("Etapa 10 · Configuración de aprendizaje")
    k1, k2, k3 = st.columns(3)
    with k1:
        modo = st.selectbox("Representación de entrada", [pc.BINARIO, pc.NORMALIZADO])
        fa = st.slider("Factor de aprendizaje (fa)", 0.01, 1.0, 0.5, 0.01)
        max_ep = st.number_input("Máximo de épocas", 1, 5000, 100, 10)
    with k2:
        w1 = st.number_input("w1 inicial", value=0.5, step=0.1)
        w2 = st.number_input("w2 inicial", value=0.4, step=0.1)
        w3 = st.number_input("w3 inicial", value=0.3, step=0.1)
        wb0 = st.number_input("wbias inicial", value=1.5, step=0.1)
    with k3:
        mezclar = st.checkbox("Mezclar patrones en cada época", value=False)
        seed_train = st.number_input("Semilla de entrenamiento", 0, 9999, 0, 1)
        n_traza = st.number_input("Épocas a mostrar paso a paso", 1, 5, 1, 1)

    if st.button("▶️ Entrenar modelo de referencia", type="primary"):
        Xtr, ytr, _ = pc.subconjunto(datos, "train", modo)
        res = pc.entrenar(Xtr, ytr, [w1, w2, w3], wb0, fa, int(max_ep), mezclar, int(seed_train), int(n_traza))
        res.update({"modo": modo, "nota_min": cfg_datos["nota_min"], "fa": fa, "mezclar": mezclar,
                    "w_init": [w1, w2, w3], "wb_init": wb0, "seed_train": int(seed_train),
                    "cfg_datos": dict(cfg_datos)})
        st.session_state["modelo_ref"] = res

    mref = st.session_state.get("modelo_ref")
    if mref:
        if mref["cfg_datos"] != cfg_datos:
            st.warning("Cambiaste la configuración de datos de la barra lateral después de entrenar. "
                       "Vuelve a entrenar para que coincidan.")
        st.divider()
        st.subheader("Etapa 08 · Modelo de referencia entrenado")
        if mref["convergio"]:
            st.success(f"✅ Convergió: una época completa sin errores (época {mref['epocas']}).")
        else:
            st.warning(f"⚠️ No llegó a 0 errores en {mref['epocas']} épocas. Se conservan los pesos de la "
                       f"mejor época ({mref['epoca_mejor']}) con {mref['errores_finales']} error(es) "
                       f"sobre {mref['n_patrones']} patrones de entrenamiento.")
        _, m_tr, _ = pc.evaluar(mref, datos, "train")
        _, m_va, _ = pc.evaluar(mref, datos, "val")
        c = st.columns(6)
        c[0].metric("Épocas", mref["epocas"])
        c[1].metric("w1", f"{mref['w'][0]:.3f}")
        c[2].metric("w2", f"{mref['w'][1]:.3f}")
        c[3].metric("w3", f"{mref['w'][2]:.3f}")
        c[4].metric("wbias", f"{mref['wb']:.3f}")
        c[5].metric("Tiempo de entrenamiento", f"{mref['tiempo_s']*1000:.1f} ms")
        d1, d2, d3 = st.columns(3)
        d1.metric("Exactitud en entrenamiento", pct(m_tr["exactitud"]))
        d2.metric("Exactitud en validación", pct(m_va["exactitud"]))
        d3.metric("Línea base (clase mayoritaria, val.)", pct(pc.baseline_mayoritaria(datos, "val")))

        w = mref["w"]
        st.latex(rf"NET = {w[0]:.3f}\,x_1 + {w[1]:.3f}\,x_2 + {w[2]:.3f}\,x_3 + ({mref['wb']:.3f})"
                 r"\qquad y = f(NET)")

        st.subheader("Etapa 12 · Seguimiento de la ejecución")
        traza = mref["traza"]
        st.markdown("**Cálculo paso a paso** (NET → y → error → nuevos pesos), como en el ejemplo de clase. "
                    "Se muestran los primeros 15 patrones de la primera época:")
        st.dataframe(traza[traza["Época"] == 1].head(15).drop(columns="Época").set_index("Patrón"))
        with st.expander("Ver todas las épocas registradas"):
            st.dataframe(traza)
        g1, g2 = st.columns(2)
        hist = mref["historial"].set_index("Época")
        with g1:
            st.markdown("**Errores por época**")
            st.line_chart(hist[["Errores en la época", "Errores con pesos finales"]])
        with g2:
            st.markdown("**Evolución de los pesos**")
            st.line_chart(hist[["w1", "w2", "w3", "wbias"]])
    else:
        st.info("Configura los parámetros y pulsa **Entrenar modelo de referencia**.")

# ======================================================================
# 4 · EXPERIMENTOS  (etapas 11, 13, 14)
# ======================================================================
with tabs[3]:
    st.header("4 · Plan de hiperparámetros, validación y selección")
    st.markdown(
        "Se comparan configuraciones **con el mismo conjunto de entrenamiento y validación**, repitiendo cada "
        "una con varias semillas (orden de patrones y, si aplica, pesos iniciales). "
        "La selección usa **solo la exactitud de validación**; el test sigue reservado."
    )
    st.subheader("Etapa 11 · Plan")
    x1, x2 = st.columns(2)
    with x1:
        modos_sel = st.multiselect("Representaciones", [pc.BINARIO, pc.NORMALIZADO], default=[pc.BINARIO, pc.NORMALIZADO])
        fas_sel = st.multiselect("Factores de aprendizaje (fa)", [0.05, 0.1, 0.3, 0.5, 1.0], default=[0.1, 0.5, 1.0])
    with x2:
        inits_sel = st.multiselect("Pesos iniciales", [pc.INIT_FIJA, pc.INIT_ALEATORIA],
                                   default=[pc.INIT_FIJA, pc.INIT_ALEATORIA],
                                   help="Fija: w = [0.5, 0.4, 0.3], wb = 1.5 (valores de clase).")
        n_sem = st.slider("Semillas por configuración (etapa 13)", 5, 20, 10, 1)
        max_ep_exp = st.number_input("Máximo de épocas por corrida", 5, 500, 50, 5)

    if st.button("🔬 Ejecutar experimentos", type="primary"):
        if not (modos_sel and fas_sel and inits_sel):
            st.error("Elige al menos una opción en cada lista.")
        else:
            with st.spinner("Entrenando todas las configuraciones…"):
                runs, resumen = pc.ejecutar_experimentos(datos, modos_sel, fas_sel, inits_sel, int(n_sem), int(max_ep_exp))
            st.session_state["exp"] = {"runs": runs, "resumen": resumen, "cfg_datos": dict(cfg_datos),
                                       "n_sem": int(n_sem), "max_ep": int(max_ep_exp)}
            st.session_state.pop("final", None)

    exp = st.session_state.get("exp")
    if exp:
        if exp["cfg_datos"] != cfg_datos:
            st.warning("La configuración de datos cambió desde que ejecutaste los experimentos. Vuelve a ejecutarlos.")
        resumen, runs = exp["resumen"], exp["runs"]
        st.subheader("Etapa 13 · Resultados de validación (media sobre semillas)")
        st.dataframe(resumen.style.format({
            "Convergió (%)": "{:.0f}", "Épocas (media)": "{:.1f}", "Acc. val (media)": "{:.3f}",
            "Acc. val (desv.)": "{:.3f}", "Acc. train (media)": "{:.3f}", "Tiempo ms (media)": "{:.2f}"}))
        st.caption(f"{len(runs)} corridas = {exp['n_sem']} semillas por configuración · máximo {exp['max_ep']} épocas · "
                   f"tiempos medidos en esta ejecución.")

        etiqueta = resumen["Modo"] + " · fa=" + resumen["fa"].astype(str) + " · " + resumen["Init"]
        graf = pd.DataFrame({"Acc. val (media)": resumen["Acc. val (media)"].values}, index=etiqueta.values)
        gg1, gg2 = st.columns(2)
        with gg1:
            st.markdown("**Exactitud de validación por configuración**")
            st.bar_chart(graf)
        with gg2:
            st.markdown("**Épocas medias por configuración**")
            st.bar_chart(pd.DataFrame({"Épocas (media)": resumen["Épocas (media)"].values}, index=etiqueta.values))

        st.subheader("Etapa 14 · Selección de configuración")
        mejor = resumen.iloc[0]
        st.session_state["config_sel"] = {"Modo": mejor["Modo"], "fa": float(mejor["fa"]), "Init": mejor["Init"]}
        st.success(
            f"Configuración seleccionada: **{mejor['Modo']}**, fa = **{mejor['fa']}**, pesos iniciales "
            f"**{mejor['Init'].lower()}** → exactitud de validación {mejor['Acc. val (media)']:.3f} "
            f"± {mejor['Acc. val (desv.)']:.3f}, {mejor['Épocas (media)']:.1f} épocas, "
            f"convergencia en {mejor['Convergió (%)']:.0f}% de las corridas."
        )
        st.caption("Criterio: mayor exactitud de validación; en empate, menos épocas (menor costo).")

        st.subheader("Hallazgos (calculados con las corridas reales)")
        hall = []
        por_modo = runs.groupby("Modo").agg(conv=("Convergió", "mean"), val=("Acc. val", "mean"), ep=("Épocas", "mean"))
        if {"Binario", "Normalizado"} <= set(por_modo.index):
            b, n = por_modo.loc["Binario"], por_modo.loc["Normalizado"]
            hall.append(
                f"**Separabilidad lineal:** con entrada binaria converge el {b['conv']:.0%} de las corridas "
                f"(exactitud val. media {b['val']:.3f}); con notas normalizadas converge el {n['conv']:.0%} "
                f"(exactitud val. media {n['val']:.3f}). Una neurona no representa exactamente la regla con notas continuas.")
            hall.append(
                f"**Costo:** la representación binaria necesita {b['ep']:.1f} épocas en promedio frente a "
                f"{n['ep']:.1f} de la normalizada (tope {exp['max_ep']}).")
        base_val = pc.baseline_mayoritaria(datos, "val")
        hall.append(
            f"**Línea base:** predecir siempre la clase mayoritaria da {pct(base_val)} en validación; "
            f"la mejor configuración logra {pct(mejor['Acc. val (media)'])}. "
            f"Como las clases están desbalanceadas, la exactitud sola exagera el desempeño.")
        for modo_nombre, g in runs.groupby("Modo"):
            ep_fa = g.groupby("fa")["Épocas"].mean()
            if len(ep_fa) > 1:
                hall.append(
                    f"**Efecto de fa ({modo_nombre}):** épocas medias entre {ep_fa.min():.1f} y {ep_fa.max():.1f} "
                    f"según fa; exactitud val. entre {g.groupby('fa')['Acc. val'].mean().min():.3f} y "
                    f"{g.groupby('fa')['Acc. val'].mean().max():.3f}.")
        for i, h in enumerate(hall, 1):
            st.markdown(f"{i}. {h}")
        with st.expander("Ver las corridas individuales"):
            st.dataframe(runs)
    else:
        st.info("Pulsa **Ejecutar experimentos** para comparar configuraciones.")

# ======================================================================
# 5 · EVALUACIÓN FINAL Y DEMO  (etapas 15, 16)
# ======================================================================
with tabs[4]:
    st.header("5 · Evaluación final y demostración")
    sel = st.session_state.get("config_sel")
    exp = st.session_state.get("exp")

    st.subheader("Etapa 15 · Evaluación final sobre el conjunto de prueba")
    if not (sel and exp and exp["cfg_datos"] == cfg_datos):
        st.info("Primero ejecuta los experimentos (pestaña 4) con la configuración de datos actual: la "
                "configuración final se elige con validación y solo entonces se evalúa en test.")
    else:
        st.markdown(f"Configuración elegida en validación: **{sel['Modo']}**, fa = **{sel['fa']}**, "
                    f"pesos iniciales **{sel['Init'].lower()}**.")
        if st.button("🧪 Entrenar la configuración elegida y evaluar en test", type="primary"):
            modo_f = pc.BINARIO if sel["Modo"] == "Binario" else pc.NORMALIZADO
            init_f = pc.INIT_FIJA if sel["Init"] == "Fija" else pc.INIT_ALEATORIA
            Xtr, ytr, _ = pc.subconjunto(datos, "train", modo_f)
            w0, b0 = pc.pesos_iniciales(init_f, 0)
            res = pc.entrenar(Xtr, ytr, w0, b0, sel["fa"], exp["max_ep"], mezclar=True, seed=0)
            res.update({"modo": modo_f, "nota_min": cfg_datos["nota_min"], "fa": sel["fa"], "mezclar": True,
                        "w_init": w0, "wb_init": b0, "seed_train": 0, "init": init_f,
                        "cfg_datos": dict(cfg_datos)})
            st.session_state["final"] = res

    final = st.session_state.get("final")
    if final and final["cfg_datos"] == cfg_datos:
        df_te, m_te, mat_te = pc.evaluar(final, datos, "test")
        base_te = pc.baseline_mayoritaria(datos, "test")
        q = st.columns(5)
        q[0].metric("Exactitud (test)", pct(m_te["exactitud"]))
        q[1].metric("Precisión (aprueba)", pct(m_te["precision"]))
        q[2].metric("Recall (aprueba)", pct(m_te["recall"]))
        q[3].metric("F1 (aprueba)", pct(m_te["f1"]))
        q[4].metric("Línea base mayoritaria", pct(base_te))
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown("**Matriz de confusión (test)**")
            st.dataframe(mat_te)
            st.caption(f"Entrenamiento: {final['epocas']} época(s) · {final['tiempo_s']*1000:.1f} ms · "
                       f"{'convergió' if final['convergio'] else 'no convergió'}.")
        with r2:
            errs = df_te[~df_te["Correcto"]]
            st.markdown(f"**Casos mal clasificados en test ({len(errs)} de {len(df_te)})**")
            st.dataframe(errs)
        if m_te["exactitud"] <= base_te:
            st.warning("La exactitud no supera la línea base: el modelo no aporta más que predecir la clase mayoritaria.")
        st.caption("El test se evalúa una vez con la configuración elegida; no se usó para ajustar nada.")

    st.divider()
    st.subheader("Etapa 16 · Demostración interactiva")
    modelo_demo = final if (final and final["cfg_datos"] == cfg_datos) else st.session_state.get("modelo_ref")
    if not modelo_demo:
        st.info("Entrena un modelo (pestaña 3) o completa la evaluación final para usar la demostración.")
    else:
        origen = "modelo final (config. seleccionada)" if modelo_demo is final else "modelo de referencia (pestaña 3)"
        st.caption(f"Modelo cargado: **{origen}** · representación: {modelo_demo['modo']} · "
                   f"nota aprobatoria {modelo_demo['nota_min']:g}.")
        c1, c2, c3 = st.columns(3)
        n1 = c1.number_input("Promedio Unidad 1", 0.0, 20.0, 15.0, 0.5)
        n2 = c2.number_input("Promedio Unidad 2", 0.0, 20.0, 10.0, 0.5)
        n3 = c3.number_input("Promedio Unidad 3", 0.0, 20.0, 16.0, 0.5)
        notas = [n1, n2, n3]
        x, net, y = pc.predecir(modelo_demo, notas)
        w, wb_f = modelo_demo["w"], modelo_demo["wb"]
        esperado = pc.regla_objetivo(notas, modelo_demo["nota_min"])
        aprob = pc.unidades_aprobadas(notas, modelo_demo["nota_min"])
        st.markdown(
            f"**Entradas de la red:** x = [{', '.join(f'{v:.2f}' for v in x)}]  \n"
            f"**NET** = {w[0]:.3f}·{x[0]:.2f} + {w[1]:.3f}·{x[1]:.2f} + {w[2]:.3f}·{x[2]:.2f} "
            f"+ {wb_f:.3f}·1 = **{net:.4f}**  \n"
            f"**y = f(NET)** = {'1 (NET ≥ 0)' if y == 1 else '0 (NET < 0)'}")
        if y == 1:
            st.success("✅ La red predice: **APRUEBA** la experiencia curricular.")
        else:
            st.error("❌ La red predice: **DESAPRUEBA** la experiencia curricular.")
        st.caption(
            f"Verificación con la regla del ejercicio: {aprob} unidad(es) con nota ≥ {modelo_demo['nota_min']:g} → "
            f"{'aprueba' if esperado else 'desaprueba'}. "
            + ("La red coincide con la regla." if esperado == y else "⚠️ La red NO coincide con la regla en este caso."))
        st.markdown(
            "**Cómo interpretar la salida:** y = 1 significa que la red clasifica al estudiante como aprobado. "
            "No es una probabilidad ni una nota final; es una decisión binaria. **Admite** 3 promedios entre 0 y 20. "
            "**Limitaciones:** modelo de una sola neurona, datos sintéticos y sin otras variables (asistencia, "
            "exámenes sustitutorios, etc.).")

# ======================================================================
# 6 · REPRODUCIBILIDAD Y CONCLUSIONES
# ======================================================================
with tabs[5]:
    st.header("6 · Reproducibilidad, limitaciones y conclusiones")

    st.subheader("Entorno y versiones")

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "no instalado"

    st.table(pd.DataFrame({
        "Componente": ["Python", "NumPy", "pandas", "Streamlit", "Sistema", "Dispositivo"],
        "Versión": [sys.version.split()[0], ver("numpy"), ver("pandas"), ver("streamlit"),
                    platform.platform(), "CPU (el perceptrón no requiere GPU)"],
    }).set_index("Componente"))

    st.subheader("Configuración final registrada")
    final = st.session_state.get("final")
    cfg_final = {
        "datos": cfg_datos,
        "modelo_seleccionado": st.session_state.get("config_sel", "aún no seleccionado (ejecutar pestaña 4)"),
        "semillas": {"datos": cfg_datos["seed"], "entrenamiento_final": 0,
                     "experimentos": "0 … n_semillas-1"},
    }
    if final:
        cfg_final["pesos_finales"] = {"w": [round(float(v), 4) for v in final["w"]], "wbias": round(float(final["wb"]), 4)}
        cfg_final["epocas"] = final["epocas"]
    st.json(cfg_final)

    st.subheader("Pasos para reproducir")
    st.code("pip install -r requirements.txt\nstreamlit run app.py\n"
            "# 1) Fijar la barra lateral (nota 14, 400 estudiantes, semilla 42, 20% val, 20% test)\n"
            "# 2) Pestaña 3: entrenar modelo de referencia\n"
            "# 3) Pestaña 4: ejecutar experimentos (10 semillas)\n"
            "# 4) Pestaña 5: evaluar la configuración seleccionada en test", language="bash")

    st.subheader("Limitaciones y sesgos")
    st.markdown(
        "- **Datos sintéticos:** las notas se generan de forma uniforme y las etiquetas salen de la propia regla; "
        "no representan la distribución real de un salón de clases.\n"
        "- **Desbalance de clases:** con notas uniformes y nota aprobatoria 14 aprueba solo cerca de una cuarta parte "
        "(ver pestaña 2); por eso se reporta línea base, "
        "precisión, recall y F1, no solo exactitud.\n"
        "- **Modelo lineal de una neurona:** con notas continuas no puede representar la regla exacta; "
        "una red multicapa (p. ej. con backpropagation) sí podría.\n"
        "- **Resultado no generalizable** a otras reglas de aprobación, otras notas aprobatorias o más unidades "
        "sin reentrenar.\n"
        "- **Evaluación con un solo conjunto de prueba** de tamaño moderado: la exactitud tiene incertidumbre."
    )

    st.subheader("Conclusiones")
    st.markdown(
        "- Con entradas **binarias** el perceptrón converge en pocas épocas y reproduce la regla «al menos 2 unidades "
        "aprobadas»; con **notas normalizadas** solo la aproxima (ver cifras reales en la pestaña 4).\n"
        "- La selección de configuración se hizo con **validación**; el **test** se usó una sola vez al final.\n"
        "- El factor de aprendizaje y los pesos iniciales influyen en el número de épocas, pero el factor decisivo "
        "es la **representación de la entrada**."
    )

    with st.expander("Preparación para la sustentación (preguntas y respuestas guía)"):
        st.markdown(
            "**¿Qué hace la neurona?** Calcula NET = Σ wⱼxⱼ + wb y aplica el escalón: 1 si NET ≥ 0.\n\n"
            "**¿Qué aprende?** Los pesos sinápticos (w1, w2, w3 y wb) que hacen que cada entrada produzca la salida deseada.\n\n"
            "**¿Qué hace la regla delta?** Si error = yd − y ≠ 0, suma fa·error·xᵢ a cada peso; si error = 0 no cambia nada.\n\n"
            "**¿Por qué supervisado?** Porque cada patrón tiene su salida deseada yd.\n\n"
            "**¿Por qué binarizar las notas?** Porque «al menos 2 de 3 aprobadas» es linealmente separable solo "
            "sobre las unidades aprobadas/desaprobadas.\n\n"
            "**¿Por qué separar validación y test?** Para no elegir la configuración mirando el test, que da una estimación honesta.\n\n"
            "**¿Por qué comparar con una línea base?** Con clases desbalanceadas, predecir siempre «desaprueba» ya da una exactitud alta.\n\n"
            "**¿Cuándo se detiene el entrenamiento?** Cuando una época completa no produce errores o se alcanza el máximo de épocas."
        )

    st.subheader("Referencias (APA 7)")
    st.markdown(
        "- Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). "
        "Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2\n"
        "- Minsky, M., & Papert, S. (1969). *Perceptrons: An introduction to computational geometry*. MIT Press.\n"
        "- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. "
        "*Psychological Review, 65*(6), 386–408. https://doi.org/10.1037/h0042519\n"
        "- Streamlit. (s. f.). *Streamlit documentation*. https://docs.streamlit.io\n"
        f"- {DOCENTE}. (s. f.). *Ejemplo de perceptrón* [Diapositivas]. {CURSO}."
    )
