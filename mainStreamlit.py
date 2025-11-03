import streamlit as st
import numpy as np
from main import solve_n_queens

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(page_title="♛ N-Reines", page_icon="♛", layout="centered")
st.title("♛ Visualisation du problème des N-Reines")

# -----------------------------
# Paramètres utilisateur
# -----------------------------
n = st.slider("Nombre de reines (taille de l’échiquier)", 4, 14, 8)
count_all = st.checkbox(
    "Compter toutes les solutions (plus lent)", value=False)
max_solutions = None if count_all else st.number_input(
    "Nombre max de solutions à afficher", 1, 200, 10)

# -----------------------------
# Résolution (mise en cache)
# -----------------------------


@st.cache_data(show_spinner=False)
def compute_solutions(n, max_solutions):
    return solve_n_queens(n, max_solutions)


if st.button("Résoudre"):
    with st.spinner("Résolution en cours..."):
        solutions, time = compute_solutions(n, max_solutions)
        st.session_state["solutions"] = solutions
        st.session_state["time"] = time
        st.session_state["n"] = n

# -----------------------------
# Affichage des résultats
# -----------------------------
if "solutions" in st.session_state:
    total = len(st.session_state["solutions"])
    st.success(
        f"✅ {total} solution(s) trouvée(s) en {st.session_state['time']:.3f} secondes")

    # Affichage du nombre total (avec message clair)
    if count_all:
        st.info(f"Nombre total exact de solutions pour N = {n} : **{total}**")
    else:
        st.caption(
            f"Seulement les {total} premières solutions sont affichées (pour aller plus vite).")

    # Sélection d’une solution à afficher
    idx = st.slider("Solution à visualiser :", 1, total, 1)
    sol = st.session_state["solutions"][idx - 1]
    n = st.session_state["n"]

    # Construction du plateau
    board = np.zeros((n, n))
    for i in range(n):
        board[i, sol[i]] = 1

    st.write(f"**Solution {idx} :** {sol}")

    # Affichage graphique de l’échiquier
    st.markdown("### 🧩 Échiquier")
    for i in range(n):
        row = ""
        for j in range(n):
            cell = "♛" if board[i, j] == 1 else "⬜" if (
                i + j) % 2 == 0 else "⬛"
            row += cell + " "
        st.markdown(row, unsafe_allow_html=True)
