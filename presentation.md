# PROJET GRAPHE ET COMPLEXITÉ

## Problème Choisi : Les N-Reines

* **Définition :** Placer **N reines** sur un échiquier **N×N** sans qu'aucune ne puisse en menacer une autre.
* **Contraintes :**
    1.  Une seule reine par **ligne**.
    2.  Une seule reine par **colonne**.
    3.  Une seule reine par **diagonale**.
* **Type :** C'est un problème de satisfaction de contraintes (CSP) classique, utilisé pour tester les algorithmes de *backtracking*.

## Le Modèle CSP

* **Objectif :** Définir les Variables (X), les Domaines (D) et les Contraintes (C).
* **Notre Modèle (Intelligent) :**
    * **Variables (X) :** Nous utilisons **N** variables entières (une par ligne).
        * *Code :* `queens = [model.NewIntVar(0, n - 1, f'x{i}') for i in range(n)]`
    * **Sémantique :** `queens[i] = j` signifie que la reine de la **ligne `i`** est placée sur la **colonne `j`**.
    * **Domaines (D) :** Chaque variable peut prendre une valeur de `0` à `n-1`.
    * **Contraintes (C) :**
        1.  **Lignes :** Garanti implicitement par le modèle (une variable par ligne).
        2.  **Colonnes :** Toutes les variables doivent avoir des valeurs différentes.
            * *Code :* `model.AddAllDifferent(queens)`
        3.  **Diagonales :** Les "offsets" de diagonales doivent aussi être différents.
            * *Code :* `model.AddAllDifferent([queens[i] + i ...])`
            * *Code :* `model.AddAllDifferent([queens[i] - i ...])`

## Le Solveur et les Outils

* **Solveur :** **Google OR-Tools** (bibliothèque open-source).
    * **Module :** `cp_model`, le solveur **CP-SAT**, qui est un solveur de contraintes moderne et très performant.
* **Paramétrable ?**
    * **Oui.** Le projet demande de "comparer au moins deux méthodes de résolution complète".

## Les Méthodes Comparées (Nos Paramètres)

* Notre code compare **deux méthodes de résolution complètes** en modifiant les paramètres de recherche d'OR-Tools.

* **Méthode 1 : "DEFAULT"**
    * **Description :** On laisse OR-Tools utiliser sa stratégie de recherche par défaut (`search_branching = 0`).
    * **Concept :** C'est une heuristique complexe et optimisée (type "boîte noire") qui apprend pendant la recherche.

* **Méthode 2 : "HEURISTIC" (Fixe/Naïve)**
    * **Description :** On force le solveur à utiliser une heuristique simple et statique (`search_branching = 1`).
    * **Nos 2 Paramètres :**
        1.  **Choix de Variable :** `cp_model.CHOOSE_FIRST` (toujours dans l'ordre : `x0`, `x1`, `x2`...).
        2.  **Choix de Valeur :** `cp_model.SELECT_MIN_VALUE` (toujours essayer les colonnes dans l'ordre : `0`, `1`, `2`...).
    * **Objectif :** Comparer la performance de cette méthode "naïve" à la méthode par défaut d'OR-Tools.