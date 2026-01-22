# PROJET GRAPHE ET COMPLEXITÉ

## Problème Choisi : Les N-Reines

* **Définition :** Placer **N reines** sur un échiquier **N×N** sans qu'aucune ne puisse en menacer une autre.
* **Contraintes :**
  1. Une seule reine par **ligne**.
  2. Une seule reine par **colonne**.
  3. Une seule reine par **diagonale**.
* **Type :** C'est un problème de satisfaction de contraintes (CSP) classique, utilisé pour tester les algorithmes de *backtracking*.

## Le Modèle CSP

* **Objectif :** Définir les Variables (X), les Domaines (D) et les Contraintes (C).
* **Notre Modèle (Intelligent) :**
  * **Variables (X) :** Nous utilisons **N** variables entières (une par ligne).
    * *Code :* `queens = [model.NewIntVar(0, n - 1, f'x{i}') for i in range(n)]`
  * **Sémantique :** `queens[i] = j` signifie que la reine de la **ligne `i`** est placée sur la **colonne `j`**.
  * **Domaines (D) :** Chaque variable peut prendre une valeur de `0` à `n-1`.
  * **Contraintes (C) :**
    1. **Lignes :** Garanti implicitement par le modèle (une variable par ligne).
    2. **Colonnes :** Toutes les variables doivent avoir des valeurs différentes.
       * *Code :* `model.AddAllDifferent(queens)`
    3. **Diagonales :** Les "offsets" de diagonales doivent aussi être différents.
       * *Code :* `model.AddAllDifferent([queens[i] + i ...])`
       * *Code :* `model.AddAllDifferent([queens[i] - i ...])`

Absolument. C'est un point crucial. Voici une version modifiée de ces deux sections qui explique *comment* le solveur est paramétrable, en se basant sur les principes de votre cours et le code que vous avez écrit.

## Le Solveur et son Paramétrage

* **Solveur :** Nous utilisons **Google OR-Tools**, et spécifiquement son module `cp_model` (CP-SAT).
* **Paramétrable ?**
  * **Oui.**
  * Dans le contexte des CSP, une "méthode de résolution" est une recherche par *backtracking* (retour sur trace). Les **paramètres** les plus importants de cette recherche sont les **heuristiques** :
    1. **Heuristique de choix de variable :** *Quelle variable choisir en premier pour lui assigner une valeur ?*
    2. **Heuristique de choix de valeur :** *Quelle valeur de son domaine essayer en premier ?*
* **Comment on le paramètre dans OR-Tools ?**
  * Notre code contrôle ces heuristiques en passant un simple `string` ("DEFAULT" ou "HEURISTIC") à notre fonction `solve_n_queens`.
  * En fonction de ce `string`, nous modifions deux paramètres du solveur :
    1. `solver.parameters.search_branching` : Dit au solveur s'il doit utiliser sa propre magie ("DEFAULT") ou suivre notre stratégie fixe ("HEURISTIC").
    2. `model.AddDecisionStrategy(...)` : C'est ici que nous définissons *explicitement* nos 2 paramètres (heuristique de variable et heuristique de valeur).

## Les Méthodes Comparées (Nos Paramètres)

Notre code implémente et compare deux méthodes de résolution complètes distinctes.

### Méthode 1 : "DEFAULT"

* **Description :** C'est la méthode "boîte noire" et optimisée d'OR-Tools.
* **Paramétrage dans notre code :**
  * Nous la sélectionnons en passant `strategy="DEFAULT"`.
  * Techniquement, cela positionne `solver.parameters.search_branching = 0` (AUTO_SEARCH).
* **Concept :** On ne définit **aucune** heuristique manuelle. On laisse le solveur CP-SAT utiliser ses propres stratégies adaptatives complexes (par exemple, des heuristiques dynamiques comme `minDomaine/Degré`, ou des techniques d'apprentissage) pour trouver la solution le plus vite possible.

### Méthode 2 : "HEURISTIC" (Fixe/Naïve)

* **Description :** C'est notre méthode manuelle, simple et statique, qui utilise des heuristiques de base.
* **Paramétrage dans notre code :**
  * Nous la sélectionnons en passant `strategy="HEURISTIC"`.
  * Cela positionne `solver.parameters.search_branching = 1` (FIXED_SEARCH), forçant le solveur à obéir à notre stratégie.
  * Puis, nous définissons les **deux paramètres** requis par le projet, via la fonction `model.AddDecisionStrategy` :
    1. **Paramètre 1 (Choix de Variable) :** `cp_model.CHOOSE_FIRST`.
       * *Signification :* L'heuristique la plus simple. Le solveur choisit toujours les variables dans l'ordre fixe où nous les avons créées (`x0`, puis `x1`, puis `x2`...).
    2. **Paramètre 2 (Choix de Valeur) :** `cp_model.SELECT_MIN_VALUE`.
       * *Signification :* Le solveur essaie toujours les valeurs dans l'ordre croissant (colonne `0`, puis `1`, puis `2`...).

* **Objectif de la comparaison :**
  * Nous allons mesurer l'impact de ces paramètres en comparant le temps d'exécution de la recherche "naïve" (`HEURISTIC`) contre la recherche optimisée (`DEFAULT`).
