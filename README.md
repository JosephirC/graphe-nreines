# Le problème des N-Reines

Projet de résolution du problème des N-Reines – Approches complètes et incomplètes

## Contexte académique

Ce projet est réalisé dans le cadre du **Master 2 – Graphes et Complexité / Résolution de Problèmes Combinatoires**
Université Claude Bernard Lyon 1.

L’objectif est d’**étudier, implémenter et comparer** différentes **méthodes de résolution complètes et incomplètes** pour un problème NP-difficile classique, en respectant une méthodologie expérimentale rigoureuse (benchmarks, paramètres contrôlés, analyse).

## Problème étudié : les N-Reines

Le problème des **N-Reines** consiste à placer N reines sur un échiquier N×N de telle sorte qu’aucune reine ne puisse en attaquer une autre.

### Contraintes

- Une reine par ligne
- Une reine par colonne
- Une reine par diagonale (↘ et ↙)

Ce problème est un problème de satisfaction de contraintes (CSP) classique, souvent utilisé pour analyser :

- les stratégies de recherche exhaustive (backtracking),
- les heuristiques de branchement,
- les méthodes de recherche locale et incomplètes.

## Objectifs du projet

- Modéliser le problème des N-Reines en programmation par contraintes
- Implémenter :
  - des méthodes complètes (exhaustives, avec preuve)
  - des méthodes incomplètes (anytime, recherche opportuniste)
- Comparer ces méthodes à l’aide de benchmarks reproductibles
- Étudier l’impact :
  - des heuristiques de recherche
  - des paramètres du solveur
  - du temps de calcul        
  - de la taille du problème (N)

## Solveur utilisé

- Google OR-Tools – CP-SAT
- Raisons du choix :
  - Solveur de contraintes moderne et performant
  - Paramétrable (branching, stratégie de recherche, limites de temps)
  - Supporte à la fois recherche exhaustive et optimisation sous contraintess
  - Utilisable pour des approches hybrides (LNS, Min-Conflicts via optimisation)

## Architecture du projet

```bash 
.
├── README.md
├── benchmarks/
│   ├── benchmark1/              # Nombre de solutions trouvées en temps limité
│   ├── benchmark2/              # Temps jusqu’à la première solution
│
├── solvers/
│   ├── base_solver.py           # Classes abstraites et résultats
│   ├── complete/
│   │   ├── cp_sat_fixed_search_first_fail.py
│   │   └── cp_sat_fixed_search_center_out.py
│   └── incomplete/
│       ├── cp_sat_lns.py        # CP-SAT + Large Neighborhood Search
│       └── cp_sat_min_conflicts.py
│
├── archive/                     # Anciennes versions / scripts exploratoires
├── test_architecture.py
├── requirements.txt
```

## Méthodes implémentées

### Approches complètes (backtracking exhaustif)

1. First-Fail strict (min domaine)
   - Heuristique : variable au domaine minimal
   - Objectif : détecter les échecs le plus tôt possible

2. Ordre fixe structurel (center-out)
   - Ordre de branchement basé sur la géométrie de l’échiquier
   - Exploite la structure du problème

Options communes :
   - Activation / désactivation du symmetry breaking
   - Recherche imposée (FIXED_SEARCH)

### Approches incomplètes

1. CP-SAT + Large Neighborhood Search (LNS)
   - Recherche locale par grands voisinages
   - Fixation partielle de variables, relaxation contrôlée
   - Objectif : maximiser le nombre de solutions distinctes trouvées sous contrainte de temps

2. CP-SAT + Min-Conflicts (contraintes molles)
   - Autorise des solutions invalides
   - Minimise le nombre de conflits (colonnes / diagonales)
   - Une solution valide correspond à un coût nul
   - Méthode interrompue par limite de temps → incomplète

## Benchmarks

Deux benchmarks principaux sont utilisés pour toutes les méthodes :

1. Nombre de solutions distinctes trouvées en temps limité (ex. 30 ou 45 secondes)
2. Temps jusqu’à la première solution valide

Contraintes méthodologiques :
- `num_workers` = 1 (reproductibilité)
- mêmes instances (mêmes valeurs de N)
- mêmes limites de temps
- anti-doublons (hash des permutations)
- sorties normalisées (CSV)

Les résultats sont stockés dans :

```bash
benchmarks/*/runs/
```

## Installation et exécution

### Environnement

- Python : 3.12.3
- OS : Linux / WSL / macOS / Windows

### Création de l’environnement virtuel

```bash
python3.12 -m venv venv
source venv/bin/activate
```


### Installation des dépendances

``` bash
pip install -r requirements.txt
```


## Auteurs

Youssef Abida

Nathan Corroller

Khalissa Rhoulam

Master 2 – Université Claude Bernard Lyon 1