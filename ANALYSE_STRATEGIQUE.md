# ChessLab — Analyse strategique et backlog prioritaire

## 1. Comprehension du projet

### Proposition de valeur

ChessLab est une plateforme d'analyse et d'entrainement aux echecs qui combine vision par ordinateur (capture d'ecran, detection d'echiquier, reconnaissance de pieces) et analyse moteur (Stockfish via UCI). Le projet adresse un besoin reel : permettre a un joueur d'echecs d'obtenir une analyse en temps reel de ses parties, qu'elles soient jouees sur un site tiers (chess.com, lichess) ou directement dans l'interface integree.

Le produit s'adresse a un joueur d'echecs intermediaire a avance qui souhaite progresser en comprenant mieux ses positions en temps reel, et non simplement apres coup.

### Etat de maturite

**Version 0.1.0 — MVP fonctionnel.** Le projet contient 11 modules (environ 6 600 lignes de production, 6 400 lignes de tests), une architecture en pipeline propre, des types annotes partout, et une couverture de tests serieuse. La qualite d'ingenierie est nettement superieure a ce qu'on attend d'un v0.1 : hierarchies d'erreurs typees, dataclasses immutables, gestion du threading, separation de responsabilites claire.

### Forces visibles

- **Architecture composable.** Chaque module (capture, detection, splitting, reconnaissance, FEN, analyse) est utilisable independamment et testable en isolation. Le pipeline se compose naturellement.
- **Deux modes d'utilisation distincts et complementaires.** Le pipeline live (capture ecran → detection → reconnaissance → FEN → analyse → overlay) et le GUI interactif (plateau cliquable + coaching Stockfish avec Elo configurable).
- **Robustesse technique.** Resolution multi-strategie de Stockfish (variable d'environnement, PATH, chemins fallback), detection de positions terminales avant appel moteur, gestion cooperative du threading, debounce sur le monitoring de positions.
- **Base de tests solide.** Tests unitaires avec mocks synthetiques (pas de dependance a des screenshots reels ni a un binaire Stockfish en CI). 17 fichiers de tests couvrant chaque module.

### Limites actuelles

1. **Le GUI et le pipeline live sont deconnectes.** C'est la fracture centrale du projet. D'un cote, une chaine complete de vision par ordinateur capable de lire une position depuis un ecran. De l'autre, un GUI ou l'utilisateur doit saisir manuellement les coups de l'adversaire. Les deux moities du projet ne communiquent pas. Cela reduit considerablement la valeur du travail de vision par ordinateur pour l'utilisateur final du GUI.

2. **Aucune persistance des parties.** Les parties jouees dans le GUI disparaissent a la fermeture. Pas d'export PGN, pas d'historique. La valeur d'entrainement est ephemere — un joueur ne peut pas revenir sur ses erreurs plus tard.

3. **Pas de mode revue post-partie.** L'analyse est uniquement en temps reel. Or, la progression aux echecs passe majoritairement par la revue de ses parties apres coup (identification des inaccuracies, erreurs, gaffes). Le produit genere de l'analyse mais ne capitalise pas dessus.

4. **Reconnaissance limitee a un seul theme operationnel.** Les templates de pieces n'existent que pour le theme "green". Les 7 autres themes ont des signatures de couleur mais pas de templates. Cela limite l'utilisation du pipeline live aux sites configurant ce theme precis.

### Hypotheses prises

- **Utilisateur cible.** Joueur intermediaire (1200-2000 Elo) qui joue regulierement en ligne et souhaite un outil de coaching en temps reel. Hypothese basee sur le systeme d'Elo configurable (1320-3190) et le mode coaching avec suggestions de coups.
- **Environnement d'utilisation.** Principalement Linux (X11) d'apres l'overlay click-through et les scripts de capture. Le workflow Windows→Ubuntu via HTTP suggere un utilisateur technique a l'aise avec une configuration multi-machine.
- **Usage du pipeline live.** Destine a observer des parties jouees sur un navigateur (chess.com/lichess), pas a jouer directement. Le GUI interactif couvre le cas "jouer contre le moteur".

---

## 2. Vision pour la suite

Le projet a atteint un MVP ou les deux grandes chaines de valeur (vision par ordinateur et coaching interactif) fonctionnent en isolation. **Le prochain cap logique est de fusionner ces deux chaines en une experience utilisateur unifiee, puis de capitaliser sur les donnees generees pour offrir une valeur durable a l'utilisateur.**

Concretement, trois axes se dessinent par ordre de priorite :

1. **Connecter la perception au coaching** — faire en sorte que le pipeline de vision par ordinateur alimente automatiquement le GUI, eliminant la saisie manuelle des coups adverses.
2. **Rendre la valeur durable** — persister les parties et permettre l'export PGN pour que le travail d'analyse ne soit pas perdu.
3. **Amplifier la valeur pedagogique** — proposer une revue post-partie avec classification des erreurs, transformant un outil d'analyse en temps reel en un veritable outil de progression.

Ces trois axes s'enchainent logiquement : le premier augmente l'utilisation, le deuxieme capture la valeur generee, le troisieme l'exploite pour la progression du joueur.

---

## 3. Backlog prioritaire

---

### Item 1 — Detection automatique des coups adverses via capture d'ecran dans le GUI

**Nature :** Nouvelle fonctionnalite — integration de modules existants.

**Probleme adresse**

Aujourd'hui, lorsqu'un utilisateur suit une partie sur chess.com ou lichess, il doit saisir manuellement chaque coup adverse dans le GUI pour obtenir l'analyse Stockfish. Cette friction rend le workflow live pratiquement inutilisable : le temps de saisir le coup, la position a deja evolue. Par ailleurs, tout le travail de vision par ordinateur (modules `capture`, `detection`, `splitting`, `recognition`, `fen`, `monitoring`, `live`) n'est accessible qu'a travers un script serveur HTTP ou une CLI, jamais depuis l'interface graphique.

**Valeur attendue**

- Elimine la friction principale du workflow d'analyse live.
- Transforme le GUI en un outil d'assistance en temps reel reellement utilisable pendant une partie en ligne.
- Donne une raison d'etre concrete a 6 des 11 modules du projet, qui aujourd'hui n'ont aucun chemin utilisateur fluide.
- Differencie ChessLab d'un simple wrapper Stockfish.

**User story**

> En tant que joueur d'echecs en ligne, je veux que ChessLab detecte automatiquement les coups joues sur mon ecran, afin d'obtenir des suggestions Stockfish en temps reel sans avoir a saisir les coups manuellement.

**Criteres d'acceptation**

1. Le GUI propose un mode "Observation live" activable depuis l'interface (en plus du mode interactif actuel).
2. En mode observation, le systeme capture periodiquement l'ecran, detecte l'echiquier, reconnait la position et genere le FEN.
3. Lorsqu'un changement de position est detecte (via le module `monitoring`), le coup est automatiquement identifie et soumis au `GameCoach`.
4. L'analyse Stockfish (evaluation, variations, meilleur coup) se met a jour automatiquement dans les panneaux lateraux.
5. L'utilisateur peut selectionner la region d'ecran a surveiller (pour eviter les faux positifs sur d'autres fenetres).
6. L'utilisateur peut choisir le theme de l'echiquier distant parmi les themes supportes.
7. Le mode observation peut etre arrete et repris sans perdre l'historique de la partie en cours.
8. En cas d'echec de detection (echiquier non trouve, theme non reconnu), un message clair est affiche dans la barre de statut sans interrompre le monitoring.

**Impact :** Tres eleve — c'est la fonctionnalite qui donne sa coherence au projet entier.

**Complexite :** Moyenne — les modules existent deja, l'effort porte sur l'integration dans le GUI, la gestion du thread de capture et la deduction du coup joue a partir du delta de FEN.

**Raison de la priorite :** C'est le pont entre les deux moities du projet. Sans cette integration, la chaine de vision par ordinateur et le GUI restent deux produits separes. C'est aussi la fonctionnalite qui differencie ChessLab d'un simple GUI Stockfish dont il existe des dizaines d'alternatives.

**Risques et points de vigilance**

- La deduction du coup a partir de deux FEN consecutifs n'est pas triviale (promotions, roques, prises en passant). `python-chess` peut resoudre cela en comparant les positions legales.
- La fiabilite de la reconnaissance de pieces est critique : une erreur de reconnaissance genere un FEN invalide et casse la chaine. Il faudra gerer gracieusement les frames incertaines (les ignorer plutot que soumettre un coup faux).
- La performance de capture + detection + reconnaissance doit rester sous le seuil de latence acceptable (< 2 secondes par cycle) pour que l'experience soit fluide.
- Le nombre de themes avec templates operationnels (actuellement 1 seul : "green") limite l'applicabilite. Un mecanisme de generation ou d'extraction de templates serait un complement utile a moyen terme (le script `extract_templates_auto.py` existe deja).

---

### Item 2 — Persistance des parties et export PGN

**Nature :** Nouvelle fonctionnalite — persistance et interoperabilite.

**Probleme adresse**

Chaque partie jouee ou observee dans ChessLab disparait a la fermeture de l'application. L'utilisateur ne peut ni revoir ses parties, ni les partager, ni les importer dans un autre outil. Pour un outil de progression, c'est une limitation majeure : la valeur de l'analyse n'existe que dans l'instant. Le format PGN est le standard universel — tout outil d'echecs (lichess, chess.com, Chessbase, SCID) peut l'importer.

**Valeur attendue**

- Les parties jouees deviennent un actif durable pour l'utilisateur.
- L'export PGN ouvre l'interoperabilite avec l'ecosysteme echiqueen existant.
- La persistence est un prerequis pour la revue post-partie (Item 3).
- Cout d'implementation faible grace a `python-chess` qui supporte nativement la lecture et l'ecriture PGN.

**User story**

> En tant que joueur utilisant ChessLab pour m'entrainer, je veux que mes parties soient sauvegardees automatiquement et exportables au format PGN, afin de pouvoir les revoir plus tard ou les analyser dans d'autres outils.

**Criteres d'acceptation**

1. Chaque partie terminee (ou en cours a la fermeture) est automatiquement sauvegardee dans un repertoire local (`~/.chesslab/games/`).
2. Le fichier sauvegarde contient le PGN complet avec les en-tetes standard (Event, Date, White, Black, Result, TimeControl).
3. L'Elo du moteur est enregistre dans les en-tetes PGN (tag `WhiteElo` ou `BlackElo` selon le cote du moteur).
4. Un bouton "Exporter PGN" dans le GUI permet de sauvegarder la partie en cours dans un fichier choisi par l'utilisateur.
5. Un bouton "Charger PGN" permet d'importer une partie existante et de la visualiser dans le GUI.
6. Les parties sauvegardees automatiquement sont listees dans un panneau "Parties recentes" accessible depuis le GUI.
7. Le format PGN genere est valide et importable sans erreur dans lichess et chess.com.

**Impact :** Eleve — prerequis structurant pour toute fonctionnalite de revue et condition necessaire pour que l'outil ait une valeur au-dela de la session.

**Complexite :** Faible — `python-chess` fournit `chess.pgn` avec lecture/ecriture native. L'essentiel du travail est cote GUI (panneau de gestion, dialogues de fichiers).

**Raison de la priorite :** C'est un enabler. Sans persistance, l'Item 3 (revue post-partie) est impossible. Par ailleurs, c'est une attente de base pour tout outil d'echecs — son absence sera percue comme un manque evident par tout joueur habitue aux plateformes existantes.

**Risques et points de vigilance**

- Les parties issues du mode observation live (Item 1) doivent egalement etre sauvegardables. La deduction des coups doit etre suffisamment fiable pour produire un PGN coherent.
- Le `GameCoach` maintient deja un `move_history` — l'effort d'integration est faible.
- Gestion de l'espace disque : les fichiers PGN sont legers (quelques Ko par partie), pas de risque reel.
- Le chargement de PGN avec variantes ou commentaires n'est pas necessaire en v1 — se limiter au PGN lineaire.

---

### Item 3 — Revue post-partie avec classification des erreurs

**Nature :** Nouvelle fonctionnalite — valeur pedagogique.

**Probleme adresse**

L'analyse en temps reel est utile pendant la partie, mais la progression aux echecs repose fondamentalement sur la revue post-partie. Aujourd'hui, ChessLab analyse en temps reel mais ne permet pas de revenir sur une partie terminee pour comprendre ou le joueur s'est trompe. Les plateformes comme lichess et chess.com proposent cette fonctionnalite — mais uniquement pour les parties jouees sur leur site. ChessLab pourrait offrir cette revue pour n'importe quelle partie (jouee localement, observee via capture, ou importee via PGN).

**Valeur attendue**

- Transforme ChessLab d'un outil d'observation en un veritable outil de progression.
- La classification des coups (excellent, bon, imprécision, erreur, gaffe) est le mecanisme pedagogique le plus efficace et le plus demande par les joueurs.
- Exploite pleinement le moteur Stockfish deja integre — il suffit d'analyser chaque position de la partie et de comparer le coup joue au meilleur coup.
- Forte retention utilisateur : la revue post-partie est la fonctionnalite qui fait revenir les joueurs.

**User story**

> En tant que joueur souhaitant progresser, je veux pouvoir revoir une partie terminee coup par coup avec une classification de la qualite de chaque coup, afin d'identifier mes erreurs et de comprendre les meilleures alternatives.

**Criteres d'acceptation**

1. Depuis le panneau "Parties recentes" ou apres une partie terminee, l'utilisateur peut lancer une revue.
2. Le mode revue affiche la partie coup par coup avec navigation avant/arriere (boutons et raccourcis clavier fleches).
3. Chaque coup du joueur est classifie selon la perte d'evaluation : excellent (0-10 cp), bon (10-30 cp), imprecision (30-80 cp), erreur (80-200 cp), gaffe (> 200 cp ou mat perdu). Les seuils sont configurables.
4. Le panneau lateral affiche pour chaque position : l'evaluation du coup joue, l'evaluation du meilleur coup, la difference, et la ligne principale du meilleur coup.
5. Un resume de la partie est affiche : nombre de coups par categorie, evaluation moyenne par phase (ouverture, milieu de jeu, finale), moments critiques.
6. Les positions ou l'utilisateur a commis une erreur ou une gaffe sont marquees visuellement et navigables directement.
7. La revue fonctionne sur les parties jouees dans le GUI, les parties observees en live, et les parties importees via PGN.

**Impact :** Eleve — c'est la fonctionnalite qui convertit un outil technique en produit de coaching a forte valeur percue.

**Complexite :** Moyenne a elevee — necessite l'analyse Stockfish de chaque position de la partie (potentiellement 40-80 positions), un mode de navigation dans le GUI, et la logique de classification. L'analyse peut etre realisee en arriere-plan apres la partie.

**Raison de la priorite :** Troisieme car elle depend des Items 1 et 2 (sans persistance, pas de revue ; sans detection automatique, le corpus de parties a revoir sera faible). Mais c'est la fonctionnalite qui donne sa raison d'etre a long terme au produit.

**Risques et points de vigilance**

- L'analyse de 60+ positions a profondeur 18 prend du temps (potentiellement 2-4 minutes selon la machine). Prevoir une barre de progression et un mode d'analyse en arriere-plan.
- La classification par perte de centipawns est l'approche standard mais peut donner des resultats contre-intuitifs dans les positions avec un seul coup legal ou les positions deja perdues. Filtrer les positions ou l'ecart entre les deux meilleurs coups est negligeable.
- La distinction ouverture/milieu/finale peut etre approximee par le numero de coup et le materiel restant — pas besoin d'un modele sophistique en v1.
- La profondeur d'analyse pour la revue peut etre inferieure a celle du temps reel (profondeur 16 suffit pour la classification), ce qui accelere le processus.

---

## 4. Recommandation strategique

**Lancer en premier : Item 1 — Detection automatique des coups adverses.**

C'est le choix le plus structurant pour trois raisons :

1. **Coherence produit.** Aujourd'hui, ChessLab est un projet coupe en deux : une chaine de vision par ordinateur sans interface utilisateur fluide, et un GUI sans lien avec cette chaine. Fusionner les deux donne au projet son identite et sa proposition de valeur unique. Sans cette fusion, ChessLab reste un exercice technique remarquable mais un produit incomplet.

2. **Effet de levier maximal.** Six modules sur onze (capture, detection, splitting, reconnaissance, monitoring, live) ont ete construits pour ce cas d'usage. Les connecter au GUI active instantanement la valeur de tout ce travail anterieur. Le ratio valeur produite / effort nouveau est le plus favorable des trois items.

3. **Generateur de contenu pour la suite.** Une fois que les parties sont detectees automatiquement, le volume de parties jouees et analysables augmente naturellement. Cela cree la masse critique necessaire pour que la persistance (Item 2) et la revue post-partie (Item 3) aient un sens. Sans detection automatique, l'utilisateur ne jouera que des parties manuelles contre le moteur — un cas d'usage deja bien couvert par des outils existants.

**Benefice global :** En realisant ces trois items dans l'ordre propose, ChessLab passera d'un MVP technique a un produit de coaching echiqueen complet et differencie — capable de lire automatiquement une partie en cours sur n'importe quel site, de l'analyser en temps reel, de la sauvegarder, et de la revoir avec une classification pedagogique des erreurs. C'est une trajectoire coherente qui maximise la valeur de chaque module deja construit.
