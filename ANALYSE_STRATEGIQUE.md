# ChessLab --- Analyse strategique et backlog prioritaire (v2)

## 1. Comprehension du projet

### Proposition de valeur

ChessLab est une plateforme de coaching echiqueen qui combine vision par ordinateur et analyse Stockfish pour offrir une experience unique : observer automatiquement une partie jouee sur un site tiers (chess.com, lichess) via capture d'ecran, l'analyser en temps reel, la sauvegarder, puis la revoir avec une classification pedagogique de chaque coup. Le projet adresse un joueur intermediaire a avance (1200-2000 Elo) qui souhaite progresser en comprenant ses erreurs, aussi bien en temps reel qu'apres la partie.

### Etat de maturite

**Version 0.1.0 --- MVP complet.** Le projet contient 16 modules (~9 800 lignes de production, ~8 100 lignes de tests, 547 tests passants). Les trois axes strategiques identifies dans la premiere analyse ont tous ete realises :

1. **Detection automatique des coups** --- Le mode observation live integre le pipeline CV complet dans le GUI. La capture d'ecran alimente automatiquement le coaching Stockfish sans saisie manuelle.
2. **Persistance des parties** --- Sauvegarde automatique, export/import PGN, panneau de parties recentes, en-tetes standards avec Elo moteur.
3. **Revue post-partie** --- Module `review` complet avec classification des coups (excellent/bon/imprecision/erreur/gaffe/force), detection de phases (ouverture/milieu/finale), resume statistique, navigation par moments critiques.

Le projet a franchi un cap significatif : il n'est plus un assemblage de modules techniques independants mais un produit coherent avec une boucle complete capturer-analyser-sauvegarder-revoir.

### Forces visibles

- **Boucle utilisateur complete.** Le chemin capture → detection → reconnaissance → analyse → coaching → sauvegarde → revue fonctionne de bout en bout. C'est rare pour un projet a ce stade de maturite.
- **Qualite d'ingenierie constante.** Les nouveaux modules (review, persistence, panels de revue) maintiennent le meme niveau que le code initial : dataclasses immutables, hierarchies d'erreurs typees, annulation cooperative via `threading.Event`, separation stricte entre logique metier et affichage.
- **Architecture GUI bien structuree.** Les panneaux sont des widgets d'affichage purs sans logique metier. Le basculement play/review dans `chess_gui.py` est gere par un flag d'etat propre. Les workers Stockfish tournent en threads daemon avec communication par file d'attente.
- **Couverture de tests des nouveaux modules.** Le module `review` est teste des sa creation (25+ tests couvrant classification, phases, coups forces, mat perdu, annulation).

### Limites actuelles

1. **Reconnaissance limitee a un seul theme.** Les templates de pieces n'existent que pour le theme "green". Sept autres themes ont des signatures de couleur (detection automatique) mais aucun template operationnel. Cela signifie que le mode observation live --- la fonctionnalite differenciante du projet --- ne fonctionne que si l'utilisateur configure son site d'echecs sur le theme vert. C'est un goulet d'etranglement majeur entre la promesse du produit et sa portee reelle.

2. **Les revues de parties sont ephemeres.** L'analyse post-partie (qui prend 2 a 4 minutes pour une partie typique de 40 coups a profondeur 16) n'est pas sauvegardee. Fermer l'application ou revenir en mode jeu perd tout le travail d'analyse. Revoir la meme partie impose de relancer l'analyse complete. Pour un outil de progression, c'est une friction significative : le joueur devrait pouvoir revenir sur ses revues anterieures sans attendre.

3. **Le coaching manque de structure pedagogique.** Le mode coaching permet de jouer contre Stockfish a un Elo configurable avec analyse en temps reel, mais chaque partie est une partie libre sans objectif. Il n'y a pas de repertoire d'ouvertures, pas de positions thematiques, pas de progression structuree. Le joueur intermediaire qui veut travailler une ouverture specifique ou un type de finale n'a aucun outil dedie pour cela.

### Hypotheses prises

- **Utilisateur type.** Joueur regulier sur chess.com ou lichess, Elo 1200-2000, qui joue plusieurs parties par semaine et souhaite identifier ses patterns d'erreurs recurrents. Hypothese basee sur le systeme d'Elo configurable, les 8 themes couvrant les deux plateformes majeures, et la classification pedagogique des coups.
- **Environnement.** Principalement Linux (X11/Wayland). Le workflow de capture d'ecran via mss et l'overlay suggere un usage desktop.
- **Usage des revues.** Un joueur qui revoit ses parties le fait plusieurs fois --- d'abord a chaud apres la partie, puis en revisant ses erreurs recurrentes sur un corpus de parties. La persistance des revues est donc un besoin reel, pas un confort marginal.

---

## 2. Vision pour la suite

Le projet a complete sa boucle fonctionnelle fondamentale. **Le prochain cap n'est plus de construire des fonctionnalites manquantes, mais d'elargir la portee du produit et d'approfondir la valeur pedagogique.**

Trois axes se dessinent par ordre de priorite :

1. **Elargir la couverture de reconnaissance** --- tant que le pipeline live ne fonctionne qu'avec un seul theme, la fonctionnalite differenciante du projet reste theorique pour la majorite des utilisateurs potentiels. C'est un plafond de verre a briser avant toute autre evolution.
2. **Rendre les revues durables** --- persister les analyses post-partie pour que la valeur pedagogique survive a la session. C'est le pendant "revue" de ce que la persistance PGN a fait pour les parties : transformer un resultat ephemere en actif reutilisable.
3. **Structurer l'entrainement** --- introduire un mode d'entrainement cible par position ou par ouverture pour passer d'un outil d'analyse passive a un outil d'apprentissage actif.

Ces axes s'enchainent logiquement : le premier maximise l'acquisition de parties analysables, le deuxieme capitalise sur les analyses produites, le troisieme exploite le corpus pour un entrainement cible.

---

## 3. Backlog prioritaire

---

### Item 1 --- Pipeline de generation de templates multi-themes

**Nature :** Amelioration d'une fonctionnalite existante --- extension de la reconnaissance de pieces.

**Probleme adresse**

Le mode observation live est la fonctionnalite qui differencie ChessLab de tout autre GUI Stockfish. Mais cette fonctionnalite repose sur la reconnaissance de pieces par template matching, et les templates n'existent que pour un seul theme ("green") sur les huit detectes. Un utilisateur de chess.com avec le theme "wood" ou un joueur lichess avec le theme "blue" ne peut tout simplement pas utiliser le mode live. Le systeme de detection de theme fonctionne (il identifie correctement le theme utilise), mais il ne peut rien en faire faute de templates correspondants. C'est comme avoir un traducteur qui reconnait la langue parlee mais ne connait qu'une seule langue.

**Valeur attendue**

- Multiplie par huit la base d'utilisateurs potentiels du mode observation live.
- Rentabilise pleinement les six modules du pipeline de vision (capture, detection, splitting, reconnaissance, monitoring, live) qui representent ~40% du code de production.
- Elimine le principal frein a l'adoption : l'obligation de changer le theme de son site d'echecs pour utiliser ChessLab.
- Le script `_generate_placeholders.py` et l'existence de `extract_templates_auto.py` suggerent que ce besoin a deja ete anticipe dans l'architecture.

**User story**

> En tant que joueur utilisant chess.com ou lichess avec mon theme prefere, je veux que ChessLab reconnaisse les pieces sur mon echiquier quel que soit le theme visuel, afin de pouvoir utiliser le mode observation live sans changer mes preferences de site.

**Criteres d'acceptation**

1. Au moins cinq themes sur les huit declares dans `themes.json` disposent de templates operationnels (les cinq chess.com ou les trois lichess au minimum).
2. Un mecanisme de generation ou d'extraction de templates a partir d'une capture d'ecran d'une position connue est disponible et documente.
3. La reconnaissance de pieces avec les nouveaux templates atteint un taux de precision >= 95% sur une position de depart standard et >= 90% sur des positions de milieu de partie, mesure sur au moins trois themes differents.
4. Le systeme de detection de theme selectionne automatiquement le bon jeu de templates sans intervention utilisateur.
5. En cas de theme non supporte, un message explicite indique les themes disponibles et propose une procedure d'ajout.
6. Les templates generes sont inclus dans le package et ne necessitent aucune etape manuelle de la part de l'utilisateur final.

**Impact :** Tres eleve --- c'est le verrou qui empeche la fonctionnalite phare du produit d'atteindre son public.

**Complexite :** Moyenne --- l'architecture de reconnaissance est deja generique (theme-based dispatch). L'effort porte sur la creation d'un outil d'extraction fiable et la generation/validation des templates pour chaque theme. Le script `extract_templates_auto.py` existant est un point de depart.

**Raison de la priorite :** Sans couverture multi-themes, le mode observation live --- qui a necessite l'integration de six modules et constitue la proposition de valeur unique du projet --- ne fonctionne que pour une fraction des utilisateurs. C'est un multiplicateur d'impact a cout modere, le meilleur ratio valeur/effort du backlog.

**Risques et points de vigilance**

- La qualite des templates extraits depend de la resolution d'ecran et du rendu du navigateur. Prevoir une validation automatique (tester les templates extraits contre une position connue).
- Certains themes ont des pieces 3D ou des effets visuels (ombres, reflets) qui compliquent le template matching. Identifier ces cas et les documenter comme limitations connues plutot que de tenter un support partiel.
- La taille du package augmentera (24 templates x N themes). Impact negligeable (quelques centaines de Ko au total) mais a surveiller.
- Le theme "brown" de lichess utilise des pieces similaires au "green" de chess.com : verifier qu'il n'y a pas de confusion de templates entre plateformes.

---

### Item 2 --- Persistance des revues de parties

**Nature :** Amelioration d'une fonctionnalite existante --- persistance et reutilisabilite.

**Probleme adresse**

La revue post-partie est l'un des piliers pedagogiques de ChessLab. Mais aujourd'hui, chaque revue est calculee a la volee : 2 a 4 minutes d'analyse Stockfish pour une partie de 40 coups, puis le resultat disparait a la fermeture ou au retour en mode jeu. Un joueur qui veut revoir ses cinq dernieres parties pour identifier ses patterns d'erreurs doit relancer cinq analyses completes. C'est un obstacle concret a l'usage prevu du produit : la revue iterative et l'identification de tendances sur un corpus de parties.

Par ailleurs, l'absence de persistance empeche toute fonctionnalite future d'agregation (statistiques de progression, taux d'erreur par phase, tendances sur N parties) qui serait naturellement alimentee par un historique de revues sauvegardees.

**Valeur attendue**

- Elimine le temps d'attente de 2-4 minutes pour chaque revisite d'une partie deja analysee.
- Permet la consultation instantanee des revues anterieures depuis le panneau "Parties recentes".
- Cree le socle de donnees necessaire a toute fonctionnalite future de suivi de progression.
- Alignement architectural : le module `persistence` sauvegarde deja les parties (PGN) --- etendre ce pattern aux revues est naturel et coherent.

**User story**

> En tant que joueur souhaitant suivre ma progression, je veux que les revues de mes parties soient sauvegardees automatiquement, afin de pouvoir les consulter instantanement sans relancer l'analyse Stockfish a chaque fois.

**Criteres d'acceptation**

1. A la fin d'une revue post-partie, le resultat complet (classifications, evaluations, resume, moments critiques) est automatiquement sauvegarde dans `~/.chesslab/games/` en association avec le fichier PGN correspondant.
2. Le panneau "Parties recentes" indique visuellement quelles parties disposent d'une revue sauvegardee.
3. Ouvrir une revue sauvegardee affiche instantanement les resultats sans lancer d'analyse moteur.
4. L'utilisateur peut relancer une analyse (par exemple a une profondeur superieure) et ecraser la revue precedente.
5. Le format de stockage des revues est stable et versionne (un changement de structure ne doit pas rendre les anciennes revues illisibles).
6. La suppression d'une partie (PGN) supprime egalement sa revue associee.

**Impact :** Eleve --- transforme la revue d'une operation couteuse et ephemere en un actif consultable instantanement, condition prealable a toute fonctionnalite de suivi de progression.

**Complexite :** Faible a moyenne --- les structures de donnees (`GameReview`, `ReviewedMove`, `PhaseSummary`) sont deja des dataclasses immutables serialisables. L'effort porte sur le choix du format de stockage (JSON structure ou extension du PGN avec commentaires), la lecture/ecriture, et l'integration dans le panneau de parties recentes.

**Raison de la priorite :** Deuxieme car elle amplifie la valeur d'une fonctionnalite deja construite (la revue) a faible cout. Elle est aussi un enabler : sans persistance des revues, aucune fonctionnalite de suivi de progression ou d'agregation statistique n'est possible. C'est le meme raisonnement qui avait place la persistance PGN en prerequis de la revue dans l'analyse precedente.

**Risques et points de vigilance**

- Le format de stockage doit etre choisi avec soin. JSON est le plus simple et le plus flexible pour des structures complexes (evaluations, classifications, lignes de jeu). L'annotation PGN (commentaires dans le PGN) est plus interoperable mais limitee en structure. Privilegier JSON avec un fichier `.review.json` associe a chaque `.pgn`.
- La compatibilite ascendante est importante : prevoir un champ de version dans le fichier de revue pour gerer les evolutions futures de la structure `GameReview`.
- Le volume de donnees reste faible (quelques Ko par revue) --- pas de risque de performance.
- Attention a la coherence : si le PGN est modifie manuellement, la revue associee peut devenir incoherente. Detecter ce cas (hash du PGN) et proposer de relancer l'analyse.

---

### Item 3 --- Mode d'entrainement par positions ciblees

**Nature :** Nouvelle fonctionnalite --- valeur pedagogique.

**Probleme adresse**

Le coaching actuel est un mode libre : le joueur affronte Stockfish a un Elo choisi, sans objectif particulier. C'est utile pour la pratique, mais insuffisant pour la progression structuree. Un joueur intermediaire qui veut travailler la defense sicilienne, ameliorer ses fins de partie tour et pion, ou s'entrainer sur les positions ou il commet des erreurs recurrentes n'a aucun moyen de le faire dans ChessLab. Il doit jouer des parties completes en esperant tomber sur la situation qu'il veut travailler.

Or, les donnees pour un entrainement cible existent deja dans le systeme : les revues post-partie identifient les moments critiques (erreurs, gaffes) avec la position FEN exacte. Il suffit de permettre au joueur de rejouer a partir de ces positions pour creer une boucle d'apprentissage ciblee.

**Valeur attendue**

- Transforme ChessLab d'un outil d'analyse passive en un outil d'entrainement actif.
- Exploite les donnees deja produites par le module de revue (positions critiques, meilleur coup) pour creer de la valeur pedagogique supplementaire sans calcul additionnel.
- Repond au besoin le plus souvent exprime par les joueurs intermediaires : "Je sais que je fais des erreurs en milieu de partie, mais comment m'entrainer specifiquement dessus ?"
- Differencie ChessLab des simples analyseurs statiques en introduisant une dimension interactive de repetition ciblee.

**User story**

> En tant que joueur souhaitant corriger mes faiblesses, je veux pouvoir m'entrainer a partir des positions ou j'ai commis des erreurs dans mes parties precedentes, afin de travailler specifiquement les situations ou je prends de mauvaises decisions.

**Criteres d'acceptation**

1. Depuis la revue d'une partie, l'utilisateur peut cliquer sur un moment critique (erreur ou gaffe) pour lancer un entrainement a partir de cette position.
2. Le mode entrainement place la position sur l'echiquier et demande au joueur de trouver le meilleur coup (ou un coup acceptable).
3. Le coup joue par l'utilisateur est compare a l'evaluation Stockfish : si le coup est bon (perte <= 30 cp), la position est validee ; sinon, une explication est donnee (meilleur coup + evaluation).
4. L'utilisateur peut demander un indice avant de jouer (afficher le meilleur coup sans l'evaluation).
5. Apres validation ou correction, la partie continue contre Stockfish pendant au moins 5 coups pour voir les consequences du bon coup dans la position.
6. Un compteur de positions travaillees et de reussites est affiche pendant la session d'entrainement.

**Impact :** Eleve --- c'est la fonctionnalite qui fait passer ChessLab d'un outil d'analyse a un outil de progression active. La boucle revoir-identifier-s'entrainer-verifier est le modele pedagogique le plus efficace aux echecs.

**Complexite :** Moyenne --- le mode coaching (`GameCoach`) gere deja le jeu contre Stockfish a partir d'une position FEN arbitraire. L'effort porte sur l'interface de lancement depuis la revue, la logique de validation du coup utilisateur, et le flux de navigation entre revue et entrainement. Depend de l'Item 2 (persistance des revues) pour acceder aux positions critiques sans recalcul.

**Raison de la priorite :** Troisieme car elle depend des deux items precedents (templates multi-themes pour generer un corpus de parties significatif, persistance des revues pour acceder aux positions critiques). Mais c'est la fonctionnalite qui complete la trajectoire produit : capturer → analyser → sauvegarder → revoir → s'entrainer. Elle ferme la boucle d'apprentissage.

**Risques et points de vigilance**

- La definition de "coup acceptable" (seuil de perte en centipawns) doit etre calibree selon le niveau du joueur. Un seuil fixe de 30 cp est raisonnable en v1 mais pourrait etre adaptatif a terme.
- Certaines positions critiques sont des situations de mat tactique ou la seule reponse est un coup force unique --- l'exercice est alors trivial. Filtrer ces positions ou les marquer comme "position a un seul bon coup".
- L'experience utilisateur de transition entre mode revue et mode entrainement doit etre fluide : le joueur ne doit pas perdre le contexte de la partie qu'il etait en train de revoir.
- Le `GameCoach` utilise actuellement `chess.engine.play()` pour les reponses du moteur. Pour le mode entrainement, il faudra analyser le coup du joueur avec `chess.engine.analyse()` pour evaluer sa qualite plutot que pour produire un coup.

---

## 4. Recommandation strategique

**Lancer en premier : Item 1 --- Pipeline de generation de templates multi-themes.**

Ce choix repose sur trois arguments :

1. **Debloquer la proposition de valeur unique.** Le mode observation live est ce qui distingue ChessLab de tout autre outil de coaching Stockfish. Mais avec un seul theme supporte, cette fonctionnalite est inaccessible a la grande majorite des utilisateurs. Chaque theme ajoute elargit directement la base d'utilisateurs potentiels. C'est le levier d'impact le plus eleve par unite d'effort.

2. **Alimenter le pipeline en aval.** Plus de themes supportes signifie plus de parties observables automatiquement, donc plus de parties sauvegardees, plus de revues, et a terme plus de positions d'entrainement. Les Items 2 et 3 beneficient mecaniquement d'un Item 1 reussi. L'inverse n'est pas vrai : persister des revues ou s'entrainer sur des positions n'a qu'un impact marginal si le corpus de parties reste limite aux parties manuelles contre le moteur.

3. **Capitaliser sur l'existant.** L'architecture de reconnaissance est deja generique (dispatch par theme, manifeste JSON, repertoire de templates). Le systeme de detection de theme identifie deja les huit themes. Le script d'extraction de templates existe. L'effort est concentre sur la production et la validation des assets (templates), pas sur de nouvelles abstractions. C'est un travail ou le risque technique est faible et le resultat directement mesurable (taux de reconnaissance par theme).

**Benefice global de la trajectoire proposee :** En realisant ces trois items dans l'ordre, ChessLab passera d'un MVP fonctionnel mais contraint a un seul theme a une plateforme de coaching complete : capable de lire automatiquement une partie sur chess.com ou lichess quel que soit le theme, de la sauvegarder avec son analyse, de la revoir instantanement avec classification des erreurs, et de proposer un entrainement cible sur les positions problematiques. Cette trajectoire est a la fois la plus coherente (chaque item enrichit le suivant), la plus pragmatique (effort croissant, valeur immediate des le premier item) et la plus differenciante (aucun outil gratuit ne propose cette boucle complete).
