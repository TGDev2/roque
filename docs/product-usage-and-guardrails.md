# Product Usage and Guardrails

ChessLab is a training and review product.

The intended scope is:

- manual coaching on a board the user controls
- passive live observation where outside tools are explicitly allowed
- post-game review of completed games

ChessLab is not positioned as:

- live move assistance during active third-party games where external help is disallowed
- a way to bypass platform fair-play policies or event rules
- an engine overlay for rated or competitive play unless the relevant context explicitly permits it

## Journey framing

### Manual Coach

Use Manual Coach when you are entering moves yourself for study, rehearsal, or offline analysis. The user is driving the board state.

### Observe Live

Use Observe Live for passive board syncing in contexts such as study boards, broadcasts, your own analysis sessions, or any environment that explicitly allows outside tools. A technical PASS means the capture pipeline is compatible; it does not grant permission from a platform or event.

Before using Observe Live on chess.com, lichess, or any other third-party service, the user is responsible for checking the current tool-use and fair-play rules for that context.

### Review

Use Review after the game. Review is for learning from completed PGNs, classifying mistakes, and extracting lessons without affecting live play.

## Product guardrails

- Observe Live starts from declared setup profiles only.
- Observe Live requires a preflight PASS before launch.
- Observe Live requires an explicit acknowledgement that the session allows passive observation and outside tools.
- Review copy and launcher copy should keep describing review as post-game analysis.
- Manual Coach copy should keep describing the mode as self-entered training on a board the user controls.

## Priority filter

Use this rule when evaluating future features:

1. Does the feature strengthen training, review, or permitted passive observation?
2. Does the feature make engine-backed guidance easier during an active third-party game?
3. If the answer to 2 is yes, the feature is out of scope unless product strategy changes explicitly.
