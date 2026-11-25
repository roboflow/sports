"""Detection class ids — matches ``examples/soccer/main.py`` and Inference football models."""

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Team ids (stabilized jersey clustering).
TEAM_NONE = -1
TEAM_LEFT = 0
TEAM_RIGHT = 1

# Role aliases used by possession and pass modules.
ROLE_BALL = BALL_CLASS_ID
ROLE_GOALKEEPER = GOALKEEPER_CLASS_ID
ROLE_PLAYER = PLAYER_CLASS_ID
ROLE_REFEREE = REFEREE_CLASS_ID
