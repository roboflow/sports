from sports.common.video_tracking import build_video_tracking_session
from distance import _render_speed_distance_traces


def run_speed_and_distance(args, session=None):
    if session is None:
        session = build_video_tracking_session(args, need_homography=True)
    focus = getattr(args, "track_id", None)
    _render_speed_distance_traces(
        args, session,
        show_speed=True,
        focus_tid=int(focus) if focus is not None else None,
        append_end_card=False,
    )
