"""Regression checks for empty player crops."""

import __future__

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_crop_filter_preserves_player_alignment():
    """Keep detections aligned with predictions in both soccer modes."""
    root = Path(__file__).resolve().parents[1]
    env = dict(
        np=np,
        torch=SimpleNamespace(no_grad=nullcontext, mean=lambda x, dim: x),
        sv=SimpleNamespace(cv2_to_pillow=lambda x: x),
        tqdm=lambda batches, desc: batches,
    )
    team = root / "sports/common/team.py"
    tree = ast.parse(team.read_text())
    tree.body = [
        node
        for node in tree.body
        if getattr(node, "name", "") in {"TeamClassifier", "create_batches"}
    ]
    exec(
        compile(tree, str(team), "exec", flags=__future__.annotations.compiler_flag),
        env,
    )

    class Inputs(dict):
        def to(self, device):
            return self

    class Embeddings:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    model = object.__new__(env["TeamClassifier"])
    model.device, model.batch_size = "cpu", 2
    model.processor = lambda images, return_tensors: Inputs(
        values=np.array([image.flat[0] for image in images])
    )
    model.features_model = lambda values: SimpleNamespace(
        last_hidden_state=Embeddings(values)
    )
    model.reducer = SimpleNamespace(transform=lambda data: data)
    model.cluster_model = SimpleNamespace(predict=lambda data: data)
    good = [np.full((1, 1, 3), value) for value in (0, 1)]
    empty = np.empty((0, 1, 3))
    assert model.predict(good).tolist() == [0, 1]
    assert model.predict([]).tolist() == []
    for operation in (
        lambda: model.predict([good[0], empty]),
        lambda: model.fit([empty]),
        lambda: model.fit([]),
    ):
        try:
            operation()
        except ValueError as error:
            assert "empty crop" in str(error)
        else:
            raise AssertionError("expected empty crop error")
    source = ast.parse((root / "examples/soccer/main.py").read_text())
    for name in ("run_team_classification", "run_radar"):
        fn = next(
            n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        frame_loop = next(
            n
            for n in fn.body
            if isinstance(n, ast.For)
            and any(
                isinstance(x, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "players" for t in x.targets
                )
                for x in n.body
            )
        )
        begin = next(
            i
            for i, n in enumerate(frame_loop.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "crops" for t in n.targets)
        )
        end = next(
            i
            for i, n in enumerate(frame_loop.body)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "players_team_id" for t in n.targets
            )
        )
        block = ast.Module(body=frame_loop.body[begin : end + 1], type_ignores=[])
        for crops, expected in (
            ([good[0], empty, good[1]], [10, 30]),
            ([empty], []),
            (good, [10, 20]),
        ):
            ids = np.array([10, 20, 30][: len(crops)])
            context = dict(
                np=np,
                frame=None,
                players=ids,
                get_crops=lambda frame, players: crops,
                team_classifier=model,
            )
            exec(compile(block, str(root / "examples/soccer/main.py"), "exec"), context)
            assert context["players"].tolist() == expected
            assert len(context["players_team_id"]) == len(expected)
    print("valid, mixed and all-empty crop checks passed in both soccer modes")


def test_missing_team_centroid_omits_goalkeepers():
    """Avoid assigning a goalkeeper when a team centroid is unavailable."""
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/soccer/main.py"
    tree = ast.parse(source.read_text())
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_goalkeepers_team_id"
    )
    resolver_module = ast.Module(body=[resolver], type_ignores=[])
    environment = {
        "np": np,
        "sv": SimpleNamespace(Position=SimpleNamespace(BOTTOM_CENTER=0)),
    }
    exec(
        compile(
            resolver_module,
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        environment,
    )
    for name in ("run_team_classification", "run_radar"):
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        guard = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and "players_team_id" in ast.unparse(node.test)
            and "goalkeepers" in ast.unparse(node)
        )
        module = ast.Module(body=[guard], type_ignores=[])
        for labels in (np.array([]), np.array([0]), np.array([1])):
            context = {
                "np": np,
                "players_team_id": labels,
                "goalkeepers": np.array([42]),
            }
            exec(compile(module, str(source), "exec"), context)
            assert context["goalkeepers"].tolist() == []
            assert (
                environment["resolve_goalkeepers_team_id"](
                    [], labels, context["goalkeepers"]
                ).tolist()
                == []
            )
        context = {
            "np": np,
            "players_team_id": np.array([0, 1]),
            "goalkeepers": np.array([42]),
        }
        exec(compile(module, str(source), "exec"), context)
        assert context["goalkeepers"].tolist() == [42]
