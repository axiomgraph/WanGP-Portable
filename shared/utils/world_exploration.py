# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Interactive exploration of a world of the Worlds gallery, between the explorer (shared.gradio.world_explorer) and the world model. The
gallery opens an Exploration for a world file and queues a task of the world's model with that file as its custom guide and EXPLORE_PROMPT
as its prompt; the task finds the open Exploration and serves its requests until the viewer leaves (or the task is aborted): each move is
generated and its frames handed back to the viewer. An exploration task whose Exploration is gone (the viewer left before the task
started) ends at once.

Requests (next_request): {"keys", "text", "seconds", "steps", "memory"} for a move (the explorer's menus: the model's own units), {"render":
record} for the whole video explored so far, {"save": True} for the world as explored so far (a new world file), None when the viewer
left. Events the model publishes: "ready" (world loaded), "clip"
(frames, fps, stats), "rendered" (frames, fps, record, moves), "saved" (data: the world file's bytes, moves), "ended"."""
import os
import queue
import threading
import uuid

EXPLORE_PROMPT = "Explore this World"  # prompt of exploration tasks (with the world file as custom guide)
_open = {}  # world file path -> Exploration waiting for or served by a task
_lock = threading.Lock()


class Exploration:
    def __init__(self, world_path):
        self.world_path, self.token = world_path, uuid.uuid4().hex
        self.requests, self.events = queue.Queue(), queue.Queue()
        self.left = threading.Event()

    def move(self, keys, text, seconds, steps, memory):
        self.requests.put({"keys": keys, "text": text, "seconds": seconds, "steps": steps, "memory": memory})

    def render(self, record):
        """The whole video explored so far (the world's own video, then the moves), to record in the Video gallery or to replay."""
        self.requests.put({"render": record})

    def save(self):
        """The world as explored so far (its own segments, then the moves), as a new world file to explore or continue later."""
        self.requests.put({"save": True})

    def leave(self):
        self.left.set()
        self.requests.put(None)
        with _lock:
            if _open.get(_key(self.world_path)) is self:
                del _open[_key(self.world_path)]

    def next_request(self, timeout):
        """The next move, None when the viewer left; raises queue.Empty after timeout (the task checks its abort flag)."""
        return self.requests.get(timeout=timeout)

    def publish(self, kind, **data):
        self.events.put({"type": kind, **data})

    def next_event(self, timeout):
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None


def _key(world_path):
    """The same file whatever its spelling: the gallery's path can be relative, the task's custom guide is made absolute."""
    return os.path.normcase(os.path.abspath(world_path))


def open_exploration(world_path):
    """A new Exploration of a world file (an earlier one of the same file is closed)."""
    with _lock:
        previous = _open.get(_key(world_path))
        exploration = _open[_key(world_path)] = Exploration(world_path)
    if previous is not None:
        previous.leave()
    return exploration


def exploration_for(world_path):
    with _lock:
        return _open.get(_key(world_path))
